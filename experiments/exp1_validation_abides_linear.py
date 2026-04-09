# experiments/exp1_validation_abides_linear.py

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl.core.federated_server import FederatedServer

DATA_PATH = Path("data/exp1_day_dataset_abides.csv")
OUT_LOG = Path("data/exp1_fedavg_abides_linear_log.csv")

LAGS = 3
BASE_FEATS = ["mid_mean", "spread_mean", "vol_sum", "fills_sum", "quote_coverage"]


def make_loader(X, y, batch_size=16, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_local_linear(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    epochs: int,
    wd: float,
    device: str = "cpu",
) -> float:
    """
    Local training for linear regressor using SGD + weight decay (ridge-style).
    Returns last batch loss (for logging only).
    """
    model.train()
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), weight_decay=float(wd))
    loss_fn = nn.MSELoss()

    last_loss = 0.0
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

    return last_loss


@torch.no_grad()
def eval_rmse_unscaled(model, loader, y_mu, y_sigma, device="cpu") -> float:
    model.eval()
    model.to(device)

    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred_z = model(xb).squeeze(-1).cpu().numpy()
        preds.append(pred_z)
        ys.append(yb.numpy())

    pred_z = np.concatenate(preds)
    y_z = np.concatenate(ys)

    pred = pred_z * y_sigma + y_mu
    y = y_z * y_sigma + y_mu
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # --------------------------
    # Build ARX features
    # --------------------------
    ar_cols = [f"abides_price_lag{i}" for i in range(1, LAGS + 1)]
    arx_cols = ar_cols + [f"{c}_lag{i}" for c in BASE_FEATS for i in range(1, LAGS + 1)]

    missing = [c for c in arx_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    X = df[arx_cols].astype(float).to_numpy()
    y = df["y_next_day_abides"].astype(float).to_numpy()

    # --------------------------
    # Time split
    # --------------------------
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --------------------------
    # Standardize X
    # --------------------------
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # --------------------------
    # Standardize y (important)
    # --------------------------
    y_mu = float(np.mean(y_train))
    y_sigma = float(np.std(y_train))
    if y_sigma == 0:
        y_sigma = 1.0
    y_train_z = (y_train - y_mu) / y_sigma
    y_test_z = (y_test - y_mu) / y_sigma

    # --------------------------
    # Clients: split training days into K time chunks
    # --------------------------
    K = 10
    idx = np.arange(len(X_train))
    chunks = np.array_split(idx, K)
    clients = [(X_train[c], y_train_z[c]) for c in chunks]

    test_loader = make_loader(X_test, y_test_z, batch_size=32, shuffle=False)

    # --------------------------
    # Model + server (Exp2-style)
    # --------------------------
    d_in = X_train.shape[1]
    global_model = nn.Linear(d_in, 1, bias=True)
    server = FederatedServer(model=global_model, algorithm="fedavg")

    # --------------------------
    # Training settings
    # --------------------------
    rounds = 200
    clients_per_round = 10
    local_epochs = 5
    lr = 0.05
    wd = 1e-2  # ridge-ish regularisation (via weight decay)
    patience = 20
    device = "cpu"

    rng = np.random.default_rng(0)

    logs = []
    best = float("inf")
    best_round = None
    best_state = None
    no_improve = 0

    print(f"[INFO] rows={len(df)} train={len(X_train)} test={len(X_test)} features={d_in}")

    # --------------------------
    # FedAvg loop (via FederatedServer)
    # --------------------------
    for r in range(1, rounds + 1):
        chosen = rng.choice(K, size=clients_per_round, replace=False)

        global_weights = copy.deepcopy(server.server_state["weights"])
        client_updates = []

        for cid in chosen:
            Xm, ym = clients[cid]
            loader = make_loader(Xm, ym, batch_size=16, shuffle=True)

            client_model = nn.Linear(d_in, 1, bias=True)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            train_loss = train_local_linear(
                client_model, loader,
                lr=lr, epochs=local_epochs, wd=wd,
                device=device,
            )

            client_updates.append({
                "weights": copy.deepcopy(client_model.state_dict()),
                "n_samples": int(len(Xm)),
                "train_loss": float(train_loss),  # optional, harmless
            })

        server.aggregate(client_updates)

        rmse = eval_rmse_unscaled(server.model, test_loader, y_mu, y_sigma, device=device)
        logs.append({"round": r, "test_rmse": rmse})

        if r == 1 or r % 10 == 0:
            print(f"Round {r:03d} | test RMSE: {rmse:.6f}")

        if rmse < best:
            best = rmse
            best_round = r
            best_state = copy.deepcopy(server.model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EARLY STOP] no improvement for {patience} rounds")
                break

    if best_state is not None:
        server.model.load_state_dict(best_state)

    pd.DataFrame(logs).to_csv(OUT_LOG, index=False)
    print(f"[OK] wrote log to {OUT_LOG}")
    print(f"[BEST] round={best_round} rmse={best:.6f}")


if __name__ == "__main__":
    main()
# experiments/exp1_validation_abides_linear.py

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated_server import fedavg

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


def train_local_linear(model: nn.Module, loader: DataLoader, lr: float, epochs: int, wd: float, device="cpu"):
    model.train()
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


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

    ar_cols = [f"abides_price_lag{i}" for i in range(1, LAGS + 1)]
    arx_cols = ar_cols + [f"{c}_lag{i}" for c in BASE_FEATS for i in range(1, LAGS + 1)]

    missing = [c for c in arx_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    X = df[arx_cols].astype(float).to_numpy()
    y = df["y_next_day_abides"].astype(float).to_numpy()

    # time split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # standardize X
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # standardize y
    y_mu = float(np.mean(y_train))
    y_sigma = float(np.std(y_train))
    if y_sigma == 0:
        y_sigma = 1.0
    y_train_z = (y_train - y_mu) / y_sigma
    y_test_z = (y_test - y_mu) / y_sigma

    # clients: split training days into K time chunks (same as before)
    K = 10
    idx = np.arange(len(X_train))
    chunks = np.array_split(idx, K)
    clients = [(X_train[c], y_train_z[c]) for c in chunks]

    test_loader = make_loader(X_test, y_test_z, batch_size=32, shuffle=False)

    # model: linear regressor
    d_in = X_train.shape[1]
    global_model = nn.Linear(d_in, 1, bias=True)

    rounds = 200
    clients_per_round = 10
    local_epochs = 5
    lr = 0.05
    wd = 1e-2  # acts like ridge (L2)

    patience = 20
    no_improve = 0

    logs = []
    rng = np.random.default_rng(0)

    best = float("inf")
    best_round = None
    best_state = None

    print(f"[INFO] rows={len(df)} train={len(X_train)} test={len(X_test)} features={d_in}")

    for r in range(1, rounds + 1):
        chosen = rng.choice(K, size=clients_per_round, replace=False)

        local_states = []
        local_weights = []

        for cid in chosen:
            Xm, ym = clients[cid]
            loader = make_loader(Xm, ym, batch_size=16, shuffle=True)

            local_model = nn.Linear(d_in, 1, bias=True)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            train_local_linear(local_model, loader, lr=lr, epochs=local_epochs, wd=wd, device="cpu")

            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(Xm))

        w = np.array(local_weights, dtype=float)
        w = (w / w.sum()).tolist()

        global_model.load_state_dict(fedavg(local_states, w))

        rmse = eval_rmse_unscaled(global_model, test_loader, y_mu, y_sigma, device="cpu")
        logs.append({"round": r, "test_rmse": rmse})

        if rmse < best:
            best = rmse
            best_round = r
            best_state = copy.deepcopy(global_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if r % 10 == 0 or r == 1:
            print(f"Round {r:03d} | test RMSE: {rmse:.6f}")

        if no_improve >= patience:
            print(f"[EARLY STOP] no improvement for {patience} rounds")
            break

    if best_state is not None:
        global_model.load_state_dict(best_state)

    pd.DataFrame(logs).to_csv(OUT_LOG, index=False)
    print(f"[OK] wrote log to {OUT_LOG}")
    print(f"[BEST] round={best_round} rmse={best:.6f}")


if __name__ == "__main__":
    main()
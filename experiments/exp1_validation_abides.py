# experiments/exp1_validation_abides.py

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fl.core.fed_model import MLPRegressor
from src.fl.core.federated_client import train_local
from src.fl.core.federated_server import FederatedServer


DATA_PATH = Path("data/exp1_day_dataset_abides.csv")
OUT_LOG = Path("data/exp1_fedavg_abides_log.csv")

LAGS = 3

BASE_FEATS = [
    "mid_mean",
    "spread_mean",
    "vol_sum",
    "fills_sum",
    "quote_coverage",
]


def make_loader(X, y, batch_size=16, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def eval_rmse_unscaled(model, loader, y_mu, y_sigma, device="cpu") -> float:
    model.eval()
    model.to(device)

    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred_z = model(xb).cpu().numpy()
        preds.append(pred_z)
        ys.append(yb.numpy())

    pred_z = np.concatenate(preds)
    y_z = np.concatenate(ys)

    pred = pred_z * y_sigma + y_mu
    y = y_z * y_sigma + y_mu

    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    df = (
        pd.read_csv(DATA_PATH, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------
    # Build ARX features
    # --------------------------
    ar_cols = [f"abides_price_lag{i}" for i in range(1, LAGS + 1)]
    arx_cols = ar_cols + [
        f"{c}_lag{i}" for c in BASE_FEATS for i in range(1, LAGS + 1)
    ]

    missing = [c for c in arx_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

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

    print("[DEBUG] y_train mean/std:", y_mu, y_sigma)
    print("[DEBUG] y_train_z mean/std:",
          float(np.mean(y_train_z)),
          float(np.std(y_train_z)))

    # --------------------------
    # Create federated clients
    # --------------------------
    K = 10
    idx = np.arange(len(X_train))
    chunks = np.array_split(idx, K)
    clients = [(X_train[c], y_train_z[c]) for c in chunks]

    test_loader = make_loader(X_test, y_test_z, batch_size=32, shuffle=False)

    # --------------------------
    # Training settings
    # --------------------------
    clients_per_round = 10
    local_epochs = 2
    lr = 3e-3
    batch_size = 16
    rounds = 100
    patience = 10
    device = "cpu"

    global_model = MLPRegressor(d_in=X_train.shape[1])
    server = FederatedServer(model=global_model, algorithm="fedavg")

    logs = []
    rng = np.random.default_rng(0)

    print(
        f"[INFO] rows={len(df)} "
        f"train={len(X_train)} "
        f"test={len(X_test)} "
        f"features={X_train.shape[1]}"
    )

    best = float("inf")
    best_round = None
    best_state = None
    no_improve = 0

    # --------------------------
    # FedAvg loop
    # --------------------------
    for r in range(1, rounds + 1):
        chosen = rng.choice(K, size=clients_per_round, replace=False)

        client_updates = []

        for cid in chosen:
            Xm, ym = clients[cid]
            loader = make_loader(Xm, ym, batch_size=batch_size, shuffle=True)

            local_model = MLPRegressor(d_in=X_train.shape[1])
            # sync from server weights (not global_model directly)
            local_model.load_state_dict(copy.deepcopy(server.server_state["weights"]))

            train_local(
                local_model,
                loader,
                lr=lr,
                epochs=local_epochs,
                device=device,
            )

            client_updates.append({
                "weights": copy.deepcopy(local_model.state_dict()),
                "n_samples": int(len(Xm)),
            })

        # server-side FedAvg aggregation
        server.aggregate(client_updates)

        # global_model is the same object as server.model, but keep this explicit if you want
        global_model = server.model

        rmse = eval_rmse_unscaled(
            global_model,
            test_loader,
            y_mu,
            y_sigma,
            device=device,
        )

        print(f"Round {r:02d} | test RMSE: {rmse:.6f}")
        logs.append({"round": r, "test_rmse": rmse})

        if rmse < best:
            best = rmse
            best_round = r
            best_state = copy.deepcopy(global_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EARLY STOP] no improvement for {patience} rounds")
                break

    # restore best model
    if best_state is not None:
        global_model.load_state_dict(best_state)

    pd.DataFrame(logs).to_csv(OUT_LOG, index=False)
    print(f"[OK] wrote log to {OUT_LOG}")
    print(f"[BEST] round={best_round} rmse={best:.6f}")


if __name__ == "__main__":
    main()
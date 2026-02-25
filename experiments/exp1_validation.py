# experiments/exp1_validation.py
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fed_model import MLPRegressor
from src.federated_client import train_local
from src.federated_server import fedavg, eval_rmse

DATA_PATH = Path("data/exp1_day_dataset_controlled.csv")
OUT_LOG = Path("data/exp1_fedavg_controlled_log.csv")

FEATURES = [
    "mid_mean","mid_std","mid_p10","mid_p90",
    "spread_mean","spread_std",
    "vol_mean","vol_sum",
    "quote_coverage",
]
TARGET = "y_synth"

def make_loader(X, y, batch_size=16, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X = df[FEATURES].astype(float).to_numpy()
    y = df[TARGET].astype(float).to_numpy()

    # time split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # standardize using train stats
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # clients = 10 chunks of training days
    K = 10
    # sort days by volume to create heterogeneity
    order = np.argsort(df.iloc[:split]["vol_sum"].to_numpy())
    chunks = np.array_split(order, K)
    clients = [(X_train[c], y_train[c]) for c in chunks]

    test_loader = make_loader(X_test, y_test, batch_size=32, shuffle=False)

    # stable settings (as we discussed)
    rounds = 30
    clients_per_round = 10
    local_epochs = 5
    lr = 3e-3
    batch_size = 16
    device = "cpu"

    global_model = MLPRegressor(d_in=X_train.shape[1])

    logs = []
    rng = np.random.default_rng(0)

    for r in range(1, rounds + 1):
        chosen = rng.choice(K, size=clients_per_round, replace=False)

        local_states = []
        local_weights = []

        for cid in chosen:
            Xm, ym = clients[cid]
            loader = make_loader(Xm, ym, batch_size=batch_size, shuffle=True)

            local_model = MLPRegressor(d_in=X_train.shape[1])
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            train_local(local_model, loader, lr=lr, epochs=local_epochs, device=device)

            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(Xm))

        w = np.array(local_weights, dtype=float)
        w = (w / w.sum()).tolist()

        global_model.load_state_dict(fedavg(local_states, w))

        rmse = eval_rmse(global_model, test_loader, device=device)
        print(f"Round {r:02d} | test RMSE: {rmse:.6f}")
        logs.append({"round": r, "test_rmse": rmse})

    pd.DataFrame(logs).to_csv(OUT_LOG, index=False)
    print(f"[OK] wrote log to {OUT_LOG}")

if __name__ == "__main__":
    main()
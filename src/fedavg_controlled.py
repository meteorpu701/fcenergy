# src/fedavg_controlled.py
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path("data/exp1_day_dataset_controlled.csv")
OUT_LOG = Path("data/exp1_fedavg_controlled_log.csv")

FEATURES = [
    "mid_mean",
    "mid_std",
    "mid_p10",
    "mid_p90",
    "spread_mean",
    "spread_std",
    "vol_mean",
    "vol_sum",
    "quote_coverage",
]

TARGET = "y_synth"


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_local(model: nn.Module, loader: DataLoader, lr: float, epochs: int, device: str) -> float:
    model.train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    last_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
    return last_loss


@torch.no_grad()
def eval_rmse(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    model.to(device)
    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return float(np.sqrt(np.mean((preds - ys) ** 2)))


def fedavg(state_dicts, weights):
    # weighted average of model parameters
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        avg[k] = avg[k] * weights[0]
        for i in range(1, len(state_dicts)):
            avg[k] = avg[k] + state_dicts[i][k] * weights[i]
    return avg


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

    # create K clients by chunking training days
    K = 10
    idx = np.arange(len(X_train))
    chunks = np.array_split(idx, K)

    clients = []
    for c in chunks:
        clients.append((X_train[c], y_train[c]))

    test_loader = make_loader(X_test, y_test, batch_size=32, shuffle=False)

    device = "cpu"
    rounds = 30
    clients_per_round = 5
    local_epochs = 20
    lr = 1e-2
    batch_size = 16

    global_model = MLP(d_in=X_train.shape[1])

    print(f"[INFO] train_days={len(X_train)} test_days={len(X_test)} clients={K}")
    print("[INFO] starting FedAvg...")

    rng = np.random.default_rng(0)

    logs = []
    for r in range(1, rounds + 1):
        chosen = rng.choice(K, size=clients_per_round, replace=False)

        local_states = []
        local_weights = []

        for cid in chosen:
            Xm, ym = clients[cid]
            loader = make_loader(Xm, ym, batch_size=batch_size, shuffle=True)

            local_model = MLP(d_in=X_train.shape[1])
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            train_local(local_model, loader, lr=lr, epochs=local_epochs, device=device)

            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(Xm))

        # normalize weights
        w = np.array(local_weights, dtype=float)
        w = (w / w.sum()).tolist()

        new_state = fedavg(local_states, w)
        global_model.load_state_dict(new_state)

        rmse = eval_rmse(global_model, test_loader, device=device)
        print(f"Round {r:02d} | test RMSE: {rmse:.6f}")
        logs.append({"round": r, "test_rmse": rmse})

    print("[DONE] FedAvg finished.")
    pd.DataFrame(logs).to_csv(OUT_LOG, index=False)
    print(f"[OK] wrote log to {OUT_LOG}")

if __name__ == "__main__":
    main()
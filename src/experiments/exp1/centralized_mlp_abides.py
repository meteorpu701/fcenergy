# src/exp1_centralized_mlp_abides.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.fl.core.fed_model import MLPRegressor

DATA_PATH = Path("data/exp1_day_dataset_abides.csv")
LAGS = 3
BASE_FEATS = ["mid_mean", "spread_mean", "vol_sum", "fills_sum", "quote_coverage"]

def make_loader(X, y, batch_size=16, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def rmse_unscaled(model, loader, y_mu, y_sigma, device="cpu"):
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
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    ar_cols = [f"abides_price_lag{i}" for i in range(1, LAGS + 1)]
    arx_cols = ar_cols + [f"{c}_lag{i}" for c in BASE_FEATS for i in range(1, LAGS + 1)]

    X = df[arx_cols].astype(float).to_numpy()
    y = df["y_next_day_abides"].astype(float).to_numpy()

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # standardize X (train stats)
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # standardize y (train stats)
    y_mu = float(np.mean(y_train))
    y_sigma = float(np.std(y_train))
    if y_sigma == 0:
        y_sigma = 1.0
    y_train_z = (y_train - y_mu) / y_sigma
    y_test_z = (y_test - y_mu) / y_sigma

    train_loader = make_loader(X_train, y_train_z, batch_size=16, shuffle=True)
    test_loader = make_loader(X_test, y_test_z, batch_size=32, shuffle=False)

    device = "cpu"
    model = MLPRegressor(d_in=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.MSELoss()

    best = float("inf")
    best_ep = None
    for ep in range(1, 301):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        rmse = rmse_unscaled(model, test_loader, y_mu, y_sigma, device=device)
        if rmse < best:
            best = rmse
            best_ep = ep
        if ep % 25 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | test RMSE: {rmse:.6f}")

    print(f"[BEST] epoch={best_ep} rmse={best:.6f}")

if __name__ == "__main__":
    main()
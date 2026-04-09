# src/experiments/exp2b/exp2b_train_fl.py
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.fl.core.federated_server import FederatedServer
from src.fl.core.federated_client import (
    client_fit_fedavg,
    client_fit_fedprox,
    client_fit_scaffold,
)
from src.fl.core.fed_model import MLPRegressor


# ============================================================
# CONFIG
# ============================================================

REQUIRED_COLS = {"hub", "price", "target_next_price", "log_ret_next"}

DROP_COLS = {
    "hub", "date", "symbol", "features_file",
    "target_next_price", "target_next_date",
    "log_ret_next",  # target
    "price",         # used for implied-price eval, not as feature
}


# ============================================================
# ARGUMENTS
# ============================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Exp2b FL training (hub-as-client). LOHO by hub."
    )

    ap.add_argument("--dataset", default="data/exp2a_dataset.csv")
    ap.add_argument("--out", default="data/exp2b_fl_log.csv")
    ap.add_argument("--best_out", default=None,
                    help="Optional CSV path to write 1-row best summary.")

    # algorithm
    ap.add_argument(
        "--algo",
        default="fedavg",
        help="fedavg / fedprox / fednova / scaffold / krum"
             "(fedprox+scaffold are client-side; fednova is server-side normalized aggregation)",
    )
    ap.add_argument("--mu", type=float, default=0.01,
                    help="FedProx proximal strength (only used when --algo fedprox)")

    # LOHO
    ap.add_argument("--test_hub", default=None)
    ap.add_argument("--all_test", action="store_true")
    ap.add_argument("--min_test_rows", type=int, default=20,
                    help="Skip hub if test set smaller than this.")

    # training
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--clients_per_round", type=int, default=None)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)

    # evaluation
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=30)
    # Krum hyperparam
    ap.add_argument("--krum_f", type=int, default=0,
                help="Krum Byzantine count f. Must satisfy n_clients >= 2f+3 (with 3 hubs, f must be 0).")

    return ap.parse_args()


# ============================================================
# DATA HELPERS
# ============================================================

def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in DROP_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _fit_transform_train_only(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              feat_cols: List[str]) -> tuple[np.ndarray, np.ndarray]:
    X_train = train_df[feat_cols].to_numpy()
    X_test = test_df[feat_cols].to_numpy()

    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train.astype(float), X_test.astype(float)


# ============================================================
# METRICS
# ============================================================

def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


@torch.no_grad()
def _predict(model, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy().reshape(-1)
        preds.append(pred)
        ys.append(yb.detach().cpu().numpy().reshape(-1))

    return np.concatenate(preds), np.concatenate(ys)


# ============================================================
# BASELINES
# ============================================================

def _baseline_metrics(test_df: pd.DataFrame) -> dict:
    """
    Baselines for Option A:
      - return baseline: r_pred = 0
      - price baseline: p_next_pred = p_today
    """
    r_true = pd.to_numeric(test_df["log_ret_next"], errors="coerce").to_numpy(dtype=float).reshape(-1)
    p_today = pd.to_numeric(test_df["price"], errors="coerce").to_numpy(dtype=float).reshape(-1)
    p_next = pd.to_numeric(test_df["target_next_price"], errors="coerce").to_numpy(dtype=float).reshape(-1)

    m_r = np.isfinite(r_true)
    m_p = np.isfinite(p_today) & np.isfinite(p_next)

    r0 = np.zeros_like(r_true)

    out = {}
    out["baseline_rmse_ret"] = _rmse(r_true[m_r], r0[m_r]) if m_r.any() else float("nan")
    out["baseline_mae_ret"] = _mae(r_true[m_r], r0[m_r]) if m_r.any() else float("nan")

    out["baseline_rmse_price"] = _rmse(p_next[m_p], p_today[m_p]) if m_p.any() else float("nan")
    out["baseline_mae_price"] = _mae(p_next[m_p], p_today[m_p]) if m_p.any() else float("nan")
    out["baseline_test_rows"] = int(len(test_df))

    return out


# ============================================================
# FedNova + SCAFFOLD utilities
# ============================================================

@torch.no_grad()
def _compute_delta(local_w: Dict[str, torch.Tensor], global_w: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    d: Dict[str, torch.Tensor] = {}
    for k in global_w.keys():
        d[k] = local_w[k] - global_w[k]
    return d


def _zeros_like_state(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Make a 'control variate' dict with same keys as weights.
    Floats -> zeros_like; non-floats -> keep as-is (safe placeholder).
    """
    z: Dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            z[k] = torch.zeros_like(v)
        else:
            z[k] = v
    return z


# ============================================================
# ONE LOHO RUN
# ============================================================

def run_one_test_hub(df: pd.DataFrame,
                     test_hub: str,
                     args: argparse.Namespace) -> pd.DataFrame:

    algo = str(args.algo).strip().lower()
    if algo not in {"fedavg", "fedprox", "fednova", "scaffold", "zeno", "krum"}:
        raise ValueError(f"Unknown --algo {algo}. Use fedavg/fedprox/fednova/scaffold.")

    hubs = sorted(df["hub"].unique())
    n_hubs = len(hubs)
    if n_hubs < 2:
        raise ValueError(f"Need at least 2 hubs, got {hubs}")

    use_loho = (n_hubs >= 3)

    # If using LOHO, we need a valid test_hub.
    if use_loho:
        if test_hub not in hubs:
            raise ValueError(f"test_hub={test_hub} not in hubs={hubs}")
        train_hubs = [h for h in hubs if h != test_hub]
    else:
        # 2-hub mode: train uses BOTH hubs as clients
        train_hubs = hubs[:]  # e.g., [NBP, TTF]
        # test_hub is optional: if provided, we’ll report on it; otherwise pooled test.
        if test_hub is not None and test_hub not in hubs:
            raise ValueError(f"test_hub={test_hub} not in hubs={hubs}")
        if test_hub not in hubs:
            raise ValueError(f"test_hub={test_hub} not in hubs={hubs}")


    train_df = df[df["hub"].isin(train_hubs)].copy()
    test_df = df[df["hub"] == test_hub].copy()
    
    train_df = train_df.dropna(subset=list(REQUIRED_COLS)).copy()
    test_df = test_df.dropna(subset=list(REQUIRED_COLS)).copy()

    base = _baseline_metrics(test_df)

    if len(test_df) < int(args.min_test_rows):
        print(f"[SKIP] {test_hub} only {len(test_df)} rows (< min_test_rows={args.min_test_rows}).")
        return pd.DataFrame()

    feat_cols = _select_feature_columns(df)
    feat_cols = [c for c in feat_cols if train_df[c].notna().any()]
    if not feat_cols:
        raise ValueError("No usable feature columns (all NaN in training split).")

    X_train_all, X_test = _fit_transform_train_only(train_df, test_df, feat_cols)

    y_train_all = pd.to_numeric(train_df["log_ret_next"], errors="coerce").to_numpy(dtype=float).reshape(-1)
    y_test = pd.to_numeric(test_df["log_ret_next"], errors="coerce").to_numpy(dtype=float).reshape(-1)

    p_today = pd.to_numeric(test_df["price"], errors="coerce").to_numpy(dtype=float).reshape(-1)
    p_next_true = pd.to_numeric(test_df["target_next_price"], errors="coerce").to_numpy(dtype=float).reshape(-1)

    # Build client loaders (hub-as-client)
    train_df = train_df.reset_index(drop=True)
    clients: dict[str, DataLoader] = {}

    for h in train_hubs:
        idx = train_df.index[train_df["hub"] == h].to_numpy()
        X_h = X_train_all[idx]
        y_h = y_train_all[idx]
        clients[h] = _make_loader(
            X_h, y_h,
            batch_size=int(args.batch_size),
            shuffle=True,
        )

    test_loader = _make_loader(
        X_test, y_test,
        batch_size=int(args.batch_size),
        shuffle=False,
    )

    # Model + server
    d_in = X_train_all.shape[1]
    global_model = MLPRegressor(d_in)

    # Server algorithm mapping:
    # - fedprox is a client update rule; server aggregation stays fedavg
    # - fedavg/fednova/scaffold use server-side aggregators
    server_algo = "fedavg" if algo == "fedprox" else algo
    agg_kwargs = {}
    agg_kwargs = None
    if algo == "zeno":
        agg_kwargs = dict(
            keep_frac=float(getattr(args, "zeno_keep_frac", 0.67)),
            rho=float(getattr(args, "zeno_rho", 1e-3)),
            min_keep=int(getattr(args, "zeno_min_keep", 1)),
        )
    elif algo == "krum":
        agg_kwargs = {"f": int(args.krum_f)}

    server = FederatedServer(model=global_model, algorithm=server_algo, agg_kwargs=agg_kwargs)

    # RNG / determinism (lightweight)
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    clients_per_round = int(args.clients_per_round) if args.clients_per_round is not None else len(train_hubs)
    clients_per_round = min(clients_per_round, len(train_hubs))

    # SCAFFOLD state:
    # server.server_state["c"] = global control variate
    # client_c[h] = local control variate for hub h (persistent across rounds)
    # ---- SCAFFOLD state (named params only) ----
    # Initialize per-client local control variates (for scaffold only)
    # ----------------------------
# SCAFFOLD control variates
# ----------------------------
    c_locals = None
    if algo == "scaffold":
        # local control variate per client (hub)
        c0 = server.server_state["c"]  # created by FederatedServer when algorithm="scaffold"
        c_locals = {h: {k: v.clone() for k, v in c0.items()} for h in train_hubs}
    best = float("inf")
    best_state = None
    no_improve = 0
    logs: list[dict] = []

    print(f"[INFO] test_hub={test_hub} algo={algo} server_algo={server_algo} "
          f"train_rows={len(train_df)} test_rows={len(test_df)} features={d_in} "
          f"seed={args.seed} local_epochs={args.local_epochs}")

    for r in range(1, int(args.rounds) + 1):
        chosen = rng.choice(
            train_hubs,
            size=clients_per_round,
            replace=False
        )

        global_weights = copy.deepcopy(server.server_state["weights"])
        client_updates = []

        for h in chosen:
            model = MLPRegressor(d_in)

            if algo == "scaffold":
                update = client_fit_scaffold(
                    model=model,
                    loader=clients[h],
                    global_weights=global_weights,
                    c_global=server.server_state["c"],   # ✅ use server c
                    c_local=c_locals[h],
                    lr=float(args.lr),
                    epochs=int(args.local_epochs),
                    device="cpu",
                )
                c_locals[h] = update["c_i_new"]
                client_updates.append(update)
            
            elif algo == "zeno":
                update = client_fit_fedavg(
                    model=model,
                    loader=clients[h],
                    global_weights=global_weights,
                    lr=float(args.lr),
                    epochs=int(args.local_epochs),
                    device="cpu",
                )

            elif algo == "fedprox":
                update = client_fit_fedprox(
                    model=model,
                    loader=clients[h],
                    global_weights=global_weights,
                    lr=float(args.lr),
                    epochs=int(args.local_epochs),
                    mu=float(args.mu),
                    device="cpu",
                )
            else:
                # fedavg / fednova both use normal local training; server decides aggregation
                update = client_fit_fedavg(
                    model=model,
                    loader=clients[h],
                    global_weights=global_weights,
                    lr=float(args.lr),
                    epochs=int(args.local_epochs),
                    device="cpu",
                )

            # FedNova requires delta + n_steps (your updated client_fit_* already provides them)
            if algo == "fednova":
                if "n_steps" not in update:
                    update["n_steps"] = 1
                if "delta" not in update:
                    local_w = update.get("weights", model.state_dict())
                    update["delta"] = _compute_delta(local_w, global_weights)

            client_updates.append(update)
        server.aggregate(client_updates)

        if int(args.eval_every) > 1 and (r % int(args.eval_every) != 0):
            continue

        r_pred, r_true = _predict(server.model, test_loader, device="cpu")

        rmse_ret = _rmse(r_true, r_pred)
        mae_ret = _mae(r_true, r_pred)

        p_next_pred = p_today * np.exp(r_pred)
        mask = np.isfinite(p_next_pred) & np.isfinite(p_next_true)
        rmse_price = _rmse(p_next_true[mask], p_next_pred[mask]) if mask.any() else float("nan")
        mae_price = _mae(p_next_true[mask], p_next_pred[mask]) if mask.any() else float("nan")

        logs.append({
            # identifiers / knobs
            "test_hub": test_hub,
            "algo": algo,
            "server_algo": server_algo,
            "seed": int(args.seed),
            "round": int(r),
            "local_epochs": int(args.local_epochs),
            "clients_per_round": int(clients_per_round),
            "lr": float(args.lr),
            "mu": float(args.mu) if algo == "fedprox" else np.nan,

            # metrics
            "rmse_ret": float(rmse_ret),
            "mae_ret": float(mae_ret),
            "rmse_price_implied": float(rmse_price),
            "mae_price_implied": float(mae_price),

            # baselines (constant per hub)
            "baseline_rmse_ret": float(base["baseline_rmse_ret"]),
            "baseline_mae_ret": float(base["baseline_mae_ret"]),
            "baseline_rmse_price": float(base["baseline_rmse_price"]),
            "baseline_mae_price": float(base["baseline_mae_price"]),
            "baseline_test_rows": int(base["baseline_test_rows"]),
        })

        if r == 1 or r % 10 == 0:
            print(f"Round {r:03d} | rmse_ret={rmse_ret:.6f}")

        if rmse_ret < best:
            best = rmse_ret
            best_state = copy.deepcopy(server.model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if int(args.patience) and no_improve >= int(args.patience):
            print("[EARLY STOP]")
            break

    if best_state is not None:
        server.model.load_state_dict(best_state)

    out = pd.DataFrame(logs)

    if not out.empty:
        best_row = out.sort_values("rmse_ret").iloc[0]
        print(f"[BEST] {test_hub} algo={algo} local_epochs={int(best_row['local_epochs'])} "
              f"round={int(best_row['round'])} rmse_ret={float(best_row['rmse_ret']):.6f}")

        best_out = getattr(args, "best_out", None)
        if best_out:
            best_path = Path(best_out)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([best_row]).to_csv(best_path, index=False)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    args = _parse_args()

    df = pd.read_csv(args.dataset)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    for c in ["price", "target_next_price", "log_ret_next"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["hub"] = df["hub"].astype(str)
    df = df.dropna(subset=["hub", "price", "target_next_price", "log_ret_next"]).copy()

    hubs = sorted(df["hub"].unique())
    if not hubs:
        raise ValueError("No hubs found after cleaning.")

    results = []

    if args.all_test:
        for h in hubs:
            results.append(run_one_test_hub(df, h, args))
    else:
        test_hub = args.test_hub or hubs[-1]
        results.append(run_one_test_hub(df, test_hub, args))

    final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    print(f"[OK] wrote log -> {out_path} rows={len(final)}")


if __name__ == "__main__":
    main()
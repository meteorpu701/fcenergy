from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor


DATASET_DEFAULT = "data/exp2a_dataset.csv"
OUT_REPORT_DEFAULT = "data/exp2a_loho_report.csv"


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Exp2a LOHO training. Option A (main): predict next-day log return."
    )
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--out", default=OUT_REPORT_DEFAULT)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hidden", default="64,32", help="MLP hidden layers like '64,32'")
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument(
        "--use_baseline",
        action="store_true",
        help="Also compute baselines: (A) return baseline=0, (price) baseline next_price=today_price",
    )
    return ap.parse_args()


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    # identifiers / labels / metadata we must exclude
    drop_cols = {
        "hub", "date", "symbol", "features_file",
        "target_next_price", "target_next_date",
        "log_ret_next",  # <- target for option A, must not be in X
    }
    cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    args = _parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset: {ds_path}")

    df = pd.read_csv(ds_path)

    required = {"hub", "price", "target_next_price", "log_ret_next"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing required columns: {sorted(missing)}")

    # enforce numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["target_next_price"] = pd.to_numeric(df["target_next_price"], errors="coerce")
    df["log_ret_next"] = pd.to_numeric(df["log_ret_next"], errors="coerce")

    # drop rows where target or price info is missing
    df = df.dropna(subset=["hub", "price", "target_next_price", "log_ret_next"]).copy()

    hubs = sorted(df["hub"].unique().tolist())

    feat_cols_all = _select_feature_columns(df)
    if not feat_cols_all:
        raise ValueError("No numeric feature columns found. Check exp2a_build_dataset output.")

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    results = []

    for test_hub in hubs:
        train_df = df[df["hub"] != test_hub].copy()
        test_df = df[df["hub"] == test_hub].copy()

        # drop feature columns that are ALL NaN in TRAIN
        keep_cols = [c for c in feat_cols_all if train_df[c].notna().any()]
        if not keep_cols:
            raise ValueError(f"All feature columns are NaN in training split when holding out {test_hub}.")

        X_train = train_df[keep_cols].to_numpy()
        y_train = train_df["log_ret_next"].to_numpy()

        X_test = test_df[keep_cols].to_numpy()
        y_test = test_df["log_ret_next"].to_numpy()

        # For price-space evaluation we need today's and true next price
        p_today = test_df["price"].to_numpy()
        p_next_true = test_df["target_next_price"].to_numpy()

        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=hidden,
                random_state=args.seed,
                max_iter=args.max_iter,
                alpha=args.alpha,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.2,
            )),
        ])

        pipe.fit(X_train, y_train)
        r_pred = pipe.predict(X_test)

        # --- Return-space metrics (Option A main)
        rmse_ret = _rmse(y_test, r_pred)
        mae_ret = float(mean_absolute_error(y_test, r_pred))

        # --- Convert to implied next-day price and evaluate
        p_next_pred = p_today * np.exp(r_pred)

        # guard finite
        mask_p = np.isfinite(p_next_pred) & np.isfinite(p_next_true)
        rmse_price = _rmse(p_next_true[mask_p], p_next_pred[mask_p])
        mae_price = float(mean_absolute_error(p_next_true[mask_p], p_next_pred[mask_p]))

        rec = {
            "test_hub": test_hub,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "n_features_total": int(len(feat_cols_all)),
            "n_features_used": int(len(keep_cols)),
            "model": f"MLP(hidden={hidden},alpha={args.alpha}) + median_impute + zscore",
            "rmse_ret": rmse_ret,
            "mae_ret": mae_ret,
            "rmse_price_implied": rmse_price,
            "mae_price_implied": mae_price,
        }

        if args.use_baseline:
            # Baseline A: predict r=0 (random walk)
            r0 = np.zeros_like(y_test)
            rec["baseline_rmse_ret"] = _rmse(y_test, r0)
            rec["baseline_mae_ret"] = float(mean_absolute_error(y_test, r0))

            # Equivalent baseline in price space: predict next price = today's price
            p_base = p_today
            mask_b = np.isfinite(p_base) & np.isfinite(p_next_true)
            rec["baseline_rmse_price"] = _rmse(p_next_true[mask_b], p_base[mask_b])
            rec["baseline_mae_price"] = float(mean_absolute_error(p_next_true[mask_b], p_base[mask_b]))

        print(
            f"[LOHO-A] test_hub={test_hub} "
            f"rmse_ret={rmse_ret:.6f} mae_ret={mae_ret:.6f} "
            f"rmse_price={rmse_price:.4f} mae_price={mae_price:.4f} "
            f"features_used={len(keep_cols)}"
        )
        results.append(rec)

    out = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[OK] wrote report -> {out_path}")
    print("[SUMMARY] mean_rmse_ret=", float(out["rmse_ret"].mean()), "mean_mae_ret=", float(out["mae_ret"].mean()))
    print("[SUMMARY] mean_rmse_price_implied=", float(out["rmse_price_implied"].mean()),
          "mean_mae_price_implied=", float(out["mae_price_implied"].mean()))

    if args.use_baseline:
        print("[SUMMARY] mean_baseline_rmse_ret=", float(out["baseline_rmse_ret"].mean()),
              "mean_baseline_mae_ret=", float(out["baseline_mae_ret"].mean()))
        print("[SUMMARY] mean_baseline_rmse_price=", float(out["baseline_rmse_price"].mean()),
              "mean_baseline_mae_price=", float(out["baseline_mae_price"].mean()))


if __name__ == "__main__":
    main()
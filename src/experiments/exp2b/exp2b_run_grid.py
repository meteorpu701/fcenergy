# src/experiments/exp2b/exp2b_run_grid.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.experiments.exp2b.exp2b_train_fl import run_one_test_hub


def _parse_args():
    ap = argparse.ArgumentParser(description="Run Exp2b grid: algo × LOHO hubs × seeds, produce summary CSV.")
    ap.add_argument("--dataset", default="data/exp2a_dataset.csv")
    ap.add_argument("--out_summary", default="data/exp2b_grid_summary.csv")
    ap.add_argument("--out_dir", default="data/exp2b_logs", help="Where to store logs (per-round + best + aggregated).")

    ap.add_argument("--algos", default="fedavg,fedprox,scaffold,zeno", help="Comma-separated list (e.g. fedavg,fedprox)")
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx mu (only used for fedprox)")
    ap.add_argument("--min_test_rows", type=int, default=20)

    # seeds
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds (default: 0,1,2,3,4)")

    # training hyperparams
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--clients_per_round", type=int, default=None)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)

    # eval / early stop
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=30)
    # ---- Zeno hyperparameters ----
    ap.add_argument("--zeno_keep_frac", type=float, default=0.67)
    ap.add_argument("--zeno_rho", type=float, default=1e-3)
    ap.add_argument("--zeno_min_keep", type=int, default=1)
    ap.add_argument("--krum_f", type=int, default=0)

    return ap.parse_args()


def _parse_seeds(seeds_str: str) -> list[int]:
    seeds = []
    for s in str(seeds_str).split(","):
        s = s.strip()
        if not s:
            continue
        seeds.append(int(s))
    if not seeds:
        raise ValueError("No seeds provided. Use --seeds like '0,1,2,3,4'")
    return seeds


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main():
    args = _parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset: {ds_path}")

    df = pd.read_csv(ds_path)

    # minimal cleanup
    must_num = ["price", "target_next_price", "log_ret_next"]
    df = _ensure_numeric(df, must_num)
    df = df.dropna(subset=["hub", "price", "target_next_price", "log_ret_next"]).copy()

    hubs = sorted(df["hub"].unique().tolist())
    if not hubs:
        raise ValueError("No hubs in dataset after cleaning.")

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    seeds = _parse_seeds(args.seeds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_round_logs = []
    all_best_rows = []

    for algo in algos:
        for test_hub in hubs:
            # optional filter: require enough test rows (so we don't get the 1-row case)
            n_test = int((df["hub"] == test_hub).sum())
            if args.min_test_rows and n_test < int(args.min_test_rows):
                print(f"[SKIP] test_hub={test_hub} (test_rows={n_test} < min_test_rows={args.min_test_rows})")
                continue

            for seed in seeds:
                # Build a train-args-like Namespace expected by run_one_test_hub
                train_args = argparse.Namespace(
                    dataset=args.dataset,
                    out=str(out_dir / f"_tmp_{algo}_{test_hub}_{seed}.csv"),  # run_one_test_hub doesn't need this
                    algo=algo,
                    mu=float(args.mu),
                    test_hub=test_hub,
                    all_test=False,
                    rounds=int(args.rounds),
                    clients_per_round=args.clients_per_round,
                    local_epochs=int(args.local_epochs),
                    lr=float(args.lr),
                    batch_size=int(args.batch_size),
                    seed=int(seed),
                    eval_every=int(args.eval_every),
                    patience=int(args.patience),
                    min_test_rows=int(args.min_test_rows),
                    zeno_keep_frac=float(args.zeno_keep_frac),
                    zeno_rho=float(args.zeno_rho),
                    zeno_min_keep=int(args.zeno_min_keep),
                    krum_f=int(args.krum_f),
                )

                print(f"[RUN] algo={algo} test_hub={test_hub} seed={seed} train_rows={len(df)} test_rows={n_test}")
                run_df = run_one_test_hub(df=df, test_hub=test_hub, args=train_args)
                if run_df is None or run_df.empty:
                    print(f"[WARN] empty run_df for algo={algo} test_hub={test_hub} seed={seed}")
                    continue

                # ---- ensure numeric before choosing "best" ----
                for c in ["rmse_ret", "mae_ret", "rmse_price_implied",
                        "baseline_rmse_ret", "baseline_mae_ret",
                        "baseline_rmse_price", "baseline_mae_price",
                        "round"]:
                    if c in run_df.columns:
                        run_df[c] = pd.to_numeric(run_df[c], errors="coerce")

                # drop rows where rmse_ret is missing (cannot rank)
                run_df = run_df.dropna(subset=["rmse_ret"]).copy()
                if run_df.empty:
                    print(f"[WARN] all rmse_ret NaN for algo={algo} test_hub={test_hub} seed={seed}")
                    continue

                # Make sure seed exists as a column (helpful even if train_fl already logs it)
                if "seed" not in run_df.columns:
                    run_df = run_df.copy()
                    run_df["seed"] = seed

                all_round_logs.append(run_df)

                # Best row for this (algo, hub, seed)
                best = run_df.sort_values("rmse_ret").iloc[0].to_dict()
                best["seed"] = seed
                best["test_hub"] = test_hub
                best["algo"] = algo
                all_best_rows.append(best)

    if not all_best_rows:
        raise RuntimeError("No runs produced any results. Check filters (min_test_rows) and dataset content.")

    # ------------------------------------------------------------
    # Write raw logs
    # ------------------------------------------------------------
    round_logs_df = pd.concat(all_round_logs, ignore_index=True) if all_round_logs else pd.DataFrame()
    best_logs_df = pd.DataFrame(all_best_rows)

    round_logs_path = out_dir / "exp2b_round_logs_all.csv"
    best_logs_path = out_dir / "exp2b_best_logs_all.csv"

    if not round_logs_df.empty:
        round_logs_df.to_csv(round_logs_path, index=False)
        print(f"[OK] wrote per-round logs -> {round_logs_path} rows={len(round_logs_df)}")

    best_logs_df.to_csv(best_logs_path, index=False)
    print(f"[OK] wrote best-per-run logs -> {best_logs_path} rows={len(best_logs_df)}")

    # ------------------------------------------------------------
    # Aggregate across seeds: mean/std per (test_hub, algo)
    # ------------------------------------------------------------
    # Ensure expected numeric fields
    num_cols = [
        "rmse_ret", "mae_ret", "rmse_price_implied",
        "baseline_rmse_ret", "baseline_mae_ret", "baseline_rmse_price", "baseline_mae_price"
    ]
    best_logs_df = _ensure_numeric(best_logs_df, num_cols)

    # Derived metrics
    best_logs_df["delta_rmse_ret"] = best_logs_df["rmse_ret"] - best_logs_df["baseline_rmse_ret"]
    best_logs_df["ratio_rmse_ret"] = best_logs_df["rmse_ret"] / best_logs_df["baseline_rmse_ret"]

    agg_cols = [
        "rmse_ret", "mae_ret", "rmse_price_implied",
        "baseline_rmse_ret", "baseline_mae_ret", "baseline_rmse_price", "baseline_mae_price",
        "delta_rmse_ret", "ratio_rmse_ret",
    ]

    grouped = best_logs_df.groupby(["test_hub", "algo"], as_index=False)

    mean_df = grouped[agg_cols].mean(numeric_only=True)
    std_df = grouped[agg_cols].std(numeric_only=True).fillna(0.0)

    # Rename columns to *_mean and *_std and merge
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in agg_cols})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in agg_cols})

    summary = pd.merge(mean_df, std_df, on=["test_hub", "algo"], how="inner")

    # Keep a handy count of seeds actually present
    n_df = grouped["seed"].nunique().rename(columns={"seed": "n_seeds"})
    summary = pd.merge(summary, n_df, on=["test_hub", "algo"], how="left")

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"[OK] wrote aggregated summary -> {out_summary} rows={len(summary)}")


if __name__ == "__main__":
    main()
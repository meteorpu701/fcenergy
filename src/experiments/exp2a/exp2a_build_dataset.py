# src/exp2a_build_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np


FEATURES_DIR_DEFAULT = "data/simulated_trades"
PRICES_CSV_DEFAULT = "data/eu_hub_prices_exp2a.csv"
OUT_CSV_DEFAULT = "data/exp2a_dataset.csv"

AGENT_FEATURE_PATTERN = re.compile(r"agent_features_(?P<hub>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$")


def _parse_args():
    ap = argparse.ArgumentParser(description="Build Exp2a dataset: aggregated ABIDES features + next-day price target.")
    ap.add_argument("--features_dir", default=FEATURES_DIR_DEFAULT)
    ap.add_argument("--prices", default=PRICES_CSV_DEFAULT)
    ap.add_argument("--out", default=OUT_CSV_DEFAULT)
    ap.add_argument("--symbol", default="ABM", help="Symbol used in simulation (only for documentation).")
    return ap.parse_args()


def _aggregate_one_day(df: pd.DataFrame) -> dict:
    num_cols = ["best_bid", "best_ask", "mid", "spread", "n_fills", "transacted_volume", "avg_tx_price"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = {}
    out["n_agents"] = len(df)

    if "agent_type" in df.columns:
        counts = df["agent_type"].value_counts(dropna=False)
        for k, v in counts.items():
            out[f"n_type_{k}"] = int(v)

    def add_stats(prefix: str, series: pd.Series):
        out[f"{prefix}_mean"] = float(series.mean()) if series.notna().any() else None
        out[f"{prefix}_std"] = float(series.std()) if series.notna().any() else None
        out[f"{prefix}_min"] = float(series.min()) if series.notna().any() else None
        out[f"{prefix}_max"] = float(series.max()) if series.notna().any() else None
        out[f"{prefix}_median"] = float(series.median()) if series.notna().any() else None

    if "mid" in df.columns:
        add_stats("mid", df["mid"])
    if "spread" in df.columns:
        add_stats("spread", df["spread"])

    if "n_fills" in df.columns:
        out["fills_total"] = float(df["n_fills"].sum(skipna=True))
        out["fills_mean_per_agent"] = float(df["n_fills"].mean()) if df["n_fills"].notna().any() else None

    if "transacted_volume" in df.columns:
        out["vol_total"] = float(df["transacted_volume"].sum(skipna=True))
        out["vol_mean_per_agent"] = float(df["transacted_volume"].mean()) if df["transacted_volume"].notna().any() else None

    if "avg_tx_price" in df.columns and "transacted_volume" in df.columns:
        px = df["avg_tx_price"]
        w = df["transacted_volume"]
        mask = px.notna() & w.notna() & (w > 0)
        if mask.any():
            out["vwap_proxy"] = float((px[mask] * w[mask]).sum() / w[mask].sum())
        else:
            out["vwap_proxy"] = None

    return out

def add_exp2a_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "price" not in out.columns:
        raise KeyError("Dataset missing required column: price")
    if "target_next_price" not in out.columns:
        raise KeyError("Dataset missing required column: target_next_price")

    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["target_next_price"] = pd.to_numeric(out["target_next_price"], errors="coerce")

    mask = (out["price"] > 0) & (out["target_next_price"] > 0)
    out = out[mask].copy()

    out["log_ret_next"] = np.log(out["target_next_price"]) - np.log(out["price"])

    out = out.sort_values(["hub", "date"]).reset_index(drop=True)

    return out


def main():
    args = _parse_args()

    features_dir = Path(args.features_dir)
    prices_path = Path(args.prices)
    out_path = Path(args.out)

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices CSV: {prices_path}")
    if not features_dir.exists():
        raise FileNotFoundError(f"Missing features dir: {features_dir}")

    prices = pd.read_csv(prices_path, parse_dates=["date"])
    if not {"hub", "date", "price"}.issubset(prices.columns):
        raise KeyError("prices CSV must include columns: hub, date, price")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["price"]).copy()
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")

    prices = prices.sort_values(["hub", "date"]).copy()
    prices["target_next_price"] = prices.groupby("hub")["price"].shift(-1)
    prices["target_next_date"] = prices.groupby("hub")["date"].shift(-1)

    prices_keyed = prices.set_index(["hub", "date"])

    rows = []
    files = sorted(features_dir.glob("agent_features_*_*.csv"))
    for fp in files:
        m = AGENT_FEATURE_PATTERN.search(fp.name)
        if not m:
            continue
        hub = m.group("hub")
        date_str = m.group("date")

        if (hub, date_str) not in prices_keyed.index:
            continue

        df = pd.read_csv(fp)
        agg = _aggregate_one_day(df)

        rec = {
            "hub": hub,
            "date": date_str,
            "features_file": str(fp),
            "symbol": args.symbol,
        }
        rec.update(agg)

        p = prices_keyed.loc[(hub, date_str)]
        rec["price"] = float(p["price"])
        rec["target_next_price"] = None if pd.isna(p["target_next_price"]) else float(p["target_next_price"])
        rec["target_next_date"] = None if pd.isna(p["target_next_date"]) else str(p["target_next_date"])

        rows.append(rec)

    ds = pd.DataFrame(rows)

    before = len(ds)
    ds = ds.dropna(subset=["target_next_price"]).copy()
    after = len(ds)

    ds = add_exp2a_targets(ds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out_path, index=False)

    print(f"[OK] built dataset -> {out_path}")
    print(f"[INFO] rows(before_drop_last_day)={before} rows(after_drop_last_day)={after}")
    print(f"[INFO] rows(final_after_log_target)={len(ds)}")
    print(f"[INFO] hubs: {sorted(ds['hub'].unique().tolist())}")
    print(f"[INFO] columns: {len(ds.columns)}")


if __name__ == "__main__":
    main()
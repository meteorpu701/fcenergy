# src/build_exp1_day_dataset_abides.py

from __future__ import annotations

import pandas as pd
from pathlib import Path
from glob import glob

from src.common.utils import add_lag_features

IN_PATH = Path("data/exp1_dataset.csv")
EU_PRICE_PATH = Path("data/eu_hub_prices.csv")  # optional plausibility check
OUT_PATH = Path("data/exp1_day_dataset_abides.csv")

LAGS = 3
SCALE = 10_000.0  # must match abides_simulation.py


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"]).sort_values("date")

    # --- Create scaled columns if raw exist (for quote-based features only) ---
    for col in ["best_bid", "best_ask", "mid", "spread", "avg_tx_price"]:
        if col in df.columns and f"{col}_scaled" not in df.columns:
            df[f"{col}_scaled"] = pd.to_numeric(df[col], errors="coerce") / SCALE

    if "mid_scaled" not in df.columns and "best_bid_scaled" in df.columns and "best_ask_scaled" in df.columns:
        df["mid_scaled"] = (df["best_bid_scaled"] + df["best_ask_scaled"]) / 2.0
    if "spread_scaled" not in df.columns and "best_bid_scaled" in df.columns and "best_ask_scaled" in df.columns:
        df["spread_scaled"] = (df["best_ask_scaled"] - df["best_bid_scaled"])

    use_market_obs = ("mid_scaled" in df.columns) and df["mid_scaled"].notna().any()

    # -------------------------------
    # Day-level aggregation from quotes
    # -------------------------------
    day_dict = {}

    if use_market_obs:
        df_valid = df.dropna(subset=["mid_scaled"]).copy()
        g = df_valid.groupby("date", sort=True)

        day_dict.update({
            "mid_mean": g["mid_scaled"].mean(),
            "mid_std": g["mid_scaled"].std(),
        })

        if "spread_scaled" in df_valid.columns and df_valid["spread_scaled"].notna().any():
            day_dict.update({
                "spread_mean": g["spread_scaled"].mean(),
                "spread_std": g["spread_scaled"].std(),
            })

    day = pd.DataFrame(day_dict).reset_index()

    # --- Quote coverage (agent-level) ---
    if use_market_obs and "agent_id" in df.columns:
        n_agents_total = df.groupby("date")["agent_id"].size()
        n_agents_with_quotes = df.dropna(subset=["mid_scaled"]).groupby("date")["agent_id"].size()
        day["n_agents_total"] = day["date"].map(n_agents_total)
        day["n_agents_with_quotes"] = day["date"].map(n_agents_with_quotes)
        day["quote_coverage"] = day["n_agents_with_quotes"] / day["n_agents_total"]
    else:
        day["quote_coverage"] = 0.0

    # -----------------------------------------------------------
    # ABIDES oracle anchor + trade stats from day_summary files
    # -----------------------------------------------------------
    summ_files = sorted(glob("data/simulated_trades/day_summary_*.csv"))
    if not summ_files:
        raise SystemExit("No day_summary_*.csv found. Run: python -m src.sim.run_abides_days")

    summ = pd.concat([pd.read_csv(f) for f in summ_files], ignore_index=True)
    summ["date"] = pd.to_datetime(summ["date"], errors="coerce")
    summ = summ.dropna(subset=["date"]).sort_values("date")

    # Oracle anchor (price level)
    if "r_bar_scaled" in summ.columns and summ["r_bar_scaled"].notna().any():
        summ["abides_price"] = pd.to_numeric(summ["r_bar_scaled"], errors="coerce")
    elif "r_bar" in summ.columns and summ["r_bar"].notna().any():
        summ["abides_price"] = pd.to_numeric(summ["r_bar"], errors="coerce") / SCALE
    else:
        raise SystemExit("day_summary files missing r_bar_scaled/r_bar columns.")

    summ = summ.dropna(subset=["abides_price"]).copy()

    # Trade stats from exchange/order book
    # (Your day_summary now has fills_buy/sell/total and vol_buy/sell/total.)
    # Older day_summary files may have NaNs for these, so we coerce + fill later.
    for c in ["fills_total", "vol_total", "fills_buy", "fills_sell", "vol_buy", "vol_sell", "last_trade"]:
        if c in summ.columns:
            summ[c] = pd.to_numeric(summ[c], errors="coerce")

    # Stable columns
    if "fills_total" not in summ.columns:
        summ["fills_total"] = 0.0
    if "vol_total" not in summ.columns:
        summ["vol_total"] = 0.0

    # Prefer total stats as daily aggregates
    summ["fills_sum"] = summ["fills_total"].fillna(0.0)
    summ["vol_sum"] = summ["vol_total"].fillna(0.0)

    keep_cols = ["date", "abides_price", "fills_sum", "vol_sum"]
    for c in ["fills_buy", "fills_sell", "vol_buy", "vol_sell", "fills_total", "vol_total", "last_trade"]:
        if c in summ.columns:
            keep_cols.append(c)
            
    # --- IMPORTANT: restrict to the hub/price file used for Exp1 ---
    summ["hub"] = summ.get("hub", "TTF").astype(str).str.strip().str.upper()

    # If you want to anchor to the short 60-day CSV:
    summ = summ[summ["hub"] == "TTF"].copy()
    summ = summ[summ["prices_csv"].astype(str) == "data/eu_hub_prices.csv"].copy()

    # Prefer rows that actually have trade stats (fills_total not null), then keep last per date
    summ["fills_total"] = pd.to_numeric(summ.get("fills_total"), errors="coerce")
    summ["_has_trades"] = summ["fills_total"].notna().astype(int)

    summ = (
        summ.sort_values(["date", "_has_trades"])   # trades last
            .drop_duplicates(subset=["date"], keep="last")
            .drop(columns=["_has_trades"])
    )

    # Merge oracle+trade stats onto day features
    out = (
        day.merge(summ[keep_cols], on="date", how="inner")
           .sort_values("date")
           .reset_index(drop=True)
    )

    # Derive per-agent mean proxies (optional)
    if "n_agents_total" in out.columns and (pd.to_numeric(out["n_agents_total"], errors="coerce") > 0).any():
        denom = pd.to_numeric(out["n_agents_total"], errors="coerce").replace(0, pd.NA)
        out["fills_mean"] = pd.to_numeric(out["fills_sum"], errors="coerce") / denom
        out["vol_mean"] = pd.to_numeric(out["vol_sum"], errors="coerce") / denom
    else:
        out["fills_mean"] = pd.NA
        out["vol_mean"] = pd.NA

    # target = next day oracle anchor
    out["y_next_day_abides"] = out["abides_price"].shift(-1)
    out = out.dropna(subset=["y_next_day_abides"]).reset_index(drop=True)

    # --- Optional: attach TTF for plausibility check only ---
    if EU_PRICE_PATH.exists():
        prices = pd.read_csv(EU_PRICE_PATH, parse_dates=["date"])
        if "hub" in prices.columns:
            prices["hub"] = prices["hub"].astype(str).str.strip().str.upper()
            ttf = prices[prices["hub"] == "TTF"].sort_values("date").copy()
            ttf["ttf_price"] = pd.to_numeric(ttf["price"], errors="coerce")
            ttf = ttf.dropna(subset=["ttf_price"])
            out = out.merge(ttf[["date", "ttf_price"]], on="date", how="left")

    # -----------------------------------------------------------
    # Add AR lags (ROBUST): only lag features that have real values
    # -----------------------------------------------------------
    base_feature_cols = ["fills_sum", "fills_mean", "vol_sum", "vol_mean", "quote_coverage", "abides_price"]
    for c in ["mid_mean", "mid_std", "spread_mean", "spread_std"]:
        if c in out.columns and out[c].notna().any():
            base_feature_cols.append(c)

    # final safety filter: keep only columns present + with some non-NaN
    base_feature_cols = [c for c in base_feature_cols if (c in out.columns) and out[c].notna().any()]

    out = add_lag_features(out, cols_to_lag=base_feature_cols, max_lag=LAGS, group_cols=None, time_col="date")

    required = [f"{c}_lag{L}" for c in base_feature_cols for L in range(1, LAGS + 1)]
    out = out.dropna(subset=required).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] wrote {OUT_PATH} rows={len(out)} cols={len(out.columns)} | use_market_obs={use_market_obs}")
    print(f"[INFO] lagged_features={base_feature_cols}")


if __name__ == "__main__":
    main()
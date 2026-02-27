# src/build_exp1_day_dataset_abides.py
import pandas as pd
from pathlib import Path
from glob import glob
from src.utils import add_lag_features

IN_PATH = Path("data/exp1_dataset.csv")
EU_PRICE_PATH = Path("data/eu_hub_prices.csv")  # optional plausibility check
OUT_PATH = Path("data/exp1_day_dataset_abides.csv")

LAGS = 3
SCALE = 10_000.0  # must match abides_simulation.py

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"]).sort_values("date")

    # --- Create scaled columns if raw exist (for features only) ---
    for col in ["best_bid", "best_ask", "mid", "spread", "avg_tx_price"]:
        if col in df.columns and f"{col}_scaled" not in df.columns:
            df[f"{col}_scaled"] = pd.to_numeric(df[col], errors="coerce") / SCALE

    if "mid_scaled" not in df.columns and "best_bid_scaled" in df.columns and "best_ask_scaled" in df.columns:
        df["mid_scaled"] = (df["best_bid_scaled"] + df["best_ask_scaled"]) / 2.0
    if "spread_scaled" not in df.columns and "best_bid_scaled" in df.columns and "best_ask_scaled" in df.columns:
        df["spread_scaled"] = (df["best_ask_scaled"] - df["best_bid_scaled"])

    use_market_obs = ("mid_scaled" in df.columns) and df["mid_scaled"].notna().any()

    # --- Day-level aggregation (features) ---
    by_day = df.groupby("date", sort=True)

    day_dict = {
        "fills_sum": by_day["n_fills"].sum(),
        "fills_mean": by_day["n_fills"].mean(),
        "vol_sum": by_day["transacted_volume"].sum(),
        "vol_mean": by_day["transacted_volume"].mean(),
    }

    if use_market_obs:
        df_valid = df.dropna(subset=["mid_scaled"])
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

    if use_market_obs:
        n_agents_total = df.groupby("date")["agent_id"].size()
        n_agents_with_quotes = df.dropna(subset=["mid_scaled"]).groupby("date")["agent_id"].size()
        day["n_agents_total"] = day["date"].map(n_agents_total)
        day["n_agents_with_quotes"] = day["date"].map(n_agents_with_quotes)
        day["quote_coverage"] = day["n_agents_with_quotes"] / day["n_agents_total"]
    else:
        day["quote_coverage"] = 0.0

    # --- ABIDES-backed price target from day_summary files (REQUIRED) ---
    summ_files = sorted(glob("data/simulated_trades/day_summary_*.csv"))
    if not summ_files:
        raise SystemExit("No day_summary_*.csv found. Run: python3 -m src.run_abides_days")

    summ = pd.concat([pd.read_csv(f) for f in summ_files], ignore_index=True)
    summ["date"] = pd.to_datetime(summ["date"], errors="coerce")
    summ = summ.dropna(subset=["date"]).sort_values("date")

    if "r_bar_scaled" in summ.columns and summ["r_bar_scaled"].notna().any():
        summ["abides_price"] = pd.to_numeric(summ["r_bar_scaled"], errors="coerce")
    elif "r_bar" in summ.columns and summ["r_bar"].notna().any():
        summ["abides_price"] = pd.to_numeric(summ["r_bar"], errors="coerce") / SCALE
    else:
        raise SystemExit("day_summary files missing r_bar_scaled/r_bar columns.")

    summ = summ.dropna(subset=["abides_price"])

    # merge oracle anchor onto day features
    out = day.merge(summ[["date", "abides_price"]], on="date", how="inner").sort_values("date").reset_index(drop=True)

    # target = next day oracle anchor
    out["y_next_day_abides"] = out["abides_price"].shift(-1)
    out = out.dropna(subset=["y_next_day_abides"]).reset_index(drop=True)

    # --- Optional: attach TTF for plausibility check only ---
    if EU_PRICE_PATH.exists():
        prices = pd.read_csv(EU_PRICE_PATH, parse_dates=["date"])
        ttf = prices[prices["hub"] == "TTF"].sort_values("date").copy()
        ttf["ttf_price"] = pd.to_numeric(ttf["price"], errors="coerce")
        ttf = ttf.dropna(subset=["ttf_price"])
        out = out.merge(ttf[["date", "ttf_price"]], on="date", how="left")

    # --- Add AR lags ---
    base_feature_cols = ["fills_sum", "fills_mean", "vol_sum", "vol_mean", "quote_coverage", "abides_price"]
    for c in ["mid_mean", "mid_std", "spread_mean", "spread_std"]:
        if c in out.columns:
            base_feature_cols.append(c)

    out = add_lag_features(out, cols_to_lag=base_feature_cols, max_lag=LAGS, group_cols=None, time_col="date")

    required = [f"{c}_lag{L}" for c in base_feature_cols for L in range(1, LAGS + 1)]
    out = out.dropna(subset=required).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] wrote {OUT_PATH} rows={len(out)} cols={len(out.columns)} | use_market_obs={use_market_obs}")

if __name__ == "__main__":
    main()
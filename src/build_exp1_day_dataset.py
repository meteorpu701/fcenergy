# src/build_exp1_day_dataset.py
import pandas as pd
from pathlib import Path
from src.utils import add_lag_features

IN_PATH = Path("data/exp1_dataset.csv")       # agent-level rows
EU_PRICE_PATH = Path("data/eu_hub_prices.csv")
OUT_PATH = Path("data/exp1_day_dataset.csv")  # day-level ARX dataset

LAGS = 3
SCALE = 10000.0  # ABIDES integer-ish price scale (e.g., 100000)

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"]).sort_values("date")

    # --- Create scaled columns if raw exist ---
    for col in ["best_bid", "best_ask", "mid", "spread", "avg_tx_price"]:
        if col in df.columns and f"{col}_scaled" not in df.columns:
            df[f"{col}_scaled"] = pd.to_numeric(df[col], errors="coerce") / SCALE

    # If mid_scaled missing but bid/ask exist, compute it
    if "mid_scaled" not in df.columns and "best_bid_scaled" in df.columns and "best_ask_scaled" in df.columns:
        df["mid_scaled"] = (df["best_bid_scaled"] + df["best_ask_scaled"]) / 2.0

    # If spread_scaled missing but bid/ask exist, compute it
    if "spread_scaled" not in df.columns and "best_bid_scaled" in df.columns and "best_ask_scaled" in df.columns:
        df["spread_scaled"] = (df["best_ask_scaled"] - df["best_bid_scaled"])

    use_market_obs = ("mid_scaled" in df.columns) and df["mid_scaled"].notna().any()

    # --- Day-level aggregation (market-state representation) ---
    # Always-available (firm-observable proxies)
    by_day = df.groupby("date", sort=True)

    day_dict = {
        "fills_sum": by_day["n_fills"].sum(),
        "fills_mean": by_day["n_fills"].mean(),
        "vol_sum": by_day["transacted_volume"].sum(),
        "vol_mean": by_day["transacted_volume"].mean(),
    }

    # Optional market observables (if present)
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

    # Quote coverage: fraction of agents with a quote (only meaningful if market obs exist)
    if use_market_obs:
        n_agents_total = df.groupby("date")["agent_id"].size()
        n_agents_with_quotes = df.dropna(subset=["mid_scaled"]).groupby("date")["agent_id"].size()
        day["n_agents_total"] = day["date"].map(n_agents_total)
        day["n_agents_with_quotes"] = day["date"].map(n_agents_with_quotes)
        day["quote_coverage"] = day["n_agents_with_quotes"] / day["n_agents_total"]
    else:
        day["quote_coverage"] = 0.0

    # --- Attach target from EU hub prices (Exp1 = TTF only) ---
    prices = pd.read_csv(EU_PRICE_PATH, parse_dates=["date"])
    prices = prices[prices["hub"] == "TTF"].sort_values("date").copy()
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["price"])

    prices["y_next_day"] = prices["price"].shift(-1)
    prices = prices.dropna(subset=["y_next_day"])

    # Merge on date intersection
    out = day.merge(prices[["date", "price", "y_next_day"]], on="date", how="inner")

    # --- Add ARX lags ---
    base_feature_cols = [
        "fills_sum", "fills_mean", "vol_sum", "vol_mean", "quote_coverage", "price"
    ]
    # only lag columns that exist (mid/spread might be absent)
    for c in ["mid_mean", "mid_std", "spread_mean", "spread_std"]:
        if c in out.columns:
            base_feature_cols.append(c)

    out = add_lag_features(out, cols_to_lag=base_feature_cols, max_lag=LAGS, group_cols=None, time_col="date")

    # Drop rows without full history
    required = [f"{c}_lag{L}" for c in base_feature_cols for L in range(1, LAGS + 1)]
    out = out.dropna(subset=required).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] wrote {OUT_PATH} rows={len(out)} cols={len(out.columns)} | use_market_obs={use_market_obs}")

if __name__ == "__main__":
    main()
# src/build_exp1_day_dataset.py
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/exp1_dataset.csv")
OUT_PATH = Path("data/exp1_day_dataset.csv")

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    # denominator: total agents per day
    n_agents_total = df.groupby("date")["agent_id"].size()  # <-- FIXED

    # only rows with mid available
    df_valid = df.dropna(subset=["mid_scaled"])
    g = df_valid.groupby("date")

    # numerator: agents with quotes per day
    n_agents_with_quotes = df_valid.groupby("date")["agent_id"].size()  # <-- FIXED

    day = pd.DataFrame({
        "mid_mean": g["mid_scaled"].mean(),
        "mid_std": g["mid_scaled"].std(),
        "mid_p10": g["mid_scaled"].quantile(0.10),
        "mid_p90": g["mid_scaled"].quantile(0.90),

        "spread_mean": g["spread_scaled"].mean(),
        "spread_std": g["spread_scaled"].std(),

        "vol_mean": g["transacted_volume"].mean(),
        "vol_sum": g["transacted_volume"].sum(),

        "n_agents_with_quotes": n_agents_with_quotes,
    })

    day["n_agents_total"] = n_agents_total
    day["quote_coverage"] = day["n_agents_with_quotes"] / day["n_agents_total"]

    day = day.reset_index()

    hub = df[["date", "price", "y_next_day"]].drop_duplicates("date")
    out = day.merge(hub, on="date", how="left").dropna(subset=["y_next_day"])

    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] Wrote {OUT_PATH} rows={len(out)} cols={len(out.columns)}")

if __name__ == "__main__":
    main()
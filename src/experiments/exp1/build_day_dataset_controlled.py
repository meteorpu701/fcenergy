# src/build_exp1_day_dataset_controlled.py
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("data/exp1_day_dataset.csv")
OUT_PATH = Path("data/exp1_day_dataset_controlled.csv")

def main():
    df = pd.read_csv(IN_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df["spread_mean"] = df["spread_mean"].fillna(df["spread_mean"].median())
    df["mid_std"] = df["mid_std"].fillna(0.0)
    df["vol_sum"] = df["vol_sum"].fillna(0.0)
    df["quote_coverage"] = df["quote_coverage"].fillna(0.0)

    vol_scaled = np.log1p(df["vol_sum"].astype(float))

    y = (
        0.7 * df["mid_mean"].astype(float)
        + 0.2 * df["spread_mean"].astype(float)
        + 0.1 * vol_scaled
        + 0.05 * df["quote_coverage"].astype(float)
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(loc=0.0, scale=0.01, size=len(df)) 
    df["y_synth"] = y + noise

    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] Wrote {OUT_PATH} rows={len(df)}")

if __name__ == "__main__":
    main()
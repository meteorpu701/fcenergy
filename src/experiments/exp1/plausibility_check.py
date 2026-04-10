# src/exp1_plausibility_check.py
import pandas as pd
from pathlib import Path

DATA = Path("data/exp1_day_dataset_abides.csv")

def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    if "ttf_price" not in df.columns or df["ttf_price"].isna().all():
        print("[WARN] No ttf_price column available (or all NaN).")
        return

    sub = df.dropna(subset=["abides_price", "ttf_price"]).copy()
    if len(sub) < 5:
        print("[WARN] Not enough overlapping rows for correlation.")
        return

    corr = sub["abides_price"].corr(sub["ttf_price"])
    print(f"[OK] overlap rows={len(sub)} corr(abides_price, ttf_price)={corr:.4f}")
    print(sub[["date","abides_price","ttf_price"]].head(10))

if __name__ == "__main__":
    main()
# src/run_abides_days.py

from pathlib import Path
import pandas as pd
from src.abides_simulation import run_and_save_agent_features

EU_PRICES = Path("data/eu_hub_prices.csv")

def main():
    df = pd.read_csv(EU_PRICES, parse_dates=["date"])
    ttf = df[df["hub"] == "TTF"].sort_values("date")

    dates = [d.strftime("%Y-%m-%d") for d in ttf["date"].tolist()]

    print(f"[INFO] Simulating {len(dates)} TTF days...")

    for d in dates:
        print(f"[RUN] {d}")
        run_and_save_agent_features(d)

if __name__ == "__main__":
    main()
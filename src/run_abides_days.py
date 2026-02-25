# src/run_abides_days.py
import pandas as pd
from pathlib import Path
from src.abides_simulation import run_and_save_agent_features

def main():
    hub = pd.read_csv("data/hub_prices_processed.csv", parse_dates=["date"]).sort_values("date")
    N_DAYS = 60
    days = hub["date"].dt.strftime("%Y-%m-%d").iloc[:N_DAYS]

    for d in days:
        out = Path("data/simulated_trades") / f"agent_features_{d}.csv"
        if out.exists():
            print(f"[SKIP] {d} already exists")
            continue
        print(f"[RUN] {d}")
        run_and_save_agent_features(d)

if __name__ == "__main__":
    main()
# src/build_exp1_dataset.py
from pathlib import Path
import pandas as pd

SIM_DIR = Path("data/simulated_trades")
HUB_PATH = Path("data/hub_prices_processed.csv")
OUT_PATH = Path("data/exp1_dataset.csv")

SCALE = 10_000 

def main():
    # 1) load all agent feature files
    files = sorted(SIM_DIR.glob("agent_features_*.csv"))
    if not files:
        raise RuntimeError("No agent_features_*.csv found. Run src.run_abides_days first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # 2) load hub prices and make next-day label
    hub = pd.read_csv(HUB_PATH, parse_dates=["date"]).sort_values("date")
    hub["y_next_day"] = hub["price"].shift(-1)

    # 3) join on date
    out = df.merge(hub[["date", "price", "y_next_day"]], on="date", how="left")

    # 4) drop rows where label missing (end of series / missing days)
    out = out.dropna(subset=["y_next_day"])

    for col in ["best_bid", "best_ask", "mid", "avg_tx_price", "spread"]:
        if col in out.columns:
            out[col + "_scaled"] = out[col] / SCALE

    # 5) save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] Saved dataset: {OUT_PATH}  rows={len(out)}  cols={len(out.columns)}")

if __name__ == "__main__":
    main()
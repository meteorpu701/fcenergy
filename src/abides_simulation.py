from pathlib import Path
import pandas as pd
from abides_core import abides
from abides_markets.configs import rmsc04

from src.extract_agent_features import extract_agent_features

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "simulated_trades"

def run_rmsc04_simulation(date: str):
    config = rmsc04.build_config(
        seed=123,
        date=date,
        log_orders=False,
    )
    end_state = abides.run(config)
    exch = end_state["agents"][0]  # ExchangeAgent
    print("[DEBUG] exchange order_books keys:", list(exch.order_books.keys())[:10])
    return end_state

def run_and_save_agent_features(date: str) -> Path:
    end_state = run_rmsc04_simulation(date)
    df = extract_agent_features(end_state, date)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"agent_features_{date}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} rows -> {out_path}")
    return out_path

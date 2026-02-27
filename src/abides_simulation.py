from pathlib import Path
import pandas as pd
from abides_core import abides
from abides_markets.configs import rmsc04

from src.extract_agent_features import extract_agent_features
from src.hub_price_loader import get_ttf_price_for_date

SCALE = 10_000

# where outputs go
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "simulated_trades"


def get_r_bar_for_date(date: str) -> float:
    """
    Returns oracle fundamental mean (r_bar) in ABIDES price units.
    """
    return float(get_ttf_price_for_date(date)) * SCALE


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
    """
    Runs ABIDES for one date and saves:
    1) agent-level microstructure features
    2) day-level oracle summary (r_bar)
    """
    end_state = run_rmsc04_simulation(date)

    # ---------------------------
    # 1️⃣ Save agent-level features
    # ---------------------------
    df = extract_agent_features(end_state, date)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUT_DIR / f"agent_features_{date}.csv"
    df.to_csv(features_path, index=False)
    print(f"[OK] Saved {len(df)} rows -> {features_path}")

    # ---------------------------
    # 2️⃣ Save oracle day summary
    # ---------------------------
    r_bar = get_r_bar_for_date(date)

    summary = pd.DataFrame([{
        "date": date,
        "r_bar": r_bar,                     # ABIDES units
        "r_bar_scaled": r_bar / SCALE,      # back to €/MWh-like units
    }])

    summary_path = OUT_DIR / f"day_summary_{date}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved oracle summary -> {summary_path}")

    return features_path
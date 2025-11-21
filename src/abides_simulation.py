from pathlib import Path
import pandas as pd
from abides_core import abides
from abides_markets.configs import rmsc04

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "simulated_trades"


def run_rmsc04_simulation(date: str, output_path: Path | None = None):
    """
    Run the ABIDES RMSC04 market simulation for a single day.

    Parameters:
        date: "YYYY-MM-DD" (will be passed to build_config(date=...))
        output_path: where to save a CSV (for now this will probably be empty
                     until we wire up proper extraction from agents / order_books)
    """
    # If your build_config signature looks like:
    # def build_config(seed=None, date=pd.Timestamp("2020-06-05"), log_orders=False, ...):
    # then this is correct:
    config = rmsc04.build_config(
        seed=123,
        date=date,  
        log_orders=False,
        use_hub = True
    )

    print(f"[ABIDES] Running RMSC04 for {date}...")
    end_state = abides.run(config)
    print("[ABIDES] Simulation finished.")

    # JPMC ABIDES often does NOT put trades in end_state["trades"].
    # This will likely be empty for you:
    trades = end_state.get("trades", [])

    if output_path is not None:
        df = pd.DataFrame(trades)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[ABIDES] Saved {len(df)} rows to {output_path}")

    return end_state
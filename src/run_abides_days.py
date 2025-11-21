from pathlib import Path
from src.abides_simulation import run_rmsc04_simulation

DAYS = [
    "2020-06-05",
    "2020-06-08",
    "2020-06-09",
    "2020-06-10",
]

def main():
    for d in DAYS:
        out = Path("data/simulated_trades") / f"trades_{d}.csv"
        if not out.exists():
            run_rmsc04_simulation(date=d, output_path=out)
        else:
            print(f"[SKIP] {d} already simulated")

if __name__ == "__main__":
    # python3 -m src.run_abides_days
    main()
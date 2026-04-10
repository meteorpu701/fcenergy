# src/build_exp1_dataset.py
from pathlib import Path
import pandas as pd

IN_DIR = Path("data/simulated_trades")
OUT_PATH = Path("data/exp1_dataset.csv")

def main():
    files = sorted(IN_DIR.glob("agent_features_*.csv"))
    if not files:
        raise SystemExit(f"No agent feature files found in {IN_DIR}")

    parts = []
    for f in files:
        df = pd.read_csv(f)
        if "date" not in df.columns:
            date_str = f.stem.replace("agent_features_", "")
            df["date"] = date_str

        if "agent_type" in df.columns:
            df = df[df["agent_type"] != "ExchangeAgent"].copy()

        parts.append(df)

    out = pd.concat(parts, ignore_index=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] Saved dataset: {OUT_PATH}  rows={len(out)}  cols={len(out.columns)}")

if __name__ == "__main__":
    main()
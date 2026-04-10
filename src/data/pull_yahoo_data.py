# src/pull_yahoo_data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

OUT_PATH = Path("data/raw_nbp_yahoo.csv")
CANDIDATES = {
    #"TTF": "TTF=F",
    "NBP": "NBP=F",

    # Optional candidates
    # "PEG": "PEG=F",
    # "THE": "THE=F",
    # "PSV": "PSV=F",
}

def download_one(symbol: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = df[[price_col]].rename(columns={price_col: "price"}).copy()
    out.index = pd.to_datetime(out.index)
    out = out.dropna()
    return out

def main():
    rows = []
    ok = []
    failed = []

    for hub, symbol in CANDIDATES.items():
        try:
            df = download_one(symbol, period="5y")
            if df.empty:
                failed.append((hub, symbol, "empty"))
                print(f"[WARN] {hub} ({symbol}) -> empty")
                continue

            df = df.reset_index().rename(columns={"index": "date"})
            df["hub"] = hub
            rows.append(df[["date", "hub", "price"]])
            ok.append((hub, symbol, len(df)))
            print(f"[OK] {hub} ({symbol}) rows={len(df)} date_range={df['date'].min().date()}..{df['date'].max().date()}")
        except Exception as e:
            failed.append((hub, symbol, repr(e)))
            print(f"[ERROR] {hub} ({symbol}) -> {e}")

    if not rows:
        raise SystemExit("No Yahoo series downloaded successfully. (TTF=F / NBP=F should work.)")

    out = pd.concat(rows, ignore_index=True).sort_values(["hub", "date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[DONE] wrote {OUT_PATH} rows={len(out)} hubs={sorted(out['hub'].unique().tolist())}")

    if failed:
        print("\n[SUMMARY] Failed candidates:")
        for hub, symbol, reason in failed:
            print(f"  - {hub} ({symbol}): {reason}")

if __name__ == "__main__":
    main()
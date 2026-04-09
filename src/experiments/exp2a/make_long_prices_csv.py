from __future__ import annotations

from pathlib import Path
import pandas as pd


IN_DIR = Path("data/eu_hub_prices")
OUT = Path("data/eu_hub_prices_long.csv")

FILES = {
    "TTF": IN_DIR / "TTF_yahoo.csv",
    "NBP": IN_DIR / "NBP_yahoo.csv",
}


def _find_price_col(df: pd.DataFrame) -> str:
    # common yahoo export columns: "Adj Close", "Close"
    for c in ["Adj Close", "Close", "close", "adjclose"]:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find a price column in columns={list(df.columns)}")


def main():
    rows = []
    for hub, fp in FILES.items():
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_csv(fp)

        # find date col
        date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
        if date_col is None:
            raise KeyError(f"{fp} missing Date column. columns={list(df.columns)}")

        px_col = _find_price_col(df)

        out = df[[date_col, px_col]].copy()
        out.columns = ["date", "price"]
        out["hub"] = hub

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")

        out = out.dropna(subset=["date", "price"]).copy()

        # ensure string format your pipeline likes (but still parseable)
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")

        rows.append(out)

    final = pd.concat(rows, ignore_index=True)
    final = final.sort_values(["hub", "date"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT} rows={len(final)} hubs={sorted(final['hub'].unique().tolist())}")


if __name__ == "__main__":
    main()
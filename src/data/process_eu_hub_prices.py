from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw_eu_hubs")
OUT_PATH = Path("data/eu_hub_prices.csv")

def detect_price_col(cols):
    if "Index Value (€/MWh)" in cols:
        return "Index Value (€/MWh)"
    if "LTU NGP (€/MWh)" in cols:
        return "LTU NGP (€/MWh)"
    for c in cols:
        s = str(c)
        if "(€/MWh)" in s and "+10%" not in s and "-10%" not in s:
            return c
    raise ValueError(f"Could not find price column. Columns={list(cols)}")

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")

    # remove extra empty col from trailing ';' (TTF file)
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

    if "Delivery date" not in df.columns:
        raise ValueError(f"Missing 'Delivery date' in {path.name}. Columns={list(df.columns)}")

    price_col = detect_price_col(df.columns)

    out = df[["Delivery date", price_col]].copy()
    out.columns = ["date", "price"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    out = out.dropna(subset=["date", "price"])

    hub = path.stem.split("_")[0].upper()  # e.g. TTF_NGP_60_Days -> TTF
    out["hub"] = hub

    return out[["date", "hub", "price"]]

def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No csv files found in {RAW_DIR}")

    parts = []
    for f in files:
        d = load_one(f)
        print(f"[OK] {f.name} -> hub={d['hub'].iloc[0]} rows={len(d)}")
        parts.append(d)

    out = pd.concat(parts, ignore_index=True).sort_values(["hub", "date"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[DONE] wrote {OUT_PATH} rows={len(out)} hubs={out['hub'].nunique()}")

if __name__ == "__main__":
    main()
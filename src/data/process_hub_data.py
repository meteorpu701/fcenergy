import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "Henry_Hub_Natural_Gas_Spot_Price.csv"
OUT_PATH = DATA_DIR / "hub_prices_processed.csv"

def preprocess_henry_hub_raw():
    print(f"Loading raw Henry Hub data from: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, header=4)

    print("Raw columns:", df.columns.tolist())

    date_col = [c for c in df.columns if "Day" in c or "Date" in c][0]
    price_col = [c for c in df.columns if "Price" in c][0]

    df[date_col] = df[date_col].astype(str).str.strip()
    df[price_col] = df[price_col].astype(str).str.strip()

    df = df.rename(columns={date_col: "date", price_col: "price"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"]).sort_values("date")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Processed hub prices saved to: {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    preprocess_henry_hub_raw()
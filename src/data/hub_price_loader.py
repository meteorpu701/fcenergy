# src/hub_price_loader.py
import pandas as pd

_PRICES = None

def get_ttf_price_for_date(date_str: str) -> float:
    """
    Returns TTF NGP in €/MWh for date_str 'YYYY-MM-DD'
    """
    global _PRICES
    if _PRICES is None:
        df = pd.read_csv("data/eu_hub_prices.csv", parse_dates=["date"])
        ttf = df[df["hub"] == "TTF"].copy()
        ttf["date"] = ttf["date"].dt.strftime("%Y-%m-%d")
        ttf["price"] = pd.to_numeric(ttf["price"], errors="coerce")
        _PRICES = dict(zip(ttf["date"], ttf["price"]))
    p = _PRICES.get(date_str)
    if p is None:
        raise KeyError(f"No TTF price found for {date_str}")
    return float(p)
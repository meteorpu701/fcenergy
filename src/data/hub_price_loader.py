# src/hub_price_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

# Cache: (hub, date_str) -> price
_PRICES: Dict[Tuple[str, str], float] = {}
_LOADED_PATH: Optional[Path] = None


def _norm_hub(hub: str) -> str:
    return str(hub).strip().upper()


def _load_prices(csv_path: str) -> None:
    """
    Internal loader.
    Builds dict: (HUB, 'YYYY-MM-DD') -> float(price)
    """
    global _PRICES, _LOADED_PATH

    path = Path(csv_path)

    # Already loaded this exact file
    if _LOADED_PATH == path and _PRICES:
        return

    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    required = {"date", "hub", "price"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)} (need {sorted(required)})")

    # Normalize
    df["hub"] = df["hub"].astype(str).map(_norm_hub)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop missing prices
    df = df.dropna(subset=["price"])

    # Build cache (vectorized, fast)
    _PRICES = {
        (h, d): float(p)
        for h, d, p in zip(df["hub"], df["date"], df["price"])
    }

    _LOADED_PATH = path


def get_price_for_hub_date(
    hub: str,
    date_str: str,
    csv_path: str = "data/eu_hub_prices_exp2a.csv",
) -> float:
    """
    Generic loader.

    Args:
        hub: e.g. "TTF", "NBP", "FIN", "LTU"
        date_str: 'YYYY-MM-DD'
        csv_path: which hub price file to use

    Returns:
        float price
    """
    _load_prices(csv_path)

    key = (_norm_hub(hub), date_str)
    p = _PRICES.get(key)

    if p is None:
        raise KeyError(f"No price found for hub={hub} date={date_str} in {csv_path}")

    return float(p)


# Backward compatibility for Exp1
def get_ttf_price_for_date(date_str: str) -> float:
    """
    Legacy function used by Exp1.
    Defaults to original price file.
    """
    return get_price_for_hub_date(
        hub="TTF",
        date_str=date_str,
        csv_path="data/eu_hub_prices.csv",
    )
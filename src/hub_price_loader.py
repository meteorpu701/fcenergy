from pathlib import Path
from typing import Union
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUB_CSV = PROJECT_ROOT / "data" / "hub_prices_processed.csv"

# Load once at import time
_HUB_DF = pd.read_csv(HUB_CSV, parse_dates=["date"]).sort_values("date")


def get_hub_price_for_date(date: Union[str, pd.Timestamp], method: str = "previous") -> float:
    """
    Return the Henry Hub price for the given calendar date.

    date: "YYYY-MM-DD" or pd.Timestamp
    method:
        - "exact": require exact match (raise if not found)
        - "previous": use the most recent previous available date
    """
    if isinstance(date, str):
        date_ts = pd.to_datetime(date).normalize()
    else:
        date_ts = pd.to_datetime(date).normalize()

    df = _HUB_DF.copy()
    df["date_norm"] = df["date"].dt.normalize()

    if method == "exact":
        row = df.loc[df["date_norm"] == date_ts]
        if row.empty:
            raise ValueError(f"No exact Henry Hub price found for date {date_ts.date()}")
        return float(row["price"].iloc[0])

    # default: "previous"
    # use last available price before or on this date
    row = df.loc[df["date_norm"] <= date_ts]
    if row.empty:
        raise ValueError(f"No Henry Hub price available before or on {date_ts.date()}")
    return float(row["price"].iloc[-1])
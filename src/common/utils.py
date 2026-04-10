import pandas as pd
from typing import List, Optional

def add_lag_features(
    df: pd.DataFrame,
    cols_to_lag: List[str],
    max_lag: int,
    group_cols: Optional[List[str]] = None,
    time_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    sort_cols = (group_cols or []) + [time_col]
    df = df.sort_values(sort_cols)

    if group_cols:
        g = df.groupby(group_cols, sort=False)
        for col in cols_to_lag:
            for L in range(1, max_lag + 1):
                df[f"{col}_lag{L}"] = g[col].shift(L)
    else:
        for col in cols_to_lag:
            for L in range(1, max_lag + 1):
                df[f"{col}_lag{L}"] = df[col].shift(L)

    return df
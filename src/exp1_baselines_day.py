# src/exp1_baselines_day.py
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/exp1_day_dataset.csv")

FEATURES = [
    "mid_mean",
    "mid_std",
    "mid_p10",
    "mid_p90",
    "spread_mean",
    "spread_std",
    "vol_mean",
    "vol_sum",
    "quote_coverage",
    "n_agents_with_quotes",
    "n_agents_total",
]

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")

    y = df["y_next_day"].astype(float)

    # naive baseline: tomorrow ≈ today
    naive_pred = df["price"].astype(float)
    print("=== Naive baseline (y_next_day ≈ price_today) ===")
    print("MAE:", mean_absolute_error(y, naive_pred))
    print("RMSE:", rmse(y, naive_pred))

    X = df[FEATURES]

    # time-aware split (important)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n=== Centralized Ridge baseline (day-level, time split) ===")
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", rmse(y_test, pred))
    print("R2:", r2_score(y_test, pred))

if __name__ == "__main__":
    main()
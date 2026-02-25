# src/exp1_baselines.py
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/exp1_dataset.csv")

FEATURES = [
    "best_bid_scaled",
    "best_ask_scaled",
    "mid_scaled",
    "spread_scaled",
    "n_fills",
    "transacted_volume",
]

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    y = df["y_next_day"].astype(float)

    # naive baseline
    naive_pred = df["price"].astype(float)
    print("=== Naive baseline (y_next_day ≈ price_today) ===")
    print("MAE:", mean_absolute_error(y, naive_pred))
    print("RMSE:", mean_squared_error(y, naive_pred) ** 0.5)

    # centralized model
    X = df[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n=== Centralized Ridge baseline ===")
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
    print("R2:", r2_score(y_test, pred))

if __name__ == "__main__":
    main()
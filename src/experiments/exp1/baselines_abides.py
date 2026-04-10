# src/exp1_baselines_abides.py

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/exp1_day_dataset_abides.csv"
TARGET = "y_next_day_abides"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")

    print(f"[INFO] rows={len(df)}")

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    print(f"[INFO] train={len(train)} test={len(test)}")

    naive_pred = test["abides_price"].values
    y_true = test[TARGET].values

    print("\n=== Naive baseline (AR1: y_{t+1}=y_t) ===")
    print("MAE:", mean_absolute_error(y_true, naive_pred))
    print("RMSE:", rmse(y_true, naive_pred))

    ar_cols = [c for c in df.columns if c.startswith("abides_price_lag")]

    X_train_ar = train[ar_cols]
    X_test_ar = test[ar_cols]

    y_train = train[TARGET]
    y_test = test[TARGET]

    model_ar = Ridge(alpha=1.0)
    model_ar.fit(X_train_ar, y_train)
    pred_ar = model_ar.predict(X_test_ar)

    print("\n=== Centralized AR-only Ridge ===")
    print("MAE:", mean_absolute_error(y_test, pred_ar))
    print("RMSE:", rmse(y_test, pred_ar))

    exclude_cols = {"date", TARGET, "ttf_price"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    model_arx = Ridge(alpha=1.0)
    model_arx.fit(X_train, y_train)
    pred_arx = model_arx.predict(X_test)

    print("\n=== Centralized ARX Ridge (lags + microstructure) ===")
    print("MAE:", mean_absolute_error(y_test, pred_arx))
    print("RMSE:", rmse(y_test, pred_arx))

if __name__ == "__main__":
    main()
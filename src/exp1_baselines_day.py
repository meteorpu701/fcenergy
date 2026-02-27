# src/exp1_baselines_day.py
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/exp1_day_dataset.csv")
LAGS = 3

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")

    y = df["y_next_day"].astype(float)

    # Naive baseline: y_{t+1} = y_t  (use price_lag1 as "today")
    naive_pred = df["price_lag1"].astype(float)
    print("=== Naive baseline (AR1: y_{t+1}=y_t) ===")
    print("MAE:", mean_absolute_error(y, naive_pred))
    print("RMSE:", rmse(y, naive_pred))

    # AR-only baseline: use only lagged price
    ar_cols = [f"price_lag{i}" for i in range(1, LAGS+1)]

    # ARX baseline: lagged price + lagged microstructure summaries
    base_feats = ["mid_mean","spread_mean","vol_sum","fills_sum","quote_coverage"]
    arx_cols = ar_cols + [f"{c}_lag{i}" for c in base_feats for i in range(1, LAGS+1)]

    # time split
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    def fit_eval(cols, name):
        X_train, X_test = train[cols], test[cols]
        model = Pipeline([("imp", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=1.0))])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print("MAE:", mean_absolute_error(y_test, pred))
        print("RMSE:", rmse(y_test, pred))

    fit_eval(ar_cols, "Centralized AR-only Ridge")
    fit_eval(arx_cols, "Centralized ARX Ridge (lags + microstructure)")

if __name__ == "__main__":
    main()
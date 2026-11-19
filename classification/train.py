"""Build the COMP9417 KNN price artifact with 3-year sliding-window retraining every 2 years.

Initial fit: 2010–2013.
Validation: 2014 (evaluation only; does not update the model).
Retraining: January 2017, 2019, 2021, 2023 and 2025, using only the preceding 3 years of data.

Usage: python train.py path/to/nem_weather_halfhourly_2010_2026.csv
No dependency on replay.py, evaluation.py, or classification/.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import market_day, price_features_from_components

INITIAL_TRAIN_START = pd.Timestamp("2010-01-01")
INITIAL_TRAIN_END = pd.Timestamp("2014-01-01")
VALIDATION_END = pd.Timestamp("2015-01-01")
RETRAIN_DATES = [pd.Timestamp(f"{year}-01-01") for year in (2017, 2019, 2021, 2023, 2025)]
WINDOW_YEARS = 3


def make_training_data(df):
    df = df.sort_index().copy()
    day = market_day(df.index)
    daily_mean = df.groupby(day)["nsw_price"].transform("mean")
    t = pd.DataFrame(index=df.index)
    t["day"] = day
    shifted = df.index - pd.Timedelta(minutes=30)
    t["period"] = shifted.hour * 2 + shifted.minute // 30
    t["y_up"] = (df["nsw_price"] > daily_mean.shift(48)).astype(int)
    t["demand_lag1"] = df["nsw_demand"].shift(48)
    t["demand_lag7"] = df["nsw_demand"].shift(48 * 7)
    t["price_lag1"] = df["nsw_price"].shift(48)
    t["previous_mean_price"] = daily_mean.shift(48)
    t["up_lag1"] = (df["nsw_price"].shift(48) > daily_mean.shift(48)).astype(int)
    t["up_lag7"] = (df["nsw_price"].shift(48 * 7) > daily_mean.shift(48)).astype(int)
    for col in ("syd_temp", "syd_solar", "syd_rh"):
        t[col] = df[col]
    t = t.iloc[48 * 7:].dropna()
    return price_features_from_components(t), t["y_up"], t["day"]


def new_model():
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=1),
    )


def main(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")
    X, y, day = make_training_data(df)

    # Initial model: 2010–2013 only.
    initial_fit = (day >= INITIAL_TRAIN_START) & (day < INITIAL_TRAIN_END)
    if not initial_fit.any():
        raise ValueError("No initial training data in 2010–2013")
    model = new_model()
    model.fit(X.loc[initial_fit], y.loc[initial_fit])
    print(f"Initial fit: {initial_fit.sum():,} rows through {day.loc[initial_fit].max().date()}")

    # Validation: 2014 only. Predicting does not change the trained model.
    validation = (day >= INITIAL_TRAIN_END) & (day < VALIDATION_END)
    if validation.any():
        predictions = model.predict(X.loc[validation])
        score = balanced_accuracy_score(y.loc[validation], predictions)
        print(f"2014 validation balanced accuracy: {score:.4f}")
    else:
        print("Warning: no 2014 validation data available")

    # Sliding-window retraining, matching replay.py --window-years 3.
    # The initial 2010–2013 model remains unchanged through validation.
    for i, retrain_date in enumerate(RETRAIN_DATES, start=1):
        window_start = retrain_date - pd.DateOffset(years=WINDOW_YEARS)
        fit = (day >= window_start) & (day < retrain_date)
        if not fit.any():
            raise ValueError(f"No training data before {retrain_date.date()}")
        model = new_model()
        model.fit(X.loc[fit], y.loc[fit])
        print(f"Retrain {i} ({retrain_date.date()}): {fit.sum():,} rows from {day.loc[fit].min().date()} through {day.loc[fit].max().date()}")

    out = Path(__file__).resolve().parent / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "price_model.joblib"
    joblib.dump(model, path)
    print(f"Saved final January 2025 KNN model (2022–2024 window): {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Released historical CSV")
    args = parser.parse_args()
    main(args.csv_path)

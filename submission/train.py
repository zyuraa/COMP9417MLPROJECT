import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import market_day


def make_training_data(df):
    """Reproduce the price-direction training table from the starter notebook."""
    df = df.sort_index().copy()
    df["day"] = market_day(df.index)
    df["period"] = ((df.index - pd.Timedelta(minutes=30)).hour * 2
                    + (df.index - pd.Timedelta(minutes=30)).minute // 30)

    daily_price_mean = df.groupby("day")["nsw_price"].transform("mean")
    t = pd.DataFrame(index=df.index)
    t["day"], t["period"] = df["day"], df["period"]
    t["y_up"] = (df["nsw_price"] > daily_price_mean.shift(48)).astype(int)
    t["demand_lag1"] = df["nsw_demand"].shift(48)
    t["demand_lag7"] = df["nsw_demand"].shift(48 * 7)
    t["price_lag1"] = df["nsw_price"].shift(48)
    t["previous_mean_price"] = daily_price_mean.shift(48)
    t["up_lag1"] = (df["nsw_price"].shift(48) > daily_price_mean.shift(48)).astype(int)
    t["up_lag7"] = (df["nsw_price"].shift(48 * 7) > daily_price_mean.shift(48)).astype(int)
    for name in ("syd_temp", "syd_solar", "syd_rh"):
        t[name] = df[name]
    t = t.iloc[48 * 7:].dropna()

    features = ["demand_lag1", "price_lag1", "previous_mean_price",
                "syd_temp", "syd_solar", "syd_rh", "up_lag1", "up_lag7"]
    X = t[features].copy()
    dow = pd.get_dummies(t["day"].dt.day_of_week, prefix="dow", dtype=int)
    # Ensure all seven columns exist even when training on a short date range.
    dow = dow.reindex(columns=[f"dow_{d}" for d in range(7)], fill_value=0)
    X = pd.concat([X, dow], axis=1)

    X["sin_period"] = np.sin(2 * np.pi * t["period"] / 48)
    X["cos_period"] = np.cos(2 * np.pi * t["period"] / 48)
    doy = t["day"].dt.day_of_year
    X["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_year"] = np.cos(2 * np.pi * doy / 365.25)

    for prefix, lag in (("demand", "demand_lag1"), ("price", "price_lag1")):
        for cycle in ("period", "year"):
            for trig in ("sin", "cos"):
                X[f"{prefix}_{trig}_{cycle}"] = X[lag] * X[f"{trig}_{cycle}"]

    X["temp_squared"] = t["syd_temp"] ** 2
    X["temp_sin_period"] = t["syd_temp"] * X["sin_period"]
    X["temp_cos_period"] = t["syd_temp"] * X["cos_period"]
    X["price_relative_mean"] = t["price_lag1"] - t["previous_mean_price"]
    X["demand_pct_change"] = ((t["demand_lag1"] - t["demand_lag7"])
                              / t["demand_lag7"].clip(lower=1))
    X["solar_sin_period"] = X["syd_solar"] * X["sin_period"]
    X["solar_cos_period"] = X["syd_solar"] * X["cos_period"]
    for cycle in ("period", "year"):
        for trig in ("sin", "cos"):
            X[f"up_lag1_{trig}_{cycle}"] = X["up_lag1"] * X[f"{trig}_{cycle}"]
    X["up_lag1_lag7"] = X["up_lag1"] * X["up_lag7"]
    X["up_lag1_price_relative"] = X["up_lag1"] * X["price_relative_mean"]
    for d in range(1, 7):
        X[f"up_lag1_dow_{d}"] = X["up_lag1"] * X[f"dow_{d}"]

    return X, t["y_up"], t["day"]


def main(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")
    X, y, days = make_training_data(df)
    train = days.dt.year <= 2014
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(X.loc[train], y.loc[train])
    out = Path(__file__).resolve().parent / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "price_model.joblib")
    print(f"Saved price model trained on {train.sum():,} half-hours to {out}")
    
    # TODO: teammate adds demand model training and saving here.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    main(parser.parse_args().csv_path)

"""COMP9417 KNN price-direction inference; demand integration placeholder."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def market_day(index):
    return pd.DatetimeIndex(index - pd.Timedelta(minutes=30)).normalize()


def price_features_from_components(t):
    """Shared feature transformations for training and live prediction."""
    X = t[["demand_lag1", "price_lag1", "previous_mean_price",
           "syd_temp", "syd_solar", "syd_rh", "up_lag1", "up_lag7"]].copy()
    for d in range(7):
        X[f"dow_{d}"] = (t["day"].dt.dayofweek == d).astype(int)
    X["sin_period"] = np.sin(2 * np.pi * t["period"] / 48)
    X["cos_period"] = np.cos(2 * np.pi * t["period"] / 48)
    doy = t["day"].dt.dayofyear
    X["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    for prefix, lag in (("demand", "demand_lag1"), ("price", "price_lag1")):
        for cycle in ("period", "year"):
            for trig in ("sin", "cos"):
                X[f"{prefix}_{trig}_{cycle}"] = X[lag] * X[f"{trig}_{cycle}"]
    X["temp_squared"] = X["syd_temp"] ** 2
    X["temp_sin_period"] = X["syd_temp"] * X["sin_period"]
    X["temp_cos_period"] = X["syd_temp"] * X["cos_period"]
    X["price_relative_mean"] = X["price_lag1"] - X["previous_mean_price"]
    X["demand_pct_change"] = ((X["demand_lag1"] - t["demand_lag7"])
                              / t["demand_lag7"].clip(lower=1))
    for trig in ("sin", "cos"):
        X[f"solar_{trig}_period"] = X["syd_solar"] * X[f"{trig}_period"]
    for cycle in ("period", "year"):
        for trig in ("sin", "cos"):
            X[f"up_lag1_{trig}_{cycle}"] = X["up_lag1"] * X[f"{trig}_{cycle}"]
    X["up_lag1_lag7"] = X["up_lag1"] * X["up_lag7"]
    X["up_lag1_price_relative"] = X["up_lag1"] * X["price_relative_mean"]
    for d in range(1, 7):
        X[f"up_lag1_dow_{d}"] = X["up_lag1"] * X[f"dow_{d}"]
    return X


def build_price_features(history, weather_next):
    h = history.sort_index()
    w = weather_next.sort_index()
    if len(w) != 48 or len(h) < 48 * 7:
        raise ValueError("Need 48 next-day intervals and at least seven historical days")
    if not h.index.is_unique or not w.index.is_unique:
        raise ValueError("Duplicate timestamps")
    hd, wd = market_day(h.index), market_day(w.index)
    if len(pd.unique(wd)) != 1 or wd[0] != hd[-1] + pd.Timedelta(days=1):
        raise ValueError("Next-day weather is not aligned to history")
    prev = h.iloc[-48:]
    week = h.iloc[-48 * 7:-48 * 6]
    previous_mean = float(prev["nsw_price"].mean())
    t = pd.DataFrame(index=w.index)
    t["day"] = wd
    shifted = w.index - pd.Timedelta(minutes=30)
    t["period"] = shifted.hour * 2 + shifted.minute // 30
    t["demand_lag1"] = prev["nsw_demand"].to_numpy()
    t["demand_lag7"] = week["nsw_demand"].to_numpy()
    t["price_lag1"] = prev["nsw_price"].to_numpy()
    t["previous_mean_price"] = previous_mean
    t["up_lag1"] = (prev["nsw_price"].to_numpy() > previous_mean).astype(int)
    t["up_lag7"] = (week["nsw_price"].to_numpy() > previous_mean).astype(int)
    for col in ("syd_temp", "syd_solar", "syd_rh"):
        t[col] = w[col].to_numpy()
    return price_features_from_components(t)


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.price_model = joblib.load(Path(artifact_dir) / "price_model.joblib")
        # Replace temporary demand baseline with teammate's fitted demand model.

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        X = build_price_features(history, weather_next)
        expected = list(self.price_model.feature_names_in_)
        probabilities = self.price_model.predict_proba(X[expected])[:, 1]
        price = (probabilities > 0.5).astype(int)
        demand = history.sort_index()["nsw_demand"].iloc[-48:].to_numpy(dtype=float)
        return pd.DataFrame({"nsw_demand_mw": demand, "nsw_price_up": price},
                            index=weather_next.index)

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_FEATURES = ["demand_lag1", "price_lag1", "previous_mean_price", "syd_temp", "syd_solar", "syd_rh", "up_lag1", "up_lag7"]


def market_day(index):
    # The 00:00 timestamp is the last interval of the preceding market day.
    return pd.DatetimeIndex(index - pd.Timedelta(minutes=30)).normalize()


def build_price_features(history, weather_next):
    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    h = history.sort_index()
    w = weather_next.sort_index()

    if len(w) != 48:
        raise ValueError("Expected exactly 48 weather rows")

    if len(h) < 48 * 7:
        raise ValueError("At least seven historical days are required")

    if not h.index.is_unique or not w.index.is_unique:
        raise ValueError("Duplicate timestamps")

    # Convert timestamps to market days.
    # A market day runs from 00:30 to the next day's 00:00.
    hday = market_day(h.index)
    wday = market_day(w.index)

    if len(pd.unique(wday)) != 1:
        raise ValueError("Expected one complete market day")

    if wday[0] != hday[-1] + pd.Timedelta(days=1):
        raise ValueError("Weather must be for the following day")

    # --------------------------------------------------
    # 2. Extract historical data
    # --------------------------------------------------

    # Same 48 half-hours from yesterday and last week
    prev = h.iloc[-48:]
    week = h.iloc[-48 * 7:-48 * 6]

    if len(prev) != 48 or len(week) != 48:
        raise ValueError("Incomplete lagged days")

    prev_demand = prev["nsw_demand"].to_numpy()
    prev_price = prev["nsw_price"].to_numpy()

    week_demand = week["nsw_demand"].to_numpy()
    week_price = week["nsw_price"].to_numpy()

    # Mean NSW price across the previous day
    previous_mean = float(prev_price.mean())

    # Create feature matrix with future timestamps
    X = pd.DataFrame(index=w.index)

    # --------------------------------------------------
    # 3. Historical demand and price features
    # --------------------------------------------------

    X["demand_lag1"] = prev_demand
    X["price_lag1"] = prev_price
    X["previous_mean_price"] = previous_mean

    # Whether historical prices exceed yesterday's mean
    X["up_lag1"] = (prev_price > previous_mean).astype(int)
    X["up_lag7"] = (week_price > previous_mean).astype(int)

    # Demand change relative to the same period last week
    X["demand_pct_change"] = (
        (prev_demand - week_demand)
        / np.maximum(week_demand, 1)
    )

    # Yesterday's price relative to yesterday's mean
    X["price_relative_mean"] = (
        X["price_lag1"] - previous_mean
    )

    # --------------------------------------------------
    # 4. Next-day weather features
    # --------------------------------------------------

    for name in ["syd_temp", "syd_solar", "syd_rh"]:
        X[name] = w[name].to_numpy()

    # Nonlinear temperature feature
    X["temp_squared"] = X["syd_temp"] ** 2

    # --------------------------------------------------
    # 5. Calendar and cyclical time features
    # --------------------------------------------------

    # One-hot encoding of day of week (Monday = 0)
    day_of_week = wday[0].dayofweek

    for d in range(7):
        X[f"dow_{d}"] = int(day_of_week == d)

    # Cyclical time-of-day encoding
    period = np.arange(48)

    X["sin_period"] = np.sin(2 * np.pi * period / 48)
    X["cos_period"] = np.cos(2 * np.pi * period / 48)

    # Cyclical time-of-year encoding
    day_of_year = wday[0].dayofyear

    X["sin_year"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    X["cos_year"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    # --------------------------------------------------
    # 6. Interaction features
    # --------------------------------------------------

    # Demand and price interactions with time cycles
    for prefix, lag in [
        ("demand", "demand_lag1"),
        ("price", "price_lag1")
    ]:
        for cycle in ["period", "year"]:
            for trig in ["sin", "cos"]:
                X[f"{prefix}_{trig}_{cycle}"] = (
                    X[lag] * X[f"{trig}_{cycle}"]
                )

    # Temperature interactions with time of day
    for trig in ["sin", "cos"]:
        X[f"temp_{trig}_period"] = (
            X["syd_temp"] * X[f"{trig}_period"]
        )

        # Solar radiation interactions with time of day
        X[f"solar_{trig}_period"] = (
            X["syd_solar"] * X[f"{trig}_period"]
        )

    # Previous price direction interactions with time
    for cycle in ["period", "year"]:
        for trig in ["sin", "cos"]:
            X[f"up_lag1_{trig}_{cycle}"] = (
                X["up_lag1"] * X[f"{trig}_{cycle}"]
            )

    # Interactions between previous price direction,
    # weekly price direction, and relative price
    X["up_lag1_lag7"] = X["up_lag1"] * X["up_lag7"]

    X["up_lag1_price_relative"] = (
        X["up_lag1"] * X["price_relative_mean"]
    )

    # Interaction between price direction and day of week
    for d in range(1, 7):
        X[f"up_lag1_dow_{d}"] = (
            X["up_lag1"] * X[f"dow_{d}"]
        )

    return X


class Forecaster:
    def __init__(self, artifact_dir: str):
        self.price_model = joblib.load(Path(artifact_dir) / "price_model.joblib")
        # TODO: teammate loads demand_model here.

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        X = build_price_features(history, weather_next)
        expected = list(self.price_model.feature_names_in_)
        price = self.price_model.predict(X[expected]).astype(int)
        # TEMPORARY: yesterday demand baseline until teammate adds the demand model.
        demand = history.sort_index()["nsw_demand"].iloc[-48:].to_numpy(dtype=float)
        return pd.DataFrame({"nsw_demand_mw": demand, "nsw_price_up": price}, index=weather_next.index)

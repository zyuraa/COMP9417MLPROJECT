"""Day-ahead NSW demand predictors shared by training and live forecasts."""

import holidays
import numpy as np
import pandas as pd


SLOTS_PER_DAY = 48
HALF_HOUR = pd.Timedelta(minutes=30)
ONE_DAY = pd.Timedelta(days=1)
WEATHER = ["syd_temp", "syd_solar", "syd_rh"]

HISTORY_FEATURES = [
    "demand_lag1", "demand_lag7", "slot", "weekday", "month",
    "non_workday", "non_workday_lag1", "non_workday_lag7",
]
WEATHER_FEATURES = HISTORY_FEATURES + WEATHER
RECENT_FEATURES = WEATHER_FEATURES + [
    "previous_day_mean", "previous_day_peak", "same_slot_week_mean",
    "demand_change_1d", "temperature_change_1d",
]
FEATURE_GROUPS = {
    "history": HISTORY_FEATURES,
    "weather": WEATHER_FEATURES,
    "recent": RECENT_FEATURES,
    "profile": RECENT_FEATURES + [
        "previous_day_final", "lag1_previous_slot", "lag1_next_slot",
        "matched_profile", "forecast_day_max_temp",
    ],
}


def market_day(index):
    """Return each half-hour's market date using the start of its interval."""
    return (pd.DatetimeIndex(index) - HALF_HOUR).normalize()


def make_features(df):
    """Build one forecast row per input timestamp using only day-ahead information."""
    index = df.index
    interval_start = index - HALF_HOUR
    slot = interval_start.hour * 2 + interval_start.minute // 30
    days = market_day(index)
    demand = df["nsw_demand"]

    # Look up by timestamp so a missing half-hour cannot shift the lags that follow it.
    lags = [demand.reindex(index - lag * ONE_DAY).to_numpy(dtype=float) for lag in range(1, 8)]

    daily = demand.groupby(days).agg(["mean", "max", "count"])
    # A day summary represents all 48 observations of the previous market day.
    previous = daily.loc[daily["count"] == SLOTS_PER_DAY].reindex(days - ONE_DAY)
    unique_days = days.unique()
    calendar_days = unique_days.union(unique_days - ONE_DAY).union(unique_days - 7 * ONE_DAY)
    calendar_years = calendar_days.year.unique().tolist()
    nsw_holidays = holidays.country_holidays("AU", subdiv="NSW", years=calendar_years)
    holiday_dates = pd.DatetimeIndex(list(nsw_holidays))

    def non_working(dates):
        return (dates.weekday >= 5) | dates.isin(holiday_dates)

    non_workday = non_working(days)

    # Average only observed same-slot values with the target day's workday status.
    matched_sum = np.zeros(len(df))
    matched_count = np.zeros(len(df), dtype=np.int8)
    for lag, values in enumerate(lags, start=1):
        past_days = days - lag * ONE_DAY
        matches = (non_working(past_days) == non_workday) & np.isfinite(values)
        np.add(matched_sum, values, out=matched_sum, where=matches)
        matched_count += matches

    week_mean = (lags[0] + lags[1] + lags[6] + lags[2] + lags[3] + lags[4] + lags[5]) / 7
    matched_profile = np.divide(matched_sum, matched_count, out=week_mean.copy(), where=matched_count > 0)
    yesterday = index - ONE_DAY
    earlier = (yesterday - HALF_HOUR).where(slot > 0, yesterday)
    later = (yesterday + HALF_HOUR).where(slot < SLOTS_PER_DAY - 1, yesterday)

    X = pd.DataFrame({
        "demand_lag1": lags[0],
        "demand_lag7": lags[6],
        "slot": slot,
        "weekday": days.dayofweek,
        "month": days.month,
        "non_workday": non_workday.astype(int),
        "non_workday_lag1": non_working(days - ONE_DAY).astype(int),
        "non_workday_lag7": non_working(days - 7 * ONE_DAY).astype(int),
    }, index=index)
    X[WEATHER] = df[WEATHER]
    X["previous_day_mean"] = previous["mean"].to_numpy()
    X["previous_day_peak"] = previous["max"].to_numpy()
    X["same_slot_week_mean"] = week_mean
    X["demand_change_1d"] = lags[0] - lags[1]
    X["temperature_change_1d"] = df["syd_temp"].to_numpy() - df["syd_temp"].reindex(index - ONE_DAY).to_numpy()
    X["previous_day_final"] = demand.reindex(days).to_numpy()
    X["lag1_previous_slot"] = demand.reindex(earlier).to_numpy()
    X["lag1_next_slot"] = demand.reindex(later).to_numpy()
    X["matched_profile"] = matched_profile
    X["forecast_day_max_temp"] = df["syd_temp"].groupby(days).transform("max")
    return X


def forecast_features(history, weather_next):
    """Build a single 48-slot forecast from seven observed days and supplied weather."""
    target = weather_next.index
    if len(target) != SLOTS_PER_DAY:
        raise ValueError("Expected 48 half-hourly weather observations")
    first = market_day(target)[0] + HALF_HOUR
    expected = pd.date_range(first, periods=SLOTS_PER_DAY, freq=HALF_HOUR)
    if not target.equals(expected):
        raise ValueError("Weather rows must cover one complete market day in order")

    recent = history.iloc[-7 * SLOTS_PER_DAY:]
    expected_history = pd.date_range(first - 7 * ONE_DAY, periods=7 * SLOTS_PER_DAY, freq=HALF_HOUR)
    if not recent.index.equals(expected_history):
        raise ValueError("Expected seven complete market days immediately before the forecast")
    # Restrict the target block to weather even if the caller supplies additional columns.
    combined = pd.concat([recent[["nsw_demand", *WEATHER]], weather_next[WEATHER]])
    return make_features(combined).loc[target]

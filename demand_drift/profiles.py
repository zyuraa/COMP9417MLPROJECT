"""Detect changes in the lagged demand profiles available to a day-ahead forecaster."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from demand_adaptation.alarms import RollingUpperAlarm
from harness.dayahead import N_PERIODS, market_day


def input_history(df):
    """Return one input-only score and alarm decision per simulation market day.

    A row for D uses the already observed 48-slot NSW demand profile from D-1.
    The profile for the final forecast day is never passed to the detector. The
    batch transformations apply a fixed reference independently to each profile.
    """
    index = df.index
    required_start = pd.Timestamp("2010-01-01 00:30")
    if not isinstance(index, pd.DatetimeIndex) or len(df) % N_PERIODS:
        raise ValueError("Expected complete, end-stamped market days in a DatetimeIndex.")
    expected = pd.date_range(required_start, periods=len(df), freq="30min")
    if not index.equals(expected) or index.empty or index[-1] < pd.Timestamp("2015-01-02"):
        raise ValueError("Expected consecutive half-hours from January 2010 through at least 1 January 2015.")

    days = market_day(index[::N_PERIODS])
    profile_days, forecast_days = days[:-1], days[1:]
    profiles = df["nsw_demand"].iloc[:-N_PERIODS].to_numpy(dtype=float).reshape(-1, N_PERIODS)
    if not (np.isfinite(profiles) & (profiles > 0)).all():
        raise ValueError("The lagged NSW demand profiles must contain finite, positive observations.")
    shape = profiles / profiles.mean(axis=1, keepdims=True)
    reference = profile_days.year <= 2013

    # Match the fixed day types in bryan_package/data_analysis.py, including
    # K-means initialisation and PCA.fit_transform on the training profiles.
    with threadpool_limits(limits=1):
        pca = PCA(n_components=5, random_state=42)
        train_scores = pca.fit_transform(shape[reference])
        clusters = KMeans(n_clusters=4, n_init=25, random_state=42).fit(train_scores)
        distances = clusters.transform(pca.transform(shape))

    nearest = distances.argmin(axis=1)
    distance = distances[np.arange(len(nearest)), nearest]
    # A summer profile is compared with the typical training-period summer
    # distance for its own calendar month, including across New Year's Day.
    historical = pd.Series(distance[reference], index=profile_days[reference])
    month_means = historical.groupby(historical.index.month).mean()
    if not (month_means > 0).all():
        raise ValueError("Every reference month must have a positive mean profile distance.")
    ratio = distance / month_means.loc[profile_days.month].to_numpy()
    history = pd.DataFrame({
        "profile_date": profile_days, "cluster_id": nearest + 1,
        "daily_distance": distance, "seasonal_distance_ratio": ratio,
    }, index=forecast_days.rename("market_day"))

    # The first 2014 issue day has a 2013 reference profile. Exclude it so every
    # calibration profile and every calibration decision come from 2014.
    calibration = history.loc["2014-01-02":"2014-12-31", "seasonal_distance_ratio"]
    alarm = RollingUpperAlarm(calibration)
    simulation = history.loc["2015-01-01":].copy()
    scores, signals = [], []
    for day, observation in simulation["seasonal_distance_ratio"].items():
        signals.append(alarm.observe(day, observation))
        scores.append(alarm.statistic)
    simulation["input_statistic"] = scores
    simulation["input_threshold"] = alarm.threshold
    simulation["input_signal"] = signals
    return simulation

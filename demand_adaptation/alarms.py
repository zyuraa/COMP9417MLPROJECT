"""End-of-market-day signals used by the demand retraining simulation."""

from collections import deque
from math import fsum, isfinite

import numpy as np
import pandas as pd


class RollingUpperAlarm:
    """Signal when a trailing daily mean exceeds its fixed 2014 reference quantile."""

    def __init__(self, calibration, window=28, quantile=0.95, cooldown_days=365):
        if window < 1 or not 0 < quantile < 1 or cooldown_days < 1:
            raise ValueError("Expected a positive window and cooldown and a quantile between 0 and 1.")
        dates = pd.DatetimeIndex(calibration.index)
        if len(dates) < window or dates.tz is not None or not (dates.year == 2014).all():
            raise ValueError("Use a chronological 2014 series of daily values to calibrate the alarm.")
        expected = pd.date_range(dates[0].normalize(), periods=len(dates), freq="D")
        if not (dates == expected).all():
            raise ValueError("Calibration needs one value per consecutive market day, in date order.")
        values = calibration.to_numpy(dtype=float)
        if not (np.isfinite(values) & (values >= 0)).all():
            raise ValueError("Calibration values must be finite and nonnegative.")

        self.window = window
        self.quantile = quantile
        self.cooldown_days = cooldown_days
        means = [fsum(values[end - window : end]) / window for end in range(window, len(values) + 1)]
        self.threshold = float(np.quantile(means, quantile))
        self.statistic = means[-1]
        self._values = deque(values[-window:], maxlen=window)
        self._last_day = dates[-1]
        self._last_alarm = None

    def observe(self, day, value):
        """Return True at the close of an eligible day above the calibrated threshold."""
        day = pd.Timestamp(day)
        if day != self._last_day + pd.Timedelta(days=1):
            raise ValueError("Observe each market day once, in chronological order.")
        if not isfinite(value) or value < 0:
            raise ValueError("Daily values must be finite and nonnegative.")
        self._last_day = day
        self._values.append(float(value))
        self.statistic = fsum(self._values) / self.window
        eligible = self._last_alarm is None or (day - self._last_alarm).days >= self.cooldown_days
        if eligible and self.statistic > self.threshold:
            self._last_alarm = day
            return True
        return False


class ErrorAlarm(RollingUpperAlarm):
    """Monitor the deployed model's daily MAPE, using 2014 out-of-sample calibration errors."""

    def __init__(self, calibration_daily_mape, window=28, quantile=0.95, cooldown_days=365):
        super().__init__(calibration_daily_mape, window, quantile, cooldown_days)

    def observe(self, day, daily_mape):
        return super().observe(day, daily_mape)


def read_alarm_dates(path):
    """Read an external detector's dates (ISO YYYY-MM-DD) at which it signalled.

    These are observation dates. Any resulting refit applies to the following
    market day. A CSV with just the alarm_date header means no signals.
    """
    frame = pd.read_csv(path, dtype=str)
    if "alarm_date" not in frame:
        raise ValueError("External alarm CSV must contain an alarm_date column.")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["alarm_date"], format="%Y-%m-%d", errors="raise"))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("External alarm dates must be nonempty, unique and in chronological order.")
    return dates

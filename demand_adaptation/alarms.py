"""End-of-market-day signals used by the demand retraining simulation."""

from collections import deque
from math import fsum, isfinite

import numpy as np
import pandas as pd


class ErrorAlarm:
    """Monitor trailing daily MAPE against a threshold calibrated on 2014.

    Calibration errors must come from forecasts of 2014 made by the model
    trained through 2013. Each call to observe() receives just one newly
    observed market day and the error of the model deployed for that day.
    """

    def __init__(self, calibration_daily_mape, window=28, quantile=0.95, cooldown_days=365):
        if window < 1 or not 0 < quantile < 1 or cooldown_days < 1:
            raise ValueError("Expected a positive window and cooldown and a quantile between 0 and 1.")
        dates = pd.DatetimeIndex(calibration_daily_mape.index)
        if len(dates) < window or dates.tz is not None or not (dates.year == 2014).all():
            raise ValueError("Use a chronological 2014 series of daily validation MAPE to calibrate the alarm.")
        expected = pd.date_range(dates[0].normalize(), periods=len(dates), freq="D")
        if not (dates == expected).all():
            raise ValueError("Calibration needs one value per consecutive market day, in date order.")
        errors = calibration_daily_mape.to_numpy(dtype=float)
        if not (np.isfinite(errors) & (errors >= 0)).all():
            raise ValueError("Calibration MAPE must be finite and nonnegative.")

        self.window = window
        self.quantile = quantile
        self.cooldown_days = cooldown_days
        means = [fsum(errors[end - window : end]) / window for end in range(window, len(errors) + 1)]
        self.threshold = float(np.quantile(means, quantile))
        self.statistic = means[-1]
        self._errors = deque(errors[-window:], maxlen=window)
        self._last_day = dates[-1]
        self._last_alarm = None

    def observe(self, day, daily_mape):
        """Return True at the close of an eligible day above the calibrated threshold."""
        day = pd.Timestamp(day)
        if day != self._last_day + pd.Timedelta(days=1):
            raise ValueError("Observe each market day once, in chronological order.")
        if not isfinite(daily_mape) or daily_mape < 0:
            raise ValueError("Daily MAPE must be finite and nonnegative.")
        self._last_day = day
        self._errors.append(float(daily_mape))
        self.statistic = fsum(self._errors) / self.window
        eligible = self._last_alarm is None or (day - self._last_alarm).days >= self.cooldown_days
        if eligible and self.statistic > self.threshold:
            self._last_alarm = day
            return True
        return False


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

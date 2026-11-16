"""Pair the online input and frozen-model error histories on the same market days."""

import json

from demand_adaptation.alarms import ErrorAlarm
from .outputs import write_diagnostics
from .profiles import input_history


def compare_detectors(df, frozen_errors, calibration, output, *, plots=True):
    daily = input_history(df)
    if not daily.index.equals(frozen_errors.index):
        raise ValueError("Demand inputs and frozen-model errors must cover identical dates.")
    daily["frozen_mape_pct"] = frozen_errors
    alarm = ErrorAlarm(calibration)
    scores, signals = [], []
    for day, error in frozen_errors.items():
        signals.append(alarm.observe(day, error))
        scores.append(alarm.statistic)
    daily["error_statistic"] = scores
    daily["error_threshold"] = alarm.threshold
    daily["error_signal"] = signals

    summary = write_diagnostics(daily, output, plots=plots)
    daily.to_csv(output / "daily_diagnostics.csv")
    input_dates = daily.index[daily["input_signal"]]
    input_dates.to_frame(name="alarm_date", index=False).to_csv(output / "input_alarm_dates.csv", index=False)
    summary["input_threshold_ratio"] = float(daily["input_threshold"].iloc[0])
    summary["error_threshold_mape_pct"] = alarm.threshold
    (output / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return set(input_dates)

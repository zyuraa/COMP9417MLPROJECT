"""Compare frozen, scheduled and alarm-triggered NSW demand forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .alarms import ErrorAlarm, read_alarm_dates
from .lasso import prepare_lasso
from .results import write_results


HERE = Path(__file__).resolve().parent
INITIAL_START = pd.Timestamp("2010-01-01")
START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-10-31")
ONE_DAY = pd.Timedelta(days=1)
MAX_REFITS = 6
WINDOW = pd.DateOffset(years=4)
# Six dates fixed before scoring, approximately two years apart.
SCHEDULE = {pd.Timestamp(year, 1, 1) - ONE_DAY for year in range(2016, 2027, 2)}


def daily_mape(actual, predicted):
    return 100 * np.mean(np.abs(actual - predicted) / actual, axis=-1)


def validation_errors(df):
    through_2014 = df.loc[:"2015-01-01 00:00"]
    data = prepare_lasso(through_2014, transformer_year=2013)
    model = data.fit(INITIAL_START, "2013-12-31")
    prediction = data.predict(model, "2014-01-01", "2014-12-31")
    mask = data.dates.year == 2014
    return pd.Series(daily_mape(data.y[mask], prediction), index=data.dates[mask])


def simulate(strategy, data, initial, days, actual, signals, error_alarm=None):
    """End-of-day decisions only see outcomes through the day being scored."""
    forecasts = np.empty_like(actual)
    events, trace = [], []
    model = initial
    start = 0
    while start < len(days):
        # Predict with a fixed fit using causal features for each row. Discard the
        # remaining batch immediately when a refit changes tomorrow's forecast.
        stop = min(start + 90, len(days))
        batch = data.predict(model, days[start], days[stop - 1])
        for offset, prediction in enumerate(batch):
            i = start + offset
            day = days[i]
            forecasts[i] = prediction
            error = float(daily_mape(actual[i], prediction))
            signal = error_alarm.observe(day, error) if error_alarm is not None else day in signals
            apply = signal and len(events) < MAX_REFITS and i + 1 < len(days)
            if strategy == "triggered":
                trace.append({
                    "market_day": day, "daily_mape_pct": error,
                    "rolling_mape_pct": error_alarm.statistic if error_alarm is not None else None,
                    "threshold_pct": error_alarm.threshold if error_alarm is not None else None,
                    "signal_at_close": signal, "refit_for_tomorrow": apply,
                })
            if apply:
                effective = day + ONE_DAY
                since = effective - WINDOW
                model = data.fit(since, day)
                events.append({"strategy": strategy, "decision_date": day, "effective_date": effective,
                               "train_start": since, "train_end": day, "refit_number": len(events) + 1})
                print(f"{strategy}: refit {len(events)} takes effect {effective.date()}", flush=True)
                stop = i + 1
                break
        start = stop
    return forecasts, events, trace


def run(csv, output, alarm_dates, plots, profile_drift=False):
    if profile_drift and alarm_dates is not None:
        raise ValueError("Choose either the profile drift comparison or external alarm dates.")
    df = pd.read_csv(csv, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    if df.index[0] != pd.Timestamp("2010-01-01 00:30") or df.index[-1] != END + ONE_DAY:
        raise ValueError("Expected the complete 2010–October 2026 course CSV.")
    days = pd.date_range(START, END)
    supplied = alarm_dates is not None
    signals = set(read_alarm_dates(alarm_dates)) if supplied else set()
    if not signals.issubset(days):
        raise ValueError("External alarm dates must fall within the historical simulation.")
    cal = validation_errors(df)
    alarm = None if supplied else ErrorAlarm(cal)
    print(f"Reproduced 2014 Lasso validation MAPE: {cal.mean():.5f}%", flush=True)
    data = prepare_lasso(df)
    frozen = data.fit(INITIAL_START, START - ONE_DAY)
    day_indices = data.dates.get_indexer(days)
    if np.any(day_indices < 7):
        raise ValueError("The simulation requires all 48 slots and seven complete prior days.")
    actual = data.y[day_indices]
    forecasts = {
        "frozen": data.predict(frozen, START, END),
        "yesterday": data.y[day_indices - 1],
        "last_week": data.y[day_indices - 7],
    }
    forecasts["scheduled"], scheduled, _ = simulate("scheduled", data, frozen, days, actual, SCHEDULE)
    forecasts["triggered"], triggered, trace = simulate("triggered", data, frozen, days, actual, signals, alarm)
    output.mkdir(parents=True, exist_ok=True)
    refits = scheduled + triggered
    candidates = ["frozen", "scheduled", "triggered"]
    if profile_drift:
        from demand_drift.compare import compare_detectors
        errors = pd.Series(daily_mape(actual, forecasts["frozen"]), index=days)
        input_signals = compare_detectors(df, errors, cal, output, plots=plots)
        forecasts["input_triggered"], input_refits, _ = simulate(
            "input_triggered", data, frozen, days, actual, input_signals)
        refits.extend(input_refits)
        candidates.append("input_triggered")
    events = pd.DataFrame(refits)
    summary = write_results(output, days, actual, forecasts, events, plots=plots)
    events.to_csv(output / "refits.csv", index=False)
    pd.DataFrame(trace).to_csv(output / "alarm_history.csv", index=False)
    winner = (summary.loc[summary["strategy"].isin(candidates)]
              .sort_values(["mape_pct", "mae_mw", "retrains"]).iloc[0]["strategy"])
    # For final deployment, apply the selected training-window rule at the release cutoff.
    since = INITIAL_START if winner == "frozen" else END + ONE_DAY - WINDOW
    until = START - ONE_DAY if winner == "frozen" else END
    settings = {
        "selected_strategy": winner, "train_start": str(since.date()), "train_end": str(until.date()),
        "reference_transformer_through": 2014, "refit_window_years": 4, "max_refits_per_strategy": MAX_REFITS,
        "alarm_source": "external CSV" if supplied else "reference demand error alarm",
        "validation_2014_mape_pct": float(cal.mean()),
    }
    if alarm is not None:
        settings.update(alarm_window_days=alarm.window, alarm_quantile=alarm.quantile,
                        alarm_cooldown_days=alarm.cooldown_days, alarm_threshold_pct=alarm.threshold)
    if profile_drift:
        settings["input_alarm_source"] = "lagged demand profile detector (see diagnostic_summary.json)"
    (output / "selection.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    from .export import export_demand
    export_demand(output, data, data.fit(since, until) if winner != "frozen" else frozen, settings)
    print(summary.to_string(index=False), flush=True)
    print(f"Selected {winner} for demand. Final fit: {since.date()} through {until.date()}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--alarm-dates", type=Path, help="CSV of chronological end-of-day alarm_date values.")
    source.add_argument("--profile-drift", action="store_true",
                        help="Compare input-profile and error alarms, then evaluate both retraining policies.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    default_output = HERE / ("results_external" if args.alarm_dates else "results")
    if args.profile_drift:
        default_output = HERE.parent / "demand_drift" / "results"
    run(args.csv, args.output_dir or default_output, args.alarm_dates, not args.no_plots, args.profile_drift)

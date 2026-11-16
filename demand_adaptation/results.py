"""Summarise matched demand forecasts over complete, end-stamped market days."""

from pathlib import Path

import numpy as np
import pandas as pd

from boosting_package.evaluate import BOOTSTRAP_REPLICATES, SEED, monthly_intervals, period_metrics


LABELS = {
    "frozen": "Frozen Lasso",
    "scheduled": "Scheduled Lasso",
    "triggered": "Alarm-triggered Lasso",
    "yesterday": "Same half-hour yesterday",
    "last_week": "Same half-hour last week",
}


def plot_comparison(monthly, days, events, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    styles = {
        "frozen": ("#4e536b", "--"),
        "scheduled": ("#226b9e", "-"),
        "triggered": ("#ac4d22", "-"),
    }
    fig, (scores, counts) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 6.8), layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for key, (color, line) in styles.items():
        subset = monthly.loc[monthly["strategy"] == key]
        month_start = pd.to_datetime(subset["month"], format="%Y-%m")
        scores.plot(month_start, subset["mape_pct"], color=color, linestyle=line,
                    linewidth=1.5, label=LABELS[key])
        scores.fill_between(month_start, subset["mape_day_bootstrap_low_pct"],
                            subset["mape_day_bootstrap_high_pct"], color=color, alpha=0.13)
    scores.set_title("Monthly NSW demand error with 95% whole-day bootstrap intervals", loc="left")
    scores.set_ylabel("Demand MAPE (%)")
    scores.legend(loc="best", ncol=3, frameon=False)

    for key, marker in [("scheduled", "o"), ("triggered", "D")]:
        active_dates = events.loc[events["strategy"] == key, "effective_date"].sort_values()
        color, _ = styles[key]
        accumulated = np.cumsum(days.isin(active_dates))
        counts.step(days, accumulated, where="post", color=color, linewidth=1.5,
                    linestyle="--" if key == "scheduled" else "-", label=LABELS[key])
        counts.scatter(active_dates, np.arange(1, len(active_dates) + 1), marker=marker,
                       color=color, s=30, zorder=3)
    counts.set_title("Retrains by the first forecast made with each new fit", loc="left")
    counts.set_ylabel("Cumulative retrains")
    counts.set_yticks(range(0, 7, 2))
    counts.set_ylim(-0.4, 6.6)
    counts.set_xlabel("Market date (AEST)")
    counts.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    counts.xaxis.set_major_formatter(mdates.ConciseDateFormatter(counts.xaxis.get_major_locator()))
    for axis in (scores, counts):
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_results(output_dir, market_days, observed, forecasts, retraining_log, *, plots=True):
    """Write metrics and return the overall comparison, using one row per market day."""
    days = pd.DatetimeIndex(market_days)
    if days.empty or not days.equals(pd.date_range(days[0].normalize(), periods=len(days), freq="D")):
        raise ValueError("Market days must be consecutive calendar dates in chronological order.")
    actual = np.asarray(observed, dtype=float)
    if actual.shape != (len(days), 48) or not np.isfinite(actual).all() or np.any(actual <= 0):
        raise ValueError("Each market day must contain 48 finite, positive NSW demand observations.")
    if set(forecasts) != set(LABELS):
        raise ValueError(f"Expected demand forecasts for {', '.join(LABELS)}.")

    events = pd.DataFrame(retraining_log).reindex(columns=["strategy", "effective_date"]).copy()
    events["effective_date"] = pd.to_datetime(events["effective_date"])
    if (not events["strategy"].isin(["scheduled", "triggered"]).all()
            or not events["effective_date"].isin(days).all()
            or events.duplicated().any() or (events.groupby("strategy").size() > 6).any()):
        raise ValueError("Expected up to six distinct effective dates within the simulation per strategy.")

    overall_rows, yearly_rows, monthly_rows = [], [], []
    years, months = days.year, days.to_period("M")
    for key, label in LABELS.items():
        pred = np.asarray(forecasts[key], dtype=float)
        if pred.shape != actual.shape or not np.isfinite(pred).all():
            raise ValueError(f"{label} must provide a finite prediction for every observed half-hour.")
        errors = np.abs(actual - pred)
        pct = 100.0 * errors / actual
        effective = events.loc[events["strategy"] == key, "effective_date"]
        row = {"strategy": key, "model": label}
        overall_rows.append({**row, **period_metrics(errors, pct), "retrains": len(effective)})
        for year in years.unique():
            match = years == year
            yearly_rows.append({**row, "year": year, **period_metrics(errors[match], pct[match]),
                                "retrains": int((effective.dt.year == year).sum())})

        # Restart the generator so every method uses the same draws in each month.
        rng = np.random.default_rng(SEED)
        for month in months.unique():
            match = months == month
            sample = pct[match]
            draws = rng.integers(len(sample), size=(BOOTSTRAP_REPLICATES, len(sample)))
            monthly_rows.append({
                **row, "month": str(month), **period_metrics(errors[match], sample),
                **monthly_intervals(sample, draws),
                "retrains": int((effective.dt.to_period("M") == month).sum()),
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame(overall_rows)
    monthly = pd.DataFrame(monthly_rows)
    overall.to_csv(output_dir / "overall.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(output_dir / "yearly.csv", index=False)
    monthly.to_csv(output_dir / "monthly.csv", index=False)
    if plots:
        plot_comparison(monthly, days, events, output_dir / "demand_adaptation.png")
    return overall

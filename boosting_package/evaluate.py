#Score the frozen Stage 1 demand model on the January 2015–October 2026 simulation

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .features import make_features, market_day
    from .model import load_demand_model
else:
    from features import make_features, market_day
    from model import load_demand_model


PACKAGE_DIR = Path(__file__).resolve().parent
SLOTS_PER_DAY = 48
BOOTSTRAP_REPLICATES = 1000
SEED = 42


def period_metrics(absolute_errors, percentage_errors):
    return {
        "n_half_hours": absolute_errors.size,
        "mape_pct": percentage_errors.mean(),
        "mae_mw": absolute_errors.mean(),
    }


def monthly_intervals(percentage_errors, draws):
    # Each row is one complete market day. Drawing its mean gives the same
    # monthly MAPE as drawing and concatenating all 48 errors from that day.
    daily_mape = percentage_errors.mean(axis=1)
    low, high = np.percentile(daily_mape[draws].mean(axis=1), [2.5, 97.5])
    mape = daily_mape.mean()
    half_width = 1.96 * percentage_errors.std(ddof=1) / np.sqrt(percentage_errors.size)
    return {
        "mape_independent_low_pct": mape - half_width,
        "mape_independent_high_pct": mape + half_width,
        "mape_day_bootstrap_low_pct": low,
        "mape_day_bootstrap_high_pct": high,
    }


def plot_results(monthly, index, actual, forecasts, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    boost_name = "Frozen gradient boosting"
    yesterday = "Same half-hour yesterday"
    last_week = "Same half-hour last week"
    boost = monthly.loc[monthly["model"] == boost_name]
    dates = pd.to_datetime(boost["month"], format="%Y-%m").to_numpy()
    teal = "#00695c"

    fig, (interval_ax, comparison_ax) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, layout="constrained"
    )
    interval_ax.set_title("Monthly forecast error and 95% intervals for frozen gradient boosting")
    interval_ax.fill_between(
        dates, boost["mape_day_bootstrap_low_pct"], boost["mape_day_bootstrap_high_pct"],
        color="#a6dbd2", label="Whole-day bootstrap (1,000 draws)",
    )
    interval_ax.fill_between(
        dates, boost["mape_independent_low_pct"], boost["mape_independent_high_pct"],
        color="#4a9e92", label="Independent half-hours",
    )
    interval_ax.plot(dates, boost["mape_pct"], color=teal, linewidth=1.4, label=boost_name)
    interval_ax.legend(frameon=False, ncol=3, fontsize=9)

    comparison_ax.set_title("NSW demand forecasts over the historical simulation")
    for name, color, style in [(yesterday, "#68727b", "--"), (last_week, "#9a7955", ":")]:
        baseline = monthly.loc[monthly["model"] == name]
        comparison_ax.plot(
            pd.to_datetime(baseline["month"], format="%Y-%m"), baseline["mape_pct"],
            color=color, linestyle=style, linewidth=1.2, label=name,
        )
    lasso_path = (PACKAGE_DIR.parent / "bryan_package" / "analysis_outputs"
                  / "table9_stage2_frozen_monthly_ci.csv")
    if lasso_path.is_file():
        published = pd.read_csv(lasso_path, usecols=["month", "model", "mape"])
        lasso = published.loc[published["model"] == "Per-Slot Lasso (L1) [Selected]"]
        comparison_ax.plot(
            pd.to_datetime(lasso["month"], format="%Y-%m"), lasso["mape"], color="#7e5496",
            linewidth=1.2, label="Frozen per-slot Lasso (Bryan)",
        )
    comparison_ax.plot(dates, boost["mape_pct"], color=teal, linewidth=1.6, label=boost_name)
    comparison_ax.set_xlabel("Market month")
    comparison_ax.xaxis.set_major_locator(mdates.YearLocator())
    comparison_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    comparison_ax.legend(frameon=False, ncol=2, fontsize=9)
    for ax in (interval_ax, comparison_ax):
        ax.set_ylabel("Demand MAPE (%)")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "stage2_frozen_monthly_mape.png", dpi=180)
    plt.close(fig)

    # Preselected illustration. The final market day ends at midnight on 8 July.
    week = (index > pd.Timestamp("2026-07-01")) & (index <= pd.Timestamp("2026-07-08"))
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.plot(index[week], forecasts[yesterday][week], color="#867664", linestyle="--",
            linewidth=1.1, label=yesterday)
    ax.plot(index[week], actual[week], color="#27323c", linewidth=1.8, label="Observed demand")
    ax.plot(index[week], forecasts[boost_name][week], color=teal, linewidth=1.5, label=boost_name)
    ax.set_title("NSW demand over seven market days, 1–7 July 2026")
    
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Half-hour ending (AEST)")
    
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.savefig(output_dir / "stage2_frozen_example_week.png", dpi=180)
    plt.close(fig)


def evaluate(csv_path, artifact_dir, output_dir, plots=False):
    data = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    
    # Keep earlier rows during feature construction so the first forecast of
    # January 2015 can use its December 2014 history.
    expected_index = pd.date_range("2015-01-01 00:30", "2026-11-01 00:00", freq="30min")
    
    features = make_features(data).loc[expected_index[0]:expected_index[-1]]
    if not features.index.equals(expected_index):
        raise ValueError("Stage 2 needs all 48 end-stamped intervals of every market day.")
    
    actual = data.loc[features.index, "nsw_demand"].to_numpy(dtype=float)
    days = market_day(features.index[::SLOTS_PER_DAY])
    months = days.to_period("M")

    frozen = load_demand_model(artifact_dir)
    
    # The same fitted estimator is used in every batch. Batching limits the
    # estimator's temporary memory when predicting the entire simulation.
    batch_size = 31 * SLOTS_PER_DAY
    boosted = np.concatenate([
        frozen.predict(features.iloc[start:start + batch_size])
        for start in range(0, len(features), batch_size)
    ])
    
    forecasts = {
        "Frozen gradient boosting": boosted,
        "Same half-hour yesterday": features["demand_lag1"].to_numpy(),
        "Same half-hour last week": features["demand_lag7"].to_numpy(),
    }

    monthly_rows, yearly_rows, overall_rows = [], [], []
    
    for name, prediction in forecasts.items():
        if prediction.shape != actual.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{name} must return one finite demand forecast per half-hour.")
    
        absolute_errors = np.abs(actual - prediction)
        percentage_errors = (100 * absolute_errors / np.abs(actual)).reshape(-1, SLOTS_PER_DAY)
        absolute_errors = absolute_errors.reshape(-1, SLOTS_PER_DAY)
        overall_rows.append({"model": name, **period_metrics(absolute_errors, percentage_errors)})
    
        for year in days.year.unique():
            mask = days.year == year
            yearly_rows.append({
                "model": name,
                "year": year,
                **period_metrics(absolute_errors[mask], percentage_errors[mask]),
            })
    
        # Reuse the same resampled days across methods for each month.
        rng = np.random.default_rng(SEED)
        for month in months.unique():
            mask = months == month
            sample = percentage_errors[mask]
            n_days = len(sample)
            draws = rng.integers(n_days, size=(BOOTSTRAP_REPLICATES, n_days))
            monthly_rows.append({
                "model": name,
                "month": str(month),
                **period_metrics(absolute_errors[mask], sample),
                **monthly_intervals(sample, draws),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(yearly_rows).to_csv(output_dir / "stage2_frozen_yearly.csv", index=False)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output_dir / "stage2_frozen_monthly.csv", index=False)
    
    if plots:
        plot_results(monthly, features.index, actual, forecasts, output_dir)
    print("Frozen demand forecasts, January 2015 through October 2026")
    print(pd.DataFrame(overall_rows).to_string(index=False, float_format="{:.4f}".format))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Released half-hourly market and weather CSV")
    parser.add_argument("--artifact-dir", type=Path, default=PACKAGE_DIR / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "results")
    parser.add_argument("--plots", action="store_true", help="Save the monthly MAPE and example-week figures")
    args = parser.parse_args()
    evaluate(args.csv_path, args.artifact_dir, args.output_dir, plots=args.plots)


if __name__ == "__main__":
    main()

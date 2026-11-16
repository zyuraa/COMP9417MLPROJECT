"""Compare online demand-profile signals with errors from the fixed Stage 1 Lasso."""

from pathlib import Path

import numpy as np
import pandas as pd


def plot_timeline(daily, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10.8, 6.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.16, hspace=0.40)
    fig.suptitle("Online drift signals in NSW demand forecasting", fontsize=17, y=0.975)
    panels = (
        ("input", "Demand profile available before the forecast", "Distance ratio", "#00766d", ""),
        ("error", "Frozen Lasso forecast error", "Demand MAPE (%)", "#864363", "%"),
    )
    for ax, (key, title, unit, color, suffix) in zip(axes, panels):
        statistic = daily[f"{key}_statistic"]
        threshold = float(daily[f"{key}_threshold"].iloc[0])
        alarms = statistic[daily[f"{key}_signal"]]
        ax.fill_between(daily.index, threshold, statistic, where=statistic > threshold,
                        interpolate=True, color=color, alpha=0.12)
        ax.plot(statistic, color=color, linewidth=1.15)
        ax.axhline(threshold, color="#424b54", linestyle="--", linewidth=1.1)
        ax.scatter(alarms.index, alarms, marker="D", s=30, color=color,
                   edgecolors="white", linewidths=0.75, zorder=3, clip_on=False)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=9)
        ax.set_title(f"2014 threshold  {threshold:.2f}{suffix}", loc="right",
                     fontsize=10, color="#454e56", pad=9)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#d7dcdf", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01, y=0.10)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Market date (AEST)")
    proxies = [
        Line2D([], [], color="#455664", label="28-day mean"),
        Line2D([], [], color="#424b54", linestyle="--", label="Calibrated threshold"),
        Line2D([], [], color="#455664", marker="D", linestyle="none",
               label="End-of-day alarm after cooldown"),
    ]
    fig.legend(handles=proxies, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=3, frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def write_diagnostics(daily, output_dir, *, plots=True):
    """Write matched-day comparisons and return counts over the whole evaluation."""
    days = pd.DatetimeIndex(daily.index)
    if days.empty or not days.equals(pd.date_range(days[0].normalize(), periods=len(days), freq="D")):
        raise ValueError("Diagnostics need one chronological observation per market day.")
    numeric = ["input_statistic", "input_threshold", "error_statistic", "error_threshold", "frozen_mape_pct"]
    if not np.isfinite(daily[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Both detectors and the frozen model must cover the same evaluation days.")
    if not daily["cluster_id"].isin([1, 2, 3, 4]).all():
        raise ValueError("Expected lagged demand profile clusters numbered 1 to 4.")

    measures = daily[["input_statistic", "error_statistic", "frozen_mape_pct"]].copy()
    states = pd.DataFrame(index=days)
    alarms = []
    for key in ("input", "error"):
        score, threshold, signal = (daily[f"{key}_{column}"] for column in ("statistic", "threshold", "signal"))
        if threshold.nunique() != 1 or not pd.api.types.is_bool_dtype(signal) or signal.isna().any():
            raise ValueError(f"Expected a fixed threshold and boolean daily signals for {key}.")
        states[f"{key}_high"] = score > threshold
        if (signal & ~states[f"{key}_high"]).any():
            raise ValueError(f"{key} signalled on a day below or equal to its threshold.")
        dates = days[signal]
        alarms.append(pd.DataFrame({"detector": key, "alarm_date": dates,
                                    "next_forecast_date": dates + pd.Timedelta(days=1),
                                    "statistic": score[signal].to_numpy(),
                                    "threshold": threshold[signal].to_numpy()}))
    states["both_high"] = states["input_high"] & states["error_high"]
    states["input_only"] = states["input_high"] & ~states["error_high"]
    states["error_only"] = states["error_high"] & ~states["input_high"]
    states["neither_high"] = ~states["input_high"] & ~states["error_high"]

    month = days.to_period("M")
    monthly = measures.groupby(month).mean().add_prefix("mean_")
    monthly.insert(0, "n_days", measures.groupby(month).size())
    fractions = states.groupby(month).mean().add_suffix("_fraction")
    shares = pd.crosstab(month, daily["cluster_id"], normalize="index")
    shares = shares.reindex(columns=range(1, 5), fill_value=0.0)
    shares.columns = [f"lagged_cluster_{key}_fraction" for key in shares.columns]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.join(fractions).join(shares).to_csv(output_dir / "monthly_diagnostics.csv", index_label="month")
    pd.concat(alarms, ignore_index=True).sort_values(["alarm_date", "detector"]).to_csv(
        output_dir / "alarm_dates.csv", index=False, date_format="%Y-%m-%d")
    if plots:
        plot_timeline(daily, output_dir / "drift_timeline.png")
    return {
        "n_days": len(days),
        **{f"{key}_days": int(states[key].sum()) for key in states},
        "input_alarms": int(daily["input_signal"].sum()), "error_alarms": int(daily["error_signal"].sum()),
    }

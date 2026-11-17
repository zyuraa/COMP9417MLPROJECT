"""
run_input_drift.py - Runs ONLY the Input-Based Drift Detector (Covariate Shift P_t(X)).

Requires only the raw half-hourly CSV and drift.py (no model predictions needed).
Outputs:
  1. analysis_outputs/input_drift_alarms.json               (handoff for retraining)
  2. analysis_outputs/input_drift_daily_series.csv          (daily score & CUSUM path)
  3. analysis_outputs/input_drift_feature_contributions.csv (which features drove each alarm)
  4. analysis_outputs/fig_input_drift_detector.png          (plot for Section 4 of report)
"""
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import holidays as hol_pkg

import drift


def main():
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join("..", "data", "nem_weather_halfhourly_2010_2026.csv")
    )
    if not os.path.exists(csv_path) and os.path.exists("nem_weather_halfhourly_2010_2026.csv"):
        csv_path = "nem_weather_halfhourly_2010_2026.csv"

    out_dir = "analysis_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load raw dataset and NSW public holidays (permitted by project spec)
    print(f"[1/4] Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    nsw_holidays = hol_pkg.country_holidays("AU", subdiv="NSW", years=range(2010, 2027))

    # 2. Build daily input features (11 daily summaries known at the end of Day D)
    print("[2/4] Computing daily input features and fitting 2010-2013 seasonal model...")
    feats = drift.daily_input_features(df)

    # Fit reference on 2010-2013, calibrate threshold h on 2014 (margin=1.1, k=0.5)
    # Cooldown=60 days and max_alarms=6 respect the Stage 2 6-retrain budget
    ref_period = ("2010-01-01", "2013-12-31")
    cal_period = ("2014-01-01", "2014-12-31")
    k_slack = 0.5
    margin = 1.1
    cooldown_days = 60
    max_alarms = 6

    sc, det_in, input_score = drift.build_input_detector(
        feats,
        ref=ref_period,
        cal=cal_period,
        k=k_slack,
        margin=margin,
        holidays=nsw_holidays,
        cooldown=cooldown_days,
        max_alarms=max_alarms,
    )

    # 3. Run online CUSUM detector day-by-day over Stage 2 (Jan 2015 - Oct 2026)
    print("[3/4] Running online input CUSUM detector across 2015-2026...")
    sim_score = input_score.loc["2015-01-01":]
    cusum_path, alarm_dates = det_in.run(sim_score)
    alarm_strs = [str(d.date()) for d in alarm_dates]

    # Print exact parameters required by the COMP9417 rubric
    print("\n=== INPUT-BASED DRIFT DETECTOR SPECIFICATION (FOR REPORT) ===")
    print(f"Monitored statistic : Mean squared deseasonalised z-score across {len(feats.columns)} daily input features")
    print(f"Reference period    : {ref_period[0]} to {ref_period[1]} (mu = {det_in.mu:.4f}, sigma = {det_in.sigma:.4f})")
    print(f"Calibration period  : {cal_period[0]} to {cal_period[1]} (k = {det_in.k}, margin = {margin})")
    print(f"Decision threshold  : h = {det_in.h:.4f} (cooldown = {cooldown_days} days, max_alarms = {max_alarms})")
    print(f"Triggered alarms ({len(alarm_strs)}): {alarm_strs}")

    # Analyze which input features drove each alarm (90-day window up to alarm date)
    z_all = sc.zscores(feats)
    contrib_rows = []
    print("\n=== TOP 3 DRIVING INPUT FEATURES AT EACH ALARM DATE ===")
    for d in alarm_dates:
        w_start = d - pd.Timedelta(days=60)
        contrib = drift.feature_contributions(z_all, w_start, d)
        top3 = ", ".join([f"{col} (z^2={val:.2f})" for col, val in contrib.head(3).items()])
        print(f"  Alarm {d.date()} -> {top3}")
        row = {"alarm_date": str(d.date()), "retrain_effective_day": str((d + pd.Timedelta(days=1)).date())}
        row.update({col: float(val) for col, val in contrib.items()})
        contrib_rows.append(row)

    # 4. Save outputs for teammates & generate report figure
    print(f"\n[4/4] Saving alarm dates, CSVs, and plot to {out_dir}/...")
    handoff_payload = {
        "detector_type": "input_cusum",
        "reference_period": list(ref_period),
        "calibration_period": list(cal_period),
        "mu_ref": float(det_in.mu),
        "sigma_ref": float(det_in.sigma),
        "k_allowance": float(det_in.k),
        "threshold_h": float(det_in.h),
        "cooldown_days": cooldown_days,
        "max_alarms": max_alarms,
        "alarm_dates_end_of_day_D": alarm_strs,
        "retrain_effective_dates_day_D_plus_1": [
            str((d + pd.Timedelta(days=1)).date()) for d in alarm_dates
        ],
    }
    with open(os.path.join(out_dir, "input_drift_alarms.json"), "w", encoding="utf-8") as f:
        json.dump(handoff_payload, f, indent=2)

    pd.DataFrame(contrib_rows).to_csv(
        os.path.join(out_dir, "input_drift_feature_contributions.csv"), index=False
    )

    daily_out = pd.DataFrame({
        "input_drift_score": input_score,
        "cusum_S": cusum_path.reindex(input_score.index),
    })
    daily_out.to_csv(os.path.join(out_dir, "input_drift_daily_series.csv"))

    # Plot 2-panel figure for Report Section 4
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    # Panel (a): Daily Input Drift Score (2010-2026) with 30-day rolling mean
    ax0 = axes[0]
    ax0.plot(input_score.index, input_score, color="lightgray", lw=0.7, label="Daily Input Score $x_t$")
    ax0.plot(
        input_score.index,
        input_score.rolling(30).mean(),
        color="navy",
        lw=1.6,
        label="30-Day Rolling Mean Input Score",
    )
    ax0.axhline(det_in.mu, color="seagreen", ls="--", lw=1.2, label=f"2010–2013 Reference Mean ($\\mu={det_in.mu:.2f}$)")
    ax0.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2013-12-31"), color="seagreen", alpha=0.08, label="Reference Period (2010–2013)")
    ax0.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31"), color="gold", alpha=0.15, label="Threshold Calibration (2014)")
    for i, d in enumerate(alarm_dates):
        ax0.axvline(d, color="crimson", ls=":", lw=1.3, label="Input Drift Alarm" if i == 0 else None)
    ax0.set_ylabel("Input Drift Score (Mean $z^2$)")
    ax0.set_title("(a) Deseasonalised Daily Input Drift Score Across Training, Calibration, and Stage 2 Simulation (2010–2026)")
    ax0.legend(loc="upper left", ncol=3, fontsize=8.5)
    ax0.grid(alpha=0.3)

    # Panel (b): Online CUSUM Statistic S_t on Stage 2 (2015-2026)
    ax1 = axes[1]
    ax1.plot(cusum_path.index, cusum_path, color="darkorange", lw=1.5, label="Online Input CUSUM Statistic $S_t$")
    ax1.axhline(det_in.h, color="crimson", ls="--", lw=1.5, label=f"Decision Threshold ($h={det_in.h:.2f}$)")
    for i, d in enumerate(alarm_dates):
        ax1.axvline(d, color="crimson", ls=":", lw=1.3)
        ax1.scatter([d], [det_in.h], color="crimson", zorder=5, s=35, label="Triggered Alarm (Reset $S_t \\to 0$)" if i == 0 else None)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CUSUM Statistic $S_t$ (std units)")
    ax1.set_title(f"(b) Online Input-Based CUSUM Drift Detector (2015–2026, Cooldown = {cooldown_days}d, Budget = {max_alarms} Alarms)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_input_drift_detector.png"), dpi=200)
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
"""Stage 2 Historical Simulation (January 2015 – October 2026) for Frozen Stage 1 Demand Models.

Implements the COMP9417 Stage 2 protocol and Appendix A Confidence Intervals:
  1. Refits all Stage 1 demand models (Per-Slot Lasso, Per-Slot Ridge, Per-Slot ElasticNet,
     and Flat Daily Multi-Output Ridge) plus preprocessing (Day-Type PCA+KMeans and
     StandardScalers) strictly on 2010–2014 data, then freezes all parameters.
  2. Simulates day-by-day deployment across January 2015 – October 2026 (4,322 days,
     142 months, 207,456 half-hours) without any updates, alongside both official baselines:
       - Same half-hour yesterday (lag = 1 day)
       - Same half-hour last week (lag = 7 days)
  3. Computes monthly Demand MAPE (%) and MAE (MW) and builds 95% confidence intervals
     for monthly MAPE in both ways specified in Appendix A:
       (a) Independent half-hours normal approximation: MAPE +/- 1.96 * s / sqrt(n)
       (b) Day-block bootstrap (resampling whole 48-half-hour days with replacement,
           B = 1,000 replicates, reporting 2.5th and 97.5th percentiles).
  4. Exports yearly and monthly tables and diagnostic plots to analysis_outputs/.
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = HERE
ROOT_DIR = os.path.dirname(HERE)
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features import (  # noqa: E402
    N_PERIODS,
    build_candidate_features,
    mae_metric,
    mape_metric,
)
from model import PerSlotLinearRegressor  # noqa: E402


def day_block_bootstrap(errors_by_day, stat=np.mean, n_boot: int = 1000, seed: int = 0):
    """Authoritative Appendix A day-block bootstrap implementation.

    errors_by_day: list of 1D arrays, one per day (each containing the 48 half-hour values a_i).
    """
    rng = np.random.default_rng(seed)
    k = len(errors_by_day)
    # Vectorised bootstrap draw of day indices (n_boot, k) for exact equivalence and speed
    day_mat = np.asarray(errors_by_day, dtype=np.float64)  # shape (k, 48)
    if day_mat.ndim == 2 and stat is np.mean:
        drawn_idx = rng.integers(0, k, size=(n_boot, k))
        # Mean over all 48*k half-hours equals mean of daily means when every day has 48 half-hours
        daily_means = day_mat.mean(axis=1)
        boot_stats = daily_means[drawn_idx].mean(axis=1)
        return np.percentile(boot_stats, [2.5, 97.5])
    stats = [
        stat(np.concatenate([errors_by_day[j] for j in rng.integers(0, k, k)]))
        for _ in range(n_boot)
    ]
    return np.percentile(stats, [2.5, 97.5])


def compute_monthly_ci_table(
    target_dates: pd.DatetimeIndex,
    Y_true: np.ndarray,
    preds_dict: dict,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute monthly MAPE, MAE, Independent 95% CI, and Day-Block Bootstrap 95% CI."""
    months = target_dates.to_period("M")
    unique_months = np.unique(months)

    rows = []
    for m_Name, Y_pred in preds_dict.items():
        # Per-half-hour percentage error a_i = 100 * |y_i - yhat_i| / y_i, shape (n_days, 48)
        ape_2d = 100.0 * np.abs(Y_true - Y_pred) / np.abs(Y_true)
        ae_2d = np.abs(Y_true - Y_pred)
        bias_2d = Y_pred - Y_true

        for m in unique_months:
            m_mask = months == m
            m_ape_days = ape_2d[m_mask]      # shape (k_days, 48)
            m_ae_days = ae_2d[m_mask]
            m_bias_days = bias_2d[m_mask]

            k_days = m_ape_days.shape[0]
            m_ape_flat = m_ape_days.reshape(-1)
            n_hh = len(m_ape_flat)

            mape_val = float(np.mean(m_ape_flat))
            mae_val = float(np.mean(m_ae_days))
            bias_val = float(np.mean(m_bias_days))

            # (a) Treating each half-hour as independent: MAPE +/- 1.96 * s / sqrt(n)
            s_hh = float(np.std(m_ape_flat, ddof=1)) if n_hh > 1 else 0.0
            ind_half_width = 1.96 * s_hh / np.sqrt(n_hh)
            ind_low = mape_val - ind_half_width
            ind_high = mape_val + ind_half_width
            ind_width = 2.0 * ind_half_width

            # (b) Day-block bootstrap (resampling whole days with replacement)
            errors_by_day = [m_ape_days[d_i] for d_i in range(k_days)]
            boot_low, boot_high = day_block_bootstrap(
                errors_by_day, stat=np.mean, n_boot=n_boot, seed=seed
            )
            boot_low, boot_high = float(boot_low), float(boot_high)
            boot_width = boot_high - boot_low

            # Also compute sample std of daily mean APEs to explain variance inflation
            daily_mean_ape = m_ape_days.mean(axis=1)
            s_day = float(np.std(daily_mean_ape, ddof=1)) if k_days > 1 else 0.0

            rows.append({
                "model": m_Name,
                "month": str(m),
                "month_dt": m.to_timestamp(),
                "year": int(m.year),
                "n_days": int(k_days),
                "n_half_hours": int(n_hh),
                "mape": mape_val,
                "mae": mae_val,
                "mean_signed_bias_mw": bias_val,
                "s_half_hour": s_hh,
                "s_daily_mean": s_day,
                "ind_ci_low": ind_low,
                "ind_ci_high": ind_high,
                "ind_ci_width": ind_width,
                "boot_ci_low": boot_low,
                "boot_ci_high": boot_high,
                "boot_ci_width": boot_width,
                "width_ratio_boot_over_ind": boot_width / max(ind_width, 1e-12),
            })

    return pd.DataFrame(rows)


def compute_error_dependence_stats(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Compute within-day and across-day autocorrelation of errors to support Appendix A analysis."""
    ape_2d = 100.0 * np.abs(Y_true - Y_pred) / np.abs(Y_true)
    ape_flat = ape_2d.reshape(-1)
    daily_ape = ape_2d.mean(axis=1)

    # Half-hourly autocorrelation at lags 1 (30m), 2 (1h), 4 (2h), 12 (6h), 24 (12h), 48 (24h)
    hh_series = pd.Series(ape_flat)
    hh_acf = {f"lag_{lag}_hh": float(hh_series.autocorr(lag=lag)) for lag in [1, 2, 4, 8, 12, 24, 48]}

    # Daily mean error autocorrelation at lags 1..7 days
    day_series = pd.Series(daily_ape)
    day_acf = {f"lag_{lag}_day": float(day_series.autocorr(lag=lag)) for lag in range(1, 8)}

    # Intraclass correlation coefficient (ICC) within days:
    # Var(daily_mean) / (Var(half_hour) / 48) = Design Effect (DEFF)
    var_hh = float(np.var(ape_flat, ddof=1))
    var_day_mean = float(np.var(daily_ape, ddof=1))
    design_effect = var_day_mean / (var_hh / N_PERIODS)
    icc_within_day = (design_effect - 1.0) / (N_PERIODS - 1.0)

    return {
        "half_hourly_acf": hh_acf,
        "daily_mean_acf": day_acf,
        "design_effect_deff": float(design_effect),
        "sqrt_design_effect": float(np.sqrt(design_effect)),
        "within_day_icc": float(icc_within_day),
        "effective_independent_hh_per_day": float(N_PERIODS / design_effect),
    }


def plot_stage2_monthly_ci_figures(
    monthly_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    target_dates_sim: pd.DatetimeIndex,
    Y_sim: np.ndarray,
    preds_sim: dict,
    dep_stats: dict,
    out_dir: str,
):
    """Generate publication-quality Stage 2 monthly error CI and degradation diagnostic plots."""
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 7: Monthly Error of Frozen Models with 95% Confidence Intervals
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 13.0), sharex=True)

    lasso_m = monthly_df[monthly_df["model"] == "Per-Slot Lasso (L1) [Selected]"].sort_values("month_dt")
    ridge_m = monthly_df[monthly_df["model"] == "Per-Slot Ridge (L2)"].sort_values("month_dt")
    enet_m = monthly_df[monthly_df["model"] == "Per-Slot ElasticNet (L1+L2)"].sort_values("month_dt")
    flat_m = monthly_df[monthly_df["model"] == "Flat Daily Multi-Output Ridge"].sort_values("month_dt")
    b1_m = monthly_df[monthly_df["model"] == "Baseline 1: Same HH Yesterday"].sort_values("month_dt")
    b7_m = monthly_df[monthly_df["model"] == "Baseline 2: Same HH Last Week"].sort_values("month_dt")

    # Panel (a): Selected Frozen Model (Per-Slot Lasso) with Both 95% Confidence Intervals
    ax0 = axes[0]
    ax0.plot(
        lasso_m["month_dt"], lasso_m["mape"],
        color="crimson", linewidth=1.8, label="Frozen Selected Model: Per-Slot Lasso (L1) Monthly MAPE"
    )
    ax0.fill_between(
        lasso_m["month_dt"], lasso_m["boot_ci_low"], lasso_m["boot_ci_high"],
        color="salmon", alpha=0.32, label="95% CI (b): Day-Block Bootstrap (1,000 replicates)"
    )
    ax0.fill_between(
        lasso_m["month_dt"], lasso_m["ind_ci_low"], lasso_m["ind_ci_high"],
        color="darkred", alpha=0.45, label="95% CI (a): Independent Half-Hours (Normal Approx.)"
    )
    ax0.plot(
        b1_m["month_dt"], b1_m["mape"],
        color="navy", linestyle="--", linewidth=1.3, alpha=0.8, label="Baseline 1: Same Half-Hour Yesterday"
    )
    ax0.set_ylabel("Monthly Demand MAPE (%)")
    ax0.set_title("(a) Monthly Demand MAPE of Frozen Selected Model (Per-Slot Lasso) with Both 95% Confidence Intervals (Jan 2015 – Oct 2026)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", ncol=2, fontsize=9.5)

    # Panel (b): Comparison of All Studied Stage 1 Frozen Models Over Historical Simulation
    ax1 = axes[1]
    ax1.plot(lasso_m["month_dt"], lasso_m["mape"], color="crimson", linewidth=1.7, label="Per-Slot Lasso (L1, k=65)")
    ax1.plot(ridge_m["month_dt"], ridge_m["mape"], color="royalblue", linewidth=1.5, linestyle="-.", label="Per-Slot Ridge (L2, k=65)")
    ax1.plot(enet_m["month_dt"], enet_m["mape"], color="seagreen", linewidth=1.4, linestyle=":", label="Per-Slot ElasticNet (L1+L2, k=65)")
    ax1.plot(flat_m["month_dt"], flat_m["mape"], color="darkorange", linewidth=1.4, alpha=0.85, label="Flat Daily Multi-Output Ridge (403 feats)")
    ax1.plot(b1_m["month_dt"], b1_m["mape"], color="black", linestyle="--", linewidth=1.3, alpha=0.75, label="Baseline 1: Same HH Yesterday")
    ax1.fill_between(
        ridge_m["month_dt"], ridge_m["boot_ci_low"], ridge_m["boot_ci_high"],
        color="royalblue", alpha=0.18, label="Per-Slot Ridge 95% Day-Block Bootstrap CI"
    )
    ax1.set_ylabel("Monthly Demand MAPE (%)")
    ax1.set_title("(b) Monthly Demand MAPE Comparison Across All Frozen Stage 1 Models (2015–2026)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=3, fontsize=9)

    # Panel (c): Direct Comparison of 95% Confidence Interval Widths (Bootstrap vs Independent)
    ax2 = axes[2]
    mean_ratio = lasso_m["width_ratio_boot_over_ind"].mean()
    ax2.plot(
        lasso_m["month_dt"], lasso_m["boot_ci_width"],
        color="crimson", linewidth=1.7,
        label=f"Day-Block Bootstrap 95% CI Width (Mean = {lasso_m['boot_ci_width'].mean():.2f}%)"
    )
    ax2.plot(
        lasso_m["month_dt"], lasso_m["ind_ci_width"],
        color="navy", linewidth=1.7, linestyle="--",
        label=f"Independent Half-Hours 95% CI Width (Mean = {lasso_m['ind_ci_width'].mean():.2f}%)"
    )
    ax2.fill_between(
        lasso_m["month_dt"], lasso_m["ind_ci_width"], lasso_m["boot_ci_width"],
        color="gold", alpha=0.25,
        label=f"Within-Day Correlation Inflation (Mean Width Ratio = {mean_ratio:.2f}x)"
    )
    ax2.set_xlabel("Simulation Date (Month, Jan 2015 – Oct 2026)")
    ax2.set_ylabel("95% CI Total Width (pct points)")
    ax2.set_title("(c) Comparison of 95% Confidence Interval Widths: Day-Block Bootstrap vs. Independent Half-Hours")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_stage2_frozen_monthly_mape_ci.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 8: Stage 2 Performance Degradation & Autocorrelation Diagnostics
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # Panel (a): Yearly MAPE Trajectory (2015-2026)
    ax_a = axes[0, 0]
    for m_name, color, marker, ls in [
        ("Per-Slot Lasso (L1) [Selected]", "crimson", "o", "-"),
        ("Per-Slot Ridge (L2)", "royalblue", "s", "-."),
        ("Per-Slot ElasticNet (L1+L2)", "seagreen", "^", ":"),
        ("Flat Daily Multi-Output Ridge", "darkorange", "d", "-"),
        ("Baseline 1: Same HH Yesterday", "black", "x", "--"),
        ("Baseline 2: Same HH Last Week", "gray", "+", ":"),
    ]:
        sub = yearly_df[yearly_df["model"] == m_name].sort_values("year")
        ax_a.plot(sub["year"], sub["mape"], color=color, marker=marker, linestyle=ls, linewidth=2.0, label=m_name)
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Annual Demand MAPE (%)")
    ax_a.set_title("(a) Annual Demand MAPE Degradation of Frozen Models (2015–2026)")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="upper left", fontsize=8.5)

    # Panel (b): Diurnal Error Profile of Frozen Lasso Model by Multi-Year Era
    ax_b = axes[0, 1]
    hours = np.arange(N_PERIODS) / 2.0
    ape_lasso = 100.0 * np.abs(Y_sim - preds_sim["Per-Slot Lasso (L1) [Selected]"]) / np.abs(Y_sim)
    sim_years = target_dates_sim.year
    for (y_start, y_end), lbl, col in [
        ((2015, 2017), "2015–2017 (Early Sim)", "seagreen"),
        ((2018, 2020), "2018–2020 (Solar Growth)", "royalblue"),
        ((2021, 2023), "2021–2023 (5-Min & Coal Exit)", "darkorange"),
        ((2024, 2026), "2024–2026 (Deep Duck Curve)", "crimson"),
    ]:
        era_mask = (sim_years >= y_start) & (sim_years <= y_end)
        ax_b.plot(hours, ape_lasso[era_mask].mean(axis=0), linewidth=2.2, color=col, label=lbl)
    ax_b.set_xlabel("Time of Day (Hours, AEST)")
    ax_b.set_ylabel("Mean Half-Hourly MAPE (%)")
    ax_b.set_title("(b) Diurnal Localization of Frozen Model Degradation (Midday Solar Drift)")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=9)

    # Panel (c): Monthly Mean Signed Forecast Bias (Predicted minus Observed MW)
    ax_c = axes[1, 0]
    ax_c.plot(
        lasso_m["month_dt"], lasso_m["mean_signed_bias_mw"],
        color="crimson", linewidth=1.6, label="Frozen Per-Slot Lasso Signed Bias (Pred - True)"
    )
    ax_c.plot(
        b1_m["month_dt"], b1_m["mean_signed_bias_mw"],
        color="navy", linestyle="--", linewidth=1.2, alpha=0.7, label="Yesterday Baseline Signed Bias"
    )
    ax_c.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax_c.set_xlabel("Simulation Date")
    ax_c.set_ylabel("Mean Signed Bias (MW)")
    ax_c.set_title("(c) Emergence of Systematic Over-Prediction Bias in Frozen Model")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper left", fontsize=9)

    # Panel (d): Within-Day vs Across-Day Error Autocorrelation (Empirical Proof for Appendix A)
    ax_d = axes[1, 1]
    ape_flat = ape_lasso.reshape(-1)
    daily_ape = ape_lasso.mean(axis=1)
    hh_lags = np.arange(1, 25)
    hh_series = pd.Series(ape_flat)
    hh_acfs = [hh_series.autocorr(lag=int(l)) for l in hh_lags]
    day_lags = np.arange(1, 15)
    day_series = pd.Series(daily_ape)
    day_acfs = [day_series.autocorr(lag=int(l)) for l in day_lags]

    ax_d.plot(hh_lags, hh_acfs, marker="o", color="crimson", linewidth=2.0, label="Within-Day Half-Hourly Error ACF (Lag in Half-Hours)")
    ax_d.plot(day_lags, day_acfs, marker="s", color="navy", linewidth=2.0, linestyle="--", label="Across-Day Daily Mean Error ACF (Lag in Days)")
    ax_d.axhline(0.0, color="black", linewidth=0.8)
    ax_d.set_xlabel("Lag k (Half-Hours for solid red; Whole Days for dashed blue)")
    ax_d.set_ylabel("Autocorrelation Coefficient r(k)")
    ax_d.set_title(
        f"(d) Error Autocorrelation: Within-Day (ICC={dep_stats['within_day_icc']:.2f}, DEFF={dep_stats['design_effect_deff']:.1f}) vs. Across-Day"
    )
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper right", fontsize=8.8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_stage2_degradation_diagnostics.png"), dpi=200)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()

    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT_DIR, "data", "nem_weather_halfhourly_2010_2026.csv")
    out_dir = os.path.join(PKG_DIR, "analysis_outputs")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(PKG_DIR, "artifacts", "config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    selected_features = config["selected_features"]

    print("[1/4] Loading full dataset (2010–2026) and fitting Day-Type transformer on 2010–2014 only...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()

    bundle = build_candidate_features(
        df,
        day_type_transformer=None,
        fit_day_type_max_year=2014,
        include_flat_daily=True,
    )
    X_3d = bundle["X_3d"]
    Y_2d = bundle["Y_2d"]
    X_flat = bundle["X_flat"]
    target_dates = bundle["target_dates"]
    feature_names = bundle["feature_names"]
    selected_indices = [feature_names.index(f) for f in selected_features]

    refit_mask = target_dates.year <= 2014
    sim_mask = target_dates.year >= 2015

    X_tr, Y_tr = X_3d[refit_mask], Y_2d[refit_mask]
    X_sim, Y_sim = X_3d[sim_mask], Y_2d[sim_mask]
    dates_sim = target_dates[sim_mask]

    print(
        f"      Stage 1 Refit Period (2010–2014): {int(refit_mask.sum())} days | "
        f"Stage 2 Simulation Period (Jan 2015 – Oct 2026): {int(sim_mask.sum())} days"
    )

    print("[2/4] Fitting all Stage 1 models on 2010–2014 and freezing parameters for 2015–2026 simulation...")
    # 1. Selected Model: Per-Slot Lasso (L1, alpha=1.0)
    lasso_model = PerSlotLinearRegressor(
        model_type="lasso", alpha=1.0, selected_indices=selected_indices, selected_feature_names=selected_features
    ).fit(X_tr, Y_tr)

    # 2. Per-Slot Ridge (L2, alpha=5.0)
    ridge_model = PerSlotLinearRegressor(
        model_type="ridge", alpha=5.0, selected_indices=selected_indices, selected_feature_names=selected_features
    ).fit(X_tr, Y_tr)

    # 3. Per-Slot ElasticNet (L1+L2, alpha=0.005, l1_ratio=0.5)
    enet_model = PerSlotLinearRegressor(
        model_type="elasticnet", alpha=0.005, l1_ratio=0.5,
        selected_indices=selected_indices, selected_feature_names=selected_features
    ).fit(X_tr, Y_tr)

    # 4. Flat Daily Multi-Output Ridge (L2, alpha=1.0)
    sc_flat = StandardScaler()
    X_flat_tr = sc_flat.fit_transform(X_flat[refit_mask])
    X_flat_sim = sc_flat.transform(X_flat[sim_mask])
    flat_model = Ridge(alpha=1.0).fit(X_flat_tr, Y_tr)

    # Generate frozen predictions over Jan 2015 - Oct 2026
    dem_d_idx = feature_names.index("dem_d_t")
    dem_d6_idx = feature_names.index("dem_d6_t")

    preds_sim = {
        "Per-Slot Lasso (L1) [Selected]": lasso_model.predict_tensor(X_sim),
        "Per-Slot Ridge (L2)": ridge_model.predict_tensor(X_sim),
        "Per-Slot ElasticNet (L1+L2)": enet_model.predict_tensor(X_sim),
        "Flat Daily Multi-Output Ridge": np.clip(flat_model.predict(X_flat_sim), 0.0, 20000.0),
        "Baseline 1: Same HH Yesterday": X_sim[:, :, dem_d_idx],
        "Baseline 2: Same HH Last Week": X_sim[:, :, dem_d6_idx],
    }

    # Compute yearly table
    yearly_rows = []
    years_sim = dates_sim.year
    for m_name, y_hat in preds_sim.items():
        for yr in sorted(np.unique(years_sim)):
            ym = years_sim == yr
            yearly_rows.append({
                "model": m_name,
                "year": int(yr),
                "mape": mape_metric(Y_sim[ym], y_hat[ym]),
                "mae": mae_metric(Y_sim[ym], y_hat[ym]),
                "mean_signed_bias_mw": float(np.mean(y_hat[ym] - Y_sim[ym])),
            })
        yearly_rows.append({
            "model": m_name,
            "year": "Overall (2015-2026)",
            "mape": mape_metric(Y_sim, y_hat),
            "mae": mae_metric(Y_sim, y_hat),
            "mean_signed_bias_mw": float(np.mean(y_hat - Y_sim)),
        })
    yearly_df_all = pd.DataFrame(yearly_rows)
    yearly_df_all.to_csv(os.path.join(out_dir, "table8_stage2_frozen_yearly_metrics.csv"), index=False)
    yearly_numeric_df = yearly_df_all[yearly_df_all["year"] != "Overall (2015-2026)"].copy()
    yearly_numeric_df["year"] = yearly_numeric_df["year"].astype(int)

    print("\n=== Stage 2 Yearly Demand MAPE (%) Across Frozen Models (2015–2026) ===")
    pivot_mape = yearly_df_all.pivot(index="year", columns="model", values="mape")
    cols_order = list(preds_sim.keys())
    print(pivot_mape[cols_order].round(3).to_string())

    print("\n[3/4] Computing monthly MAPE and both 95% Confidence Intervals (Independent vs Day-Block Bootstrap)...")
    monthly_df = compute_monthly_ci_table(dates_sim, Y_sim, preds_sim, n_boot=1000, seed=0)
    monthly_df.to_csv(os.path.join(out_dir, "table9_stage2_frozen_monthly_ci.csv"), index=False)

    dep_stats = compute_error_dependence_stats(Y_sim, preds_sim["Per-Slot Lasso (L1) [Selected]"])
    with open(os.path.join(out_dir, "stage2_ci_dependence_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(dep_stats, fh, indent=2)

    ci_summary = monthly_df.groupby("model")[
        ["mape", "mae", "ind_ci_width", "boot_ci_width", "width_ratio_boot_over_ind"]
    ].mean().loc[cols_order]
    print("\n=== Stage 2 Monthly 95% Confidence Interval Summary (Mean over 142 Months, Jan 2015 – Oct 2026) ===")
    print(ci_summary.round(3).to_string())
    print(
        f"\nWithin-day error ICC = {dep_stats['within_day_icc']:.3f} | "
        f"Design Effect (DEFF) = {dep_stats['design_effect_deff']:.2f} "
        f"(sqrt(DEFF) = {dep_stats['sqrt_design_effect']:.2f}) | "
        f"Effective independent half-hours per 48-hh day = {dep_stats['effective_independent_hh_per_day']:.2f}"
    )
    print("Half-hourly error ACF:", {k: round(v, 3) for k, v in dep_stats["half_hourly_acf"].items()})
    print("Daily mean error ACF:", {k: round(v, 3) for k, v in dep_stats["daily_mean_acf"].items()})

    print("\n[4/4] Generating Stage 2 monthly CI and degradation figures...")
    plot_stage2_monthly_ci_figures(
        monthly_df=monthly_df,
        yearly_df=yearly_numeric_df,
        target_dates_sim=dates_sim,
        Y_sim=Y_sim,
        preds_sim=preds_sim,
        dep_stats=dep_stats,
        out_dir=out_dir,
    )
    print(f"Completed Stage 2 frozen simulation in {time.time() - t0:.1f}s. Saved outputs to {out_dir}/.")


if __name__ == "__main__":
    main()

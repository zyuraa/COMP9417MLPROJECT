"""Stage 2 Model Adaptation & Retraining Budget Comparison (Jan 2015 – Oct 2026).

Implements Step 2 of the COMP9417 requirements for NSW Day-Ahead Demand Forecasting:
  - Evaluates adaptation strategies on the selected Stage 1 model (Per-Slot Lasso,
    alpha=1.0, k*=65 selected features) under the strict budget of <= 6 retrains
    across Jan 2015 – Oct 2026 (4,322 days / 142 months), using only data available
    up to the end of day D (taking effect from forecast day D+1):
      0. Frozen Stage 1 Model (2010–2014, 0 retrains) + Official Baselines
      1. Periodic Expanding Window (6 evenly spaced retrains, uniform weights)
      2. Periodic Sliding Window (6 retrains, trailing 2Y / 3Y window)
      3. Periodic Exponential Recency Weighting (6 retrains, half-life H = 548 days)
      4. Online Input-Drift-Triggered Retraining (6 alarms from Step 1 Input-EWMA-Reset)
      5. Online Error-Drift-Triggered Retraining (closed-loop OnlineErrorDriftDetector,
         30-day EWMA + 365-day cooldown, <= 6 retrains) under Expanding, Sliding-3Y,
         and Recency weighting.
  - Evaluates the Retraining Budget Trade-Off Curve (B = 0, 1, ..., 6) to answer:
    "Does detecting distribution shift help decide when to retrain?"
  - Evaluates Sliding Window Length & Exponential Half-Life Sensitivity on 2015–2024
    (Selection Period) vs. 2025–2026 (Late Holdout Check) to justify Stage 3.
  - Computes monthly 95% CIs (Independent Normal & Appendix A Day-Block Bootstrap) and
    exports tables (table14–table18), JSON summary, and figures (fig11, fig12).
"""
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features import (  # noqa: E402
    DayTypeTransformer,
    N_PERIODS,
    WINDOW_DAYS,
    build_candidate_features,
    mae_metric,
    mape_metric,
    market_day,
)
from model import PerSlotLinearRegressor  # noqa: E402
from stage2_drift_detection import (  # noqa: E402
    COOLDOWN_DAYS,
    EWMA_SPAN,
    MAX_RETRAIN_BUDGET,
    OnlineErrorDriftDetector,
    compute_causal_ewma,
)
from stage2_frozen_simulation import compute_monthly_ci_table  # noqa: E402


DEFAULT_SLIDING_3Y_DAYS = 1095
DEFAULT_SLIDING_2Y_DAYS = 730
DEFAULT_HALFLIFE_DAYS = 548  # 1.5 years


def extract_daily_demand_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Extract raw (n_days, 48) NSW demand matrix and valid sample indices matching features.py."""
    days = market_day(df.index)
    counts = pd.Series(1, index=df.index).groupby(days).sum()
    valid_days = counts[counts == N_PERIODS].index

    if len(valid_days) == len(df) // N_PERIODS:
        df_clean = df
        unique_days = pd.DatetimeIndex(valid_days)
    else:
        mask = days.isin(valid_days)
        df_clean = df.loc[mask]
        unique_days = pd.DatetimeIndex(sorted(valid_days))

    n_days = len(unique_days)
    nsw_dem_all = df_clean["nsw_demand"].to_numpy(dtype=np.float64).reshape(n_days, N_PERIODS)

    all_idx = np.arange(WINDOW_DAYS, n_days)
    day_span = (unique_days[all_idx] - unique_days[all_idx - WINDOW_DAYS]).days == WINDOW_DAYS
    idx_valid = all_idx[day_span]
    return nsw_dem_all, idx_valid, unique_days


def build_daytype_column_map(selected_features: List[str]) -> List[Tuple[int, str, Optional[int]]]:
    """Identify which columns in the selected feature tensor come from DayTypeTransformer."""
    col_specs: List[Tuple[int, str, Optional[int]]] = []
    for col_i, fname in enumerate(selected_features):
        if fname.startswith("day_d_cluster_") and fname != "day_d_cluster_dist":
            c_idx = int(fname.rsplit("_", 1)[-1])
            col_specs.append((col_i, "d_cluster", c_idx))
        elif fname.startswith("day_d6_cluster_"):
            c_idx = int(fname.rsplit("_", 1)[-1])
            col_specs.append((col_i, "d6_cluster", c_idx))
        elif fname == "day_d_cluster_dist":
            col_specs.append((col_i, "d_dist", None))
    return col_specs


def update_selected_daytype_features(
    X_sub_buf: np.ndarray,
    nsw_dem_all: np.ndarray,
    idx_valid: np.ndarray,
    daytype_col_specs: List[Tuple[int, str, Optional[int]]],
    dt_transformer: DayTypeTransformer,
) -> None:
    """In-place update of DayType features inside a (n_samples, 48, k*) feature buffer."""
    clusters_all, dists_all = dt_transformer.transform(nsw_dem_all)
    c_d = clusters_all[idx_valid - 1]
    c_d6 = clusters_all[idx_valid - 7]
    dist_d = dists_all[idx_valid - 1]

    for col_i, kind, c_idx in daytype_col_specs:
        if kind == "d_cluster":
            vals = (c_d == c_idx).astype(np.float64)[:, None]
        elif kind == "d6_cluster":
            vals = (c_d6 == c_idx).astype(np.float64)[:, None]
        else:
            vals = dist_d[:, None]
        X_sub_buf[:, :, col_i] = vals


def fit_adapted_lasso_at_day(
    retrain_sample_idx: int,
    target_dates: pd.DatetimeIndex,
    X_sub_buf: np.ndarray,
    Y_2d: np.ndarray,
    nsw_dem_all: np.ndarray,
    idx_valid: np.ndarray,
    daytype_col_specs: List[Tuple[int, str, Optional[int]]],
    selected_features: List[str],
    weighting_mode: str,
    window_days: int = DEFAULT_SLIDING_3Y_DAYS,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    alpha: float = 1.0,
) -> PerSlotLinearRegressor:
    """Fit DayTypeTransformer and Per-Slot Lasso strictly on past data <= Day D_k."""
    raw_day_idx = int(idx_valid[retrain_sample_idx])
    d_retrain = target_dates[retrain_sample_idx]

    if weighting_mode == "expanding":
        dt_start = 0
        tr_start = 0
        sample_weight = None
    elif weighting_mode == "sliding":
        dt_start = max(0, raw_day_idx - window_days + 1)
        tr_start = max(0, retrain_sample_idx - window_days + 1)
        sample_weight = None
    elif weighting_mode == "recency":
        # Fit DayType clustering on recent 3 years so clusters track modern shapes
        dt_start = max(0, raw_day_idx - DEFAULT_SLIDING_3Y_DAYS + 1)
        tr_start = 0
        ages_days = (d_retrain - target_dates[: retrain_sample_idx + 1]).days.to_numpy(dtype=np.float64)
        raw_w = np.power(2.0, -ages_days / float(halflife_days))
        sample_weight = raw_w / np.mean(raw_w)
    else:
        raise ValueError(f"Unsupported weighting_mode: {weighting_mode}")

    dt = DayTypeTransformer(n_components=5, n_clusters=4, random_state=42)
    dt.fit(nsw_dem_all[dt_start : raw_day_idx + 1])
    update_selected_daytype_features(X_sub_buf, nsw_dem_all, idx_valid, daytype_col_specs, dt)

    model = PerSlotLinearRegressor(
        model_type="lasso",
        alpha=alpha,
        selected_indices=None,
        selected_feature_names=selected_features,
        day_type_transformer=dt,
    )
    model.fit(
        X_sub_buf[tr_start : retrain_sample_idx + 1],
        Y_2d[tr_start : retrain_sample_idx + 1],
        sample_weight=sample_weight,
    )
    return model


def get_periodic_retrain_indices(sim_indices: np.ndarray, budget: int) -> np.ndarray:
    """Return `budget` evenly spaced retrain sample indices inside `sim_indices`."""
    if budget <= 0:
        return np.array([], dtype=int)
    n_sim = len(sim_indices)
    offsets = [int(np.floor(m * n_sim / (budget + 1))) - 1 for m in range(1, budget + 1)]
    return sim_indices[np.asarray(offsets, dtype=int)]


def run_fixed_schedule_strategy(
    retrain_sample_indices: np.ndarray,
    initial_model: PerSlotLinearRegressor,
    X_sub_initial: np.ndarray,
    Y_2d: np.ndarray,
    target_dates: pd.DatetimeIndex,
    sim_indices: np.ndarray,
    nsw_dem_all: np.ndarray,
    idx_valid: np.ndarray,
    daytype_col_specs: List[Tuple[int, str, Optional[int]]],
    selected_features: List[str],
    weighting_mode: str,
    window_days: int = DEFAULT_SLIDING_3Y_DAYS,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    capture_snapshots_years: Optional[List[int]] = None,
) -> Dict[str, object]:
    """Simulate a fixed retrain schedule (Periodic or fixed Input-Drift alarms) across 2015–2026."""
    n_sim = len(sim_indices)
    sim_start = int(sim_indices[0])
    sim_end = int(sim_indices[-1]) + 1

    preds_sim = np.zeros((n_sim, N_PERIODS), dtype=np.float64)
    X_sub_buf = X_sub_initial.copy()
    curr_model = initial_model

    valid_retrains = sorted(
        int(r_idx) for r_idx in retrain_sample_indices if sim_start <= int(r_idx) < sim_end - 1
    )
    retrain_dates = [str(target_dates[r_idx].date()) for r_idx in valid_retrains]
    effective_dates = [str(target_dates[r_idx + 1].date()) for r_idx in valid_retrains]

    snapshots: Dict[int, np.ndarray] = {}
    if capture_snapshots_years and 2014 in capture_snapshots_years:
        snapshots[2014] = initial_model.get_coef_matrix()

    seg_start = sim_start
    for r_idx in valid_retrains:
        # Predict segment up to and including day D_k = target_dates[r_idx] using current model
        seg_end = r_idx + 1
        if seg_end > seg_start:
            preds_sim[seg_start - sim_start : seg_end - sim_start] = curr_model.predict_tensor(
                X_sub_buf[seg_start:seg_end]
            )
        # Retrain at end of day D_k -> takes effect from D_k + 1 (seg_end)
        curr_model = fit_adapted_lasso_at_day(
            retrain_sample_idx=r_idx,
            target_dates=target_dates,
            X_sub_buf=X_sub_buf,
            Y_2d=Y_2d,
            nsw_dem_all=nsw_dem_all,
            idx_valid=idx_valid,
            daytype_col_specs=daytype_col_specs,
            selected_features=selected_features,
            weighting_mode=weighting_mode,
            window_days=window_days,
            halflife_days=halflife_days,
        )
        if capture_snapshots_years:
            yr_r = int(target_dates[r_idx].year)
            for tgt_yr in capture_snapshots_years:
                if yr_r == tgt_yr or (tgt_yr not in snapshots and yr_r >= tgt_yr):
                    snapshots[tgt_yr] = curr_model.get_coef_matrix()
        seg_start = seg_end

    if sim_end > seg_start:
        preds_sim[seg_start - sim_start : sim_end - sim_start] = curr_model.predict_tensor(
            X_sub_buf[seg_start:sim_end]
        )

    return {
        "preds_sim": preds_sim,
        "n_retrains": len(valid_retrains),
        "retrain_dates": retrain_dates,
        "effective_dates": effective_dates,
        "coef_snapshots": snapshots,
    }


def run_closed_loop_error_strategy(
    initial_model: PerSlotLinearRegressor,
    X_sub_initial: np.ndarray,
    Y_2d: np.ndarray,
    target_dates: pd.DatetimeIndex,
    sim_indices: np.ndarray,
    nsw_dem_all: np.ndarray,
    idx_valid: np.ndarray,
    daytype_col_specs: List[Tuple[int, str, Optional[int]]],
    selected_features: List[str],
    err_detector: OnlineErrorDriftDetector,
    weighting_mode: str,
    window_days: int = DEFAULT_SLIDING_3Y_DAYS,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    max_retrains: int = MAX_RETRAIN_BUDGET,
    cooldown_days: int = COOLDOWN_DAYS,
) -> Dict[str, object]:
    """Closed-loop online Error-EWMA drift-triggered retraining across 2015–2026."""
    n_sim = len(sim_indices)
    preds_sim = np.zeros((n_sim, N_PERIODS), dtype=np.float64)
    ewma_sim = np.zeros(n_sim, dtype=np.float64)
    thresh_sim = np.zeros(n_sim, dtype=np.float64)

    X_sub_buf = X_sub_initial.copy()
    curr_model = initial_model
    curr_thresh = err_detector.ewma_threshold_
    alpha_ewma = 2.0 / (err_detector.ewma_span + 1.0)
    z_curr = float(err_detector.ewma_ref_mean_)

    retrain_dates: List[str] = []
    effective_dates: List[str] = []
    last_retrain_offset = -10**9
    post_retrain_z_buffer: List[float] = []
    reanchored = True

    for offset, idx in enumerate(sim_indices):
        y_day = Y_2d[idx]
        y_hat = curr_model.predict_single_day(X_sub_buf[idx])
        preds_sim[offset] = y_hat

        e_day = float(np.mean(np.abs(y_day - y_hat) / np.abs(y_day)) * 100.0)
        z_curr = alpha_ewma * e_day + (1.0 - alpha_ewma) * z_curr
        ewma_sim[offset] = z_curr
        thresh_sim[offset] = curr_thresh

        if not reanchored:
            post_retrain_z_buffer.append(z_curr)
            if len(post_retrain_z_buffer) >= cooldown_days:
                z_eff = np.asarray(post_retrain_z_buffer[30:], dtype=np.float64)
                mu_new = float(np.mean(z_eff))
                sd_new = float(np.std(z_eff, ddof=1))
                mx_new = float(np.max(z_eff))
                curr_thresh = float(
                    max(mu_new + err_detector.sigma_mult * sd_new, err_detector.max_safety * mx_new)
                )
                reanchored = True

        if len(retrain_dates) >= max_retrains or offset >= n_sim - 1:
            continue
        if (offset - last_retrain_offset) < cooldown_days:
            continue
        if z_curr > curr_thresh:
            retrain_dates.append(str(target_dates[idx].date()))
            effective_dates.append(str(target_dates[idx + 1].date()))
            last_retrain_offset = offset
            reanchored = False
            post_retrain_z_buffer = []

            curr_model = fit_adapted_lasso_at_day(
                retrain_sample_idx=int(idx),
                target_dates=target_dates,
                X_sub_buf=X_sub_buf,
                Y_2d=Y_2d,
                nsw_dem_all=nsw_dem_all,
                idx_valid=idx_valid,
                daytype_col_specs=daytype_col_specs,
                selected_features=selected_features,
                weighting_mode=weighting_mode,
                window_days=window_days,
                halflife_days=halflife_days,
            )

    return {
        "preds_sim": preds_sim,
        "ewma_sim": ewma_sim,
        "thresh_sim": thresh_sim,
        "n_retrains": len(retrain_dates),
        "retrain_dates": retrain_dates,
        "effective_dates": effective_dates,
    }


def summarize_strategy_metrics(
    name: str,
    family: str,
    trigger_mode: str,
    weighting_mode: str,
    res: dict,
    Y_sim: np.ndarray,
    dates_sim: pd.DatetimeIndex,
    frozen_mape: float,
    b1_mape: float,
) -> dict:
    """Compute overall, 2015–2024, 2025–2026, midday, and signed-bias metrics for one strategy."""
    preds = res["preds_sim"]
    m_sel = dates_sim.year <= 2024
    m_hold = dates_sim.year >= 2025

    overall_mape = mape_metric(Y_sim, preds)
    overall_mae = mae_metric(Y_sim, preds)
    sel_mape = mape_metric(Y_sim[m_sel], preds[m_sel])
    sel_mae = mae_metric(Y_sim[m_sel], preds[m_sel])
    hold_mape = mape_metric(Y_sim[m_hold], preds[m_hold])
    hold_mae = mae_metric(Y_sim[m_hold], preds[m_hold])
    midday_mape = mape_metric(Y_sim[:, 18:32], preds[:, 18:32])
    bias_mw = float(np.mean(preds - Y_sim))

    return {
        "strategy": name,
        "family": family,
        "trigger_mode": trigger_mode,
        "weighting_mode": weighting_mode,
        "n_retrains": int(res["n_retrains"]),
        "retrain_dates": "; ".join(res["retrain_dates"]) if res["retrain_dates"] else "None",
        "mape_2015_2024": sel_mape,
        "mae_2015_2024": sel_mae,
        "mape_2025_2026": hold_mape,
        "mae_2025_2026": hold_mae,
        "overall_mape_2015_2026": overall_mape,
        "overall_mae_2015_2026": overall_mae,
        "midday_09_16_mape": midday_mape,
        "mean_signed_bias_mw": bias_mw,
        "improv_vs_frozen_pct": 100.0 * (1.0 - overall_mape / frozen_mape),
        "improv_vs_yesterday_pct": 100.0 * (1.0 - overall_mape / b1_mape),
    }


def plot_adaptation_figures(
    yearly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    tradeoff_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    Y_sim: np.ndarray,
    dates_sim: pd.DatetimeIndex,
    preds_map: Dict[str, np.ndarray],
    retrain_dates_map: Dict[str, List[str]],
    coef_snapshots: Dict[int, np.ndarray],
    selected_features: List[str],
    out_dir: str,
) -> None:
    """Generate publication-quality Figures 11 and 12 for Stage 2 Model Adaptation."""
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 11: Yearly & Monthly Performance + Retrain Timing Across Strategies
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 13.5))

    # Panel (a): Annual Demand MAPE (2015–2026)
    ax0 = axes[0]
    plot_styles = [
        ("Frozen Stage 1 (0 Retrains)", "crimson", "o", "-", 2.1),
        ("Baseline 1: Same HH Yesterday", "black", "x", "--", 1.5),
        ("Periodic-Expanding (6)", "darkorange", "^", "-.", 1.9),
        ("Periodic-Sliding-3Y (6)", "seagreen", "s", "-.", 1.9),
        ("Periodic-Recency H=548d (6)", "teal", "d", "-", 2.1),
        ("Drift-Input-Recency H=548d (6)", "purple", "v", ":", 1.8),
        ("Drift-Error-Expanding (Closed-Loop)", "royalblue", "P", "--", 1.8),
        ("Drift-Error-Recency H=548d (Closed-Loop)", "navy", "*", "-", 2.4),
    ]
    yr_num = yearly_df[yearly_df["year"] != "Overall (2015-2026)"].copy()
    yr_num["year"] = yr_num["year"].astype(int)
    for m_name, col, mk, ls, lw in plot_styles:
        sub = yr_num[yr_num["strategy"] == m_name].sort_values("year")
        if len(sub) > 0:
            ax0.plot(sub["year"], sub["mape"], color=col, marker=mk, linestyle=ls, linewidth=lw, label=m_name)
    ax0.set_xlabel("Year")
    ax0.set_ylabel("Annual Demand MAPE (%)")
    ax0.set_title("(a) Annual Demand MAPE (2015–2026): Frozen vs. Periodic vs. Drift-Triggered Adaptation (Budget <= 6 Retrains)")
    ax0.set_xticks(range(2015, 2027))
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", ncol=2, fontsize=8.5)

    # Panel (b): Monthly Demand MAPE with 95% Day-Block Bootstrap Confidence Intervals
    ax1 = axes[1]
    m_frz = monthly_df[monthly_df["model"] == "Frozen Stage 1 (0 Retrains)"].sort_values("month_dt")
    m_per = monthly_df[monthly_df["model"] == "Periodic-Recency H=548d (6)"].sort_values("month_dt")
    m_err = monthly_df[monthly_df["model"] == "Drift-Error-Recency H=548d (Closed-Loop)"].sort_values("month_dt")

    ax1.plot(m_frz["month_dt"], m_frz["mape"], color="crimson", linewidth=1.6, alpha=0.85, label="Frozen Stage 1 (0 Retrains)")
    ax1.fill_between(
        m_frz["month_dt"], m_frz["boot_ci_low"], m_frz["boot_ci_high"],
        color="salmon", alpha=0.22, label="Frozen 95% Day-Block Bootstrap CI",
    )
    ax1.plot(m_per["month_dt"], m_per["mape"], color="teal", linewidth=1.6, linestyle="--", label="Periodic-Recency H=548d (6 Retrains)")
    ax1.plot(m_err["month_dt"], m_err["mape"], color="navy", linewidth=1.9, label="Drift-Error-Recency H=548d (Closed-Loop)")
    ax1.fill_between(
        m_err["month_dt"], m_err["boot_ci_low"], m_err["boot_ci_high"],
        color="cornflowerblue", alpha=0.30, label="Drift-Error-Recency 95% Day-Block Bootstrap CI",
    )
    for k_i, d_str in enumerate(retrain_dates_map["Drift-Error-Recency H=548d (Closed-Loop)"]):
        dt_r = pd.Timestamp(d_str)
        ax1.axvline(dt_r, color="navy", linestyle=":", alpha=0.6, linewidth=1.1)
        ax1.text(
            dt_r, 15.8 if k_i % 2 == 0 else 13.8, f"R{k_i+1}: {d_str[:7]}",
            rotation=0, fontsize=7.8, fontweight="bold", color="navy",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8),
        )
    ax1.set_ylabel("Monthly Demand MAPE (%)")
    ax1.set_title("(b) Monthly Demand MAPE with 95% Day-Block Bootstrap CIs (Jan 2015 – Oct 2026)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel (c): Rolling 90-Day Daily MAPE & Retrain Timing Comparison
    ax2 = axes[2]
    timing_specs = [
        ("Periodic-Recency H=548d (6)", "teal", "--", "Periodic-Recency (6)", "P"),
        ("Drift-Input-Recency H=548d (6)", "purple", ":", "Drift-Input-Recency (6)", "I"),
        ("Drift-Error-Recency H=548d (Closed-Loop)", "navy", "-", "Drift-Error-Recency (Closed-Loop)", "E"),
    ]
    frz_daily = 100.0 * np.mean(np.abs(Y_sim - preds_map["Frozen Stage 1 (0 Retrains)"]) / np.abs(Y_sim), axis=1)
    frz_roll90 = pd.Series(frz_daily, index=dates_sim).rolling(90, min_periods=15).mean()
    ax2.plot(dates_sim, frz_roll90, color="crimson", alpha=0.65, linewidth=1.4, label="Frozen (90d Rolling MAPE)")

    for key, col, ls, lbl, tag in timing_specs:
        daily_e = 100.0 * np.mean(np.abs(Y_sim - preds_map[key]) / np.abs(Y_sim), axis=1)
        roll90 = pd.Series(daily_e, index=dates_sim).rolling(90, min_periods=15).mean()
        ax2.plot(dates_sim, roll90, color=col, linestyle=ls, linewidth=1.8, label=f"{lbl} (n={len(retrain_dates_map[key])} retrains)")
        for d_str in retrain_dates_map[key]:
            dt_r = pd.Timestamp(d_str)
            idx_pos = dates_sim.get_loc(dt_r)
            ax2.scatter([dt_r], [roll90.iloc[idx_pos]], color=col, edgecolor="black", s=48, zorder=6)

    ax2.set_xlabel("Simulation Date (Jan 2015 – Oct 2026)")
    ax2.set_ylabel("90-Day Rolling Demand MAPE (%)")
    ax2.set_title("(c) Retrain Timing & Post-Retrain Error Suppression (90-Day Rolling Daily MAPE)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig11_stage2_adaptation_yearly_and_monthly_comparison.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 12: Trade-Off Curve, Diurnal Error Recovery, Sensitivity, Coef Shift
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # Panel (a): Retraining Budget Trade-Off Curve (B = 0..6)
    ax_a = axes[0, 0]
    curve_specs = [
        ("Periodic-Expanding", "evenly_spaced", "darkorange", "o", "-."),
        ("Periodic-Sliding-3Y", "evenly_spaced", "seagreen", "s", "-."),
        ("Periodic-Recency (H=548d)", "evenly_spaced", "teal", "d", "-"),
        ("Drift-Input-Recency (H=548d)", "chronological_prefix_cap", "purple", "v", ":"),
        ("Drift-Error-Expanding", "cooldown_scaled", "royalblue", "P", "--"),
        ("Drift-Error-Recency (H=548d)", "cooldown_scaled", "navy", "*", "-"),
    ]
    for cname, alloc, col, mk, ls in curve_specs:
        sub = tradeoff_df[
            (tradeoff_df["strategy_family"] == cname) & (tradeoff_df["allocation_mode"] == alloc)
        ].sort_values("n_retrains_used")
        if len(sub) > 0:
            ax_a.plot(
                sub["n_retrains_used"], sub["overall_mape_2015_2026"],
                color=col, marker=mk, linestyle=ls, linewidth=2.0, markersize=7,
                label=f"{cname} ({alloc.replace('_', ' ')})" if "Drift" in cname else cname,
            )
    ax_a.axhline(6.150, color="black", linestyle=":", linewidth=1.2, label="Yesterday Baseline Overall (6.150%)")
    ax_a.set_xlabel("Actual Number of Retrains Used (2015–2026, Max Budget = 6)")
    ax_a.set_ylabel("Overall 2015–2026 Demand MAPE (%)")
    ax_a.set_title("(a) Retrain Budget Trade-Off Curve: Number of Retrains vs. Overall Error")
    ax_a.set_xticks(range(0, 7))
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=8.0)

    # Panel (b): Late-Era (2023–2026) Diurnal 48-Half-Hour MAPE Profile
    ax_b = axes[0, 1]
    hours = np.arange(N_PERIODS) / 2.0
    late_mask = dates_sim.year >= 2023
    diurnal_specs = [
        ("Frozen Stage 1 (0 Retrains)", "crimson", "-", 2.0),
        ("Baseline 1: Same HH Yesterday", "gray", "--", 1.5),
        ("Periodic-Expanding (6)", "darkorange", "-.", 1.9),
        ("Drift-Error-Expanding (Closed-Loop)", "royalblue", "--", 1.9),
        ("Periodic-Sliding-3Y (6)", "seagreen", ":", 2.0),
        ("Periodic-Recency H=548d (6)", "teal", "-", 2.1),
        ("Drift-Error-Recency H=548d (Closed-Loop)", "navy", "-", 2.3),
    ]
    for m_name, col, ls, lw in diurnal_specs:
        if m_name in preds_map:
            hh_mape = np.mean(
                np.abs(Y_sim[late_mask] - preds_map[m_name][late_mask]) / np.abs(Y_sim[late_mask]),
                axis=0,
            ) * 100.0
            ax_b.plot(hours, hh_mape, color=col, linestyle=ls, linewidth=lw, label=m_name)
    ax_b.axvspan(9.0, 16.0, color="gold", alpha=0.12, label="Midday Solar Duck-Curve Window")
    ax_b.set_xlabel("Time of Day (Hours, AEST)")
    ax_b.set_ylabel("Mean 2023–2026 Half-Hourly MAPE (%)")
    ax_b.set_title("(b) Late-Era (2023–2026) Diurnal Error Recovery Across 48 Half-Hours")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=8.0)

    # Panel (c): Window Length & Half-Life Sensitivity (2015–2024 vs 2025–2026 Holdout)
    ax_c = axes[1, 0]
    sl_sub = sens_df[sens_df["kind"] == "sliding"].sort_values("param_days")
    rc_sub = sens_df[sens_df["kind"] == "recency"].sort_values("param_days")
    ax_c.plot(sl_sub["param_years"], sl_sub["mape_2015_2024"], marker="o", linestyle="--", color="seagreen", linewidth=1.8, label="Sliding Window: 2015–2024 (Selection)")
    ax_c.plot(sl_sub["param_years"], sl_sub["mape_2025_2026"], marker="s", linestyle="-", color="darkgreen", linewidth=2.1, label="Sliding Window: 2025–2026 (Holdout Check)")
    ax_c.plot(rc_sub["param_years"], rc_sub["mape_2015_2024"], marker="^", linestyle="--", color="cornflowerblue", linewidth=1.8, label="Exponential Recency: 2015–2024 (Selection)")
    ax_c.plot(rc_sub["param_years"], rc_sub["mape_2025_2026"], marker="d", linestyle="-", color="navy", linewidth=2.1, label="Exponential Recency: 2025–2026 (Holdout Check)")
    ax_c.set_xlabel("Window Length W or Exponential Half-Life H (Years)")
    ax_c.set_ylabel("Demand MAPE (%)")
    ax_c.set_title("(c) Memory Horizon Sensitivity (6-Retrain Periodic): Selection vs. Holdout")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper left", fontsize=8.2)

    # Panel (d): Diurnal Standardised Lasso Coefficient Evolution (syd_solar_t)
    ax_d = axes[1, 1]
    if "syd_solar_t" in selected_features and len(coef_snapshots) >= 2:
        sol_idx = selected_features.index("syd_solar_t")
        snap_styles = {
            2014: ("2014 Frozen (2010–2014 Fit)", "crimson", "--"),
            2020: ("2020 Adapted (Recency-Weighted)", "darkorange", "-."),
            2025: ("2025 Adapted (Recency-Weighted)", "navy", "-"),
        }
        for yr_s, coef_mat in sorted(coef_snapshots.items()):
            lbl, col, ls = snap_styles.get(yr_s, (f"{yr_s} Adapted", "teal", "-"))
            ax_d.plot(hours, coef_mat[:, sol_idx], color=col, linestyle=ls, linewidth=2.2, label=lbl)
        ax_d.axhline(0.0, color="black", linestyle=":", linewidth=1.1)
    ax_d.set_xlabel("Time of Day (Hours, AEST)")
    ax_d.set_ylabel("Standardised Lasso Coef for syd_solar_t (MW / SD)")
    ax_d.set_title("(d) Evolution of Learned Midday Rooftop Solar Slope (syd_solar_t)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig12_retrain_budget_tradeoff_and_mechanisms.png"), dpi=200)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()

    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT_DIR, "data", "nem_weather_halfhourly_2010_2026.csv")
    out_dir = os.path.join(HERE, "analysis_outputs")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(HERE, "artifacts", "config.json"), "r", encoding="utf-8") as fh:
        config = json.load(fh)
    selected_features = config["selected_features"]

    with open(os.path.join(out_dir, "stage2_drift_detection_summary.json"), "r", encoding="utf-8") as fh:
        drift_summary = json.load(fh)

    print("[1/5] Loading dataset, base feature tensor, and calibrating Stage 1 frozen & error-detector baselines...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()

    bundle_2013 = build_candidate_features(df, day_type_transformer=None, fit_day_type_max_year=2013, include_flat_daily=False)
    bundle_2014 = build_candidate_features(df, day_type_transformer=None, fit_day_type_max_year=2014, include_flat_daily=False)

    dates = bundle_2014["target_dates"]
    feature_names = bundle_2014["feature_names"]
    selected_indices = [feature_names.index(f) for f in selected_features]
    Y_2d = bundle_2014["Y_2d"]

    tr13_mask = dates.year <= 2013
    va14_mask = dates.year == 2014
    rf14_mask = dates.year <= 2014
    sim_mask = dates.year >= 2015
    sim_indices = np.where(sim_mask)[0]
    dates_sim = dates[sim_mask]
    Y_sim = Y_2d[sim_mask]

    # Calibrate OnlineErrorDriftDetector strictly on 2014 out-of-sample validation errors
    m_2013 = PerSlotLinearRegressor(
        model_type="lasso", alpha=1.0, selected_indices=selected_indices,
        selected_feature_names=selected_features, day_type_transformer=bundle_2013["day_type_transformer"],
    ).fit(bundle_2013["X_3d"][tr13_mask], bundle_2013["Y_2d"][tr13_mask])
    preds_val14 = m_2013.predict_tensor(bundle_2013["X_3d"][va14_mask])
    e_val_2014 = (100.0 * np.abs(Y_2d[va14_mask] - preds_val14) / np.abs(Y_2d[va14_mask])).mean(axis=1)
    z_val_2014 = compute_causal_ewma(e_val_2014, span=EWMA_SPAN)
    err_detector = OnlineErrorDriftDetector().calibrate(e_val_2014, z_val_2014, warmup=30)

    # Pre-subset X_3d_2014 to selected 65 columns for fast leakage-free adaptation loops
    X_sub_initial = bundle_2014["X_3d"][:, :, selected_indices].copy()
    nsw_dem_all, idx_valid, _ = extract_daily_demand_matrix(df)
    daytype_col_specs = build_daytype_column_map(selected_features)

    # Frozen Stage 1 Model (2010–2014, 0 retrains)
    initial_model = PerSlotLinearRegressor(
        model_type="lasso", alpha=1.0, selected_indices=None,
        selected_feature_names=selected_features, day_type_transformer=bundle_2014["day_type_transformer"],
    ).fit(X_sub_initial[rf14_mask], Y_2d[rf14_mask])

    dem_d_idx = feature_names.index("dem_d_t")
    dem_d6_idx = feature_names.index("dem_d6_t")
    b1_preds_sim = bundle_2014["X_3d"][sim_mask, :, dem_d_idx]
    b7_preds_sim = bundle_2014["X_3d"][sim_mask, :, dem_d6_idx]
    frozen_preds_sim = initial_model.predict_tensor(X_sub_initial[sim_mask])

    frozen_mape = mape_metric(Y_sim, frozen_preds_sim)
    b1_mape = mape_metric(Y_sim, b1_preds_sim)

    # Convert Step 1 alarm date strings to sample indices
    date_to_sample_idx = {str(d.date()): int(i) for i, d in enumerate(dates)}
    input_alarm_indices = np.asarray(
        [date_to_sample_idx[a["alarm_date"]] for a in drift_summary["mode_b_input_alarms"]], dtype=int
    )
    err_frozen_alarm_indices = np.asarray(
        [date_to_sample_idx[a["alarm_date"]] for a in drift_summary["mode_b_error_frozen_stream_alarms"]], dtype=int
    )
    periodic_6_indices = get_periodic_retrain_indices(sim_indices, MAX_RETRAIN_BUDGET)

    print("[2/5] Running Stage 2 adaptation strategies (<= 6 retrains, Jan 2015 – Oct 2026)...")
    strategies: Dict[str, dict] = {}
    strategy_meta: List[Tuple[str, str, str, str]] = []

    def register(name: str, family: str, trigger: str, weight: str, res_dict: dict):
        strategies[name] = res_dict
        strategy_meta.append((name, family, trigger, weight))

    register(
        "Baseline 1: Same HH Yesterday", "Baseline", "None", "Lag=1d",
        {"preds_sim": b1_preds_sim, "n_retrains": 0, "retrain_dates": [], "effective_dates": []},
    )
    register(
        "Baseline 2: Same HH Last Week", "Baseline", "None", "Lag=7d",
        {"preds_sim": b7_preds_sim, "n_retrains": 0, "retrain_dates": [], "effective_dates": []},
    )
    register(
        "Frozen Stage 1 (0 Retrains)", "Frozen", "None", "2010–2014 Fixed",
        {"preds_sim": frozen_preds_sim, "n_retrains": 0, "retrain_dates": [], "effective_dates": []},
    )

    # Periodic Strategies (6 evenly spaced retrains)
    register(
        "Periodic-Expanding (6)", "Periodic", "Evenly Spaced (6)", "Expanding (Uniform)",
        run_fixed_schedule_strategy(
            periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "expanding",
        ),
    )
    register(
        "Periodic-Sliding-3Y (6)", "Periodic", "Evenly Spaced (6)", "Sliding 3Y (1095d)",
        run_fixed_schedule_strategy(
            periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "sliding", window_days=DEFAULT_SLIDING_3Y_DAYS,
        ),
    )
    register(
        "Periodic-Sliding-2Y (6)", "Periodic", "Evenly Spaced (6)", "Sliding 2Y (730d)",
        run_fixed_schedule_strategy(
            periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "sliding", window_days=DEFAULT_SLIDING_2Y_DAYS,
        ),
    )
    periodic_recency_res = run_fixed_schedule_strategy(
        periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
        nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "recency",
        halflife_days=DEFAULT_HALFLIFE_DAYS, capture_snapshots_years=[2014, 2020, 2025],
    )
    register(
        "Periodic-Recency H=548d (6)", "Periodic", "Evenly Spaced (6)", "Exp. Recency (H=548d)",
        periodic_recency_res,
    )

    # Input-Drift-Triggered Strategies (Step 1 Input-EWMA-Reset alarm dates)
    register(
        "Drift-Input-Expanding (6)", "Drift-Input", "Input-EWMA-Reset", "Expanding (Uniform)",
        run_fixed_schedule_strategy(
            input_alarm_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "expanding",
        ),
    )
    register(
        "Drift-Input-Sliding-3Y (6)", "Drift-Input", "Input-EWMA-Reset", "Sliding 3Y (1095d)",
        run_fixed_schedule_strategy(
            input_alarm_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "sliding", window_days=DEFAULT_SLIDING_3Y_DAYS,
        ),
    )
    register(
        "Drift-Input-Recency H=548d (6)", "Drift-Input", "Input-EWMA-Reset", "Exp. Recency (H=548d)",
        run_fixed_schedule_strategy(
            input_alarm_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "recency", halflife_days=DEFAULT_HALFLIFE_DAYS,
        ),
    )

    # Error-Drift-Triggered Strategies (Closed-Loop Online Detection)
    register(
        "Drift-Error-Expanding (Closed-Loop)", "Drift-Error", "Closed-Loop Error-EWMA", "Expanding (Uniform)",
        run_closed_loop_error_strategy(
            initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, err_detector, "expanding",
        ),
    )
    register(
        "Drift-Error-Sliding-3Y (Closed-Loop)", "Drift-Error", "Closed-Loop Error-EWMA", "Sliding 3Y (1095d)",
        run_closed_loop_error_strategy(
            initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, err_detector,
            "sliding", window_days=DEFAULT_SLIDING_3Y_DAYS,
        ),
    )
    register(
        "Drift-Error-Recency H=548d (Closed-Loop)", "Drift-Error", "Closed-Loop Error-EWMA", "Exp. Recency (H=548d)",
        run_closed_loop_error_strategy(
            initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, err_detector,
            "recency", halflife_days=DEFAULT_HALFLIFE_DAYS,
        ),
    )
    # Fixed-stream Error-EWMA schedule + Recency for controlled timing comparison
    register(
        "Drift-Error-Recency H=548d (Frozen-Stream Triggers)", "Drift-Error", "Frozen-Stream Error-EWMA", "Exp. Recency (H=548d)",
        run_fixed_schedule_strategy(
            err_frozen_alarm_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "recency", halflife_days=DEFAULT_HALFLIFE_DAYS,
        ),
    )

    # Export Table 14: Overall Adaptation Summary
    overall_rows = [
        summarize_strategy_metrics(name, fam, trig, wt, strategies[name], Y_sim, dates_sim, frozen_mape, b1_mape)
        for name, fam, trig, wt in strategy_meta
    ]
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(os.path.join(out_dir, "table14_stage2_adaptation_overall_summary.csv"), index=False)

    # Export Table 15: Yearly Metrics Across All Adaptation Strategies
    yearly_rows = []
    years_sim = dates_sim.year
    for name, _, _, _ in strategy_meta:
        y_hat = strategies[name]["preds_sim"]
        for yr in sorted(np.unique(years_sim)):
            ym = years_sim == yr
            yearly_rows.append({
                "strategy": name,
                "year": int(yr),
                "mape": mape_metric(Y_sim[ym], y_hat[ym]),
                "mae": mae_metric(Y_sim[ym], y_hat[ym]),
                "mean_signed_bias_mw": float(np.mean(y_hat[ym] - Y_sim[ym])),
            })
        yearly_rows.append({
            "strategy": name,
            "year": "Overall (2015-2026)",
            "mape": mape_metric(Y_sim, y_hat),
            "mae": mae_metric(Y_sim, y_hat),
            "mean_signed_bias_mw": float(np.mean(y_hat - Y_sim)),
        })
    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(os.path.join(out_dir, "table15_stage2_adaptation_yearly_metrics.csv"), index=False)

    print("[3/5] Computing Budget Trade-Off Curve (B = 0..6) & Window/Half-Life Sensitivity...")
    # Table 17: Budget Trade-Off Curve B = 0..6
    tradeoff_rows = []
    for b in range(0, MAX_RETRAIN_BUDGET + 1):
        # Periodic evenly spaced
        p_idx = get_periodic_retrain_indices(sim_indices, b)
        for label, w_mode in [
            ("Periodic-Expanding", "expanding"),
            ("Periodic-Sliding-3Y", "sliding"),
            ("Periodic-Recency (H=548d)", "recency"),
        ]:
            r = run_fixed_schedule_strategy(
                p_idx, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
                nsw_dem_all, idx_valid, daytype_col_specs, selected_features, w_mode,
                window_days=DEFAULT_SLIDING_3Y_DAYS, halflife_days=DEFAULT_HALFLIFE_DAYS,
            )
            tradeoff_rows.append({
                "budget_cap": b,
                "n_retrains_used": r["n_retrains"],
                "strategy_family": label,
                "allocation_mode": "evenly_spaced",
                "overall_mape_2015_2026": mape_metric(Y_sim, r["preds_sim"]),
                "overall_mae_2015_2026": mae_metric(Y_sim, r["preds_sim"]),
                "mape_2025_2026": mape_metric(Y_sim[years_sim >= 2025], r["preds_sim"][years_sim >= 2025]),
            })

        # Input-Triggered prefix cap (first b alarms)
        r_in = run_fixed_schedule_strategy(
            input_alarm_indices[:b], initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "recency",
            halflife_days=DEFAULT_HALFLIFE_DAYS,
        )
        tradeoff_rows.append({
            "budget_cap": b,
            "n_retrains_used": r_in["n_retrains"],
            "strategy_family": "Drift-Input-Recency (H=548d)",
            "allocation_mode": "chronological_prefix_cap",
            "overall_mape_2015_2026": mape_metric(Y_sim, r_in["preds_sim"]),
            "overall_mae_2015_2026": mae_metric(Y_sim, r_in["preds_sim"]),
            "mape_2025_2026": mape_metric(Y_sim[years_sim >= 2025], r_in["preds_sim"][years_sim >= 2025]),
        })

        # Closed-loop Error-Triggered: both chronological prefix cap and cooldown-scaled budget
        cd_scaled = COOLDOWN_DAYS if b == 0 else max(COOLDOWN_DAYS, int(3650 // b))
        for alloc_lbl, cd_days in [("chronological_prefix_cap", COOLDOWN_DAYS), ("cooldown_scaled", cd_scaled)]:
            if b in (0, MAX_RETRAIN_BUDGET) and alloc_lbl == "chronological_prefix_cap":
                # Same cooldown=365 when b==0 or b==6; still record both rows cleanly if desired,
                # or skip duplicate only when identical? Better keep both rows so curves are complete!
                pass
            for label, w_mode in [
                ("Drift-Error-Expanding", "expanding"),
                ("Drift-Error-Sliding-3Y", "sliding"),
                ("Drift-Error-Recency (H=548d)", "recency"),
            ]:
                r_err = run_closed_loop_error_strategy(
                    initial_model, X_sub_initial, Y_2d, dates, sim_indices,
                    nsw_dem_all, idx_valid, daytype_col_specs, selected_features, err_detector,
                    w_mode, window_days=DEFAULT_SLIDING_3Y_DAYS, halflife_days=DEFAULT_HALFLIFE_DAYS,
                    max_retrains=b, cooldown_days=cd_days,
                )
                tradeoff_rows.append({
                    "budget_cap": b,
                    "n_retrains_used": r_err["n_retrains"],
                    "strategy_family": label,
                    "allocation_mode": alloc_lbl,
                    "overall_mape_2015_2026": mape_metric(Y_sim, r_err["preds_sim"]),
                    "overall_mae_2015_2026": mae_metric(Y_sim, r_err["preds_sim"]),
                    "mape_2025_2026": mape_metric(Y_sim[years_sim >= 2025], r_err["preds_sim"][years_sim >= 2025]),
                })

    tradeoff_df = pd.DataFrame(tradeoff_rows)
    tradeoff_df.to_csv(os.path.join(out_dir, "table17_retrain_budget_tradeoff_curve.csv"), index=False)

    # Table 18: Window Length & Half-Life Sensitivity (Selection 2015–2024 vs Holdout 2025–2026)
    sens_rows = []
    m_sel = years_sim <= 2024
    m_hold = years_sim >= 2025
    for w_days in [365, 548, 730, 1095, 1460, 1825]:
        r_s = run_fixed_schedule_strategy(
            periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "sliding", window_days=w_days,
        )
        sens_rows.append({
            "kind": "sliding",
            "param_days": w_days,
            "param_years": round(w_days / 365.0, 2),
            "mape_2015_2024": mape_metric(Y_sim[m_sel], r_s["preds_sim"][m_sel]),
            "mae_2015_2024": mae_metric(Y_sim[m_sel], r_s["preds_sim"][m_sel]),
            "mape_2025_2026": mape_metric(Y_sim[m_hold], r_s["preds_sim"][m_hold]),
            "mae_2025_2026": mae_metric(Y_sim[m_hold], r_s["preds_sim"][m_hold]),
            "overall_mape_2015_2026": mape_metric(Y_sim, r_s["preds_sim"]),
            "overall_mae_2015_2026": mae_metric(Y_sim, r_s["preds_sim"]),
        })
    for h_days in [180, 365, 548, 730, 1095, 1460]:
        r_h = run_fixed_schedule_strategy(
            periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
            nsw_dem_all, idx_valid, daytype_col_specs, selected_features, "recency", halflife_days=h_days,
        )
        sens_rows.append({
            "kind": "recency",
            "param_days": h_days,
            "param_years": round(h_days / 365.0, 2),
            "mape_2015_2024": mape_metric(Y_sim[m_sel], r_h["preds_sim"][m_sel]),
            "mae_2015_2024": mae_metric(Y_sim[m_sel], r_h["preds_sim"][m_sel]),
            "mape_2025_2026": mape_metric(Y_sim[m_hold], r_h["preds_sim"][m_hold]),
            "mae_2025_2026": mae_metric(Y_sim[m_hold], r_h["preds_sim"][m_hold]),
            "overall_mape_2015_2026": mape_metric(Y_sim, r_h["preds_sim"]),
            "overall_mae_2015_2026": mae_metric(Y_sim, r_h["preds_sim"]),
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(out_dir, "table18_window_and_halflife_sensitivity.csv"), index=False)

    print("[4/5] Computing Monthly 95% CIs (Appendix A Day-Block Bootstrap, n_boot=1000)...")
    ci_keys = [
        "Frozen Stage 1 (0 Retrains)",
        "Baseline 1: Same HH Yesterday",
        "Periodic-Expanding (6)",
        "Periodic-Sliding-3Y (6)",
        "Periodic-Recency H=548d (6)",
        "Drift-Input-Recency H=548d (6)",
        "Drift-Error-Expanding (Closed-Loop)",
        "Drift-Error-Recency H=548d (Closed-Loop)",
    ]
    ci_preds_dict = {k: strategies[k]["preds_sim"] for k in ci_keys}
    monthly_df = compute_monthly_ci_table(dates_sim, Y_sim, ci_preds_dict, n_boot=1000, seed=0)
    monthly_df.to_csv(os.path.join(out_dir, "table16_stage2_adaptation_monthly_ci.csv"), index=False)

    preds_map = {k: v["preds_sim"] for k, v in strategies.items()}
    retrain_dates_map = {k: v["retrain_dates"] for k, v in strategies.items()}

    # Export JSON summary
    best_overall_row = overall_df[overall_df["family"] != "Baseline"].sort_values("overall_mape_2015_2026").iloc[0].to_dict()
    best_hold_row = overall_df[overall_df["family"] != "Baseline"].sort_values("mape_2025_2026").iloc[0].to_dict()
    ci_means = monthly_df.groupby("model")[["mape", "ind_ci_width", "boot_ci_width", "width_ratio_boot_over_ind"]].mean().to_dict(orient="index")

    json_payload = {
        "max_retrain_budget": MAX_RETRAIN_BUDGET,
        "simulation_days": int(len(dates_sim)),
        "simulation_months": 142,
        "overall_summary": overall_rows,
        "best_overall_strategy": best_overall_row,
        "best_holdout_2025_2026_strategy": best_hold_row,
        "monthly_ci_summary_142_months": ci_means,
        "sensitivity_results": sens_rows,
    }
    with open(os.path.join(out_dir, "stage2_adaptation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2)

    print("[5/5] Generating Figures 11 & 12...")
    plot_adaptation_figures(
        yearly_df=yearly_df,
        monthly_df=monthly_df,
        tradeoff_df=tradeoff_df,
        sens_df=sens_df,
        Y_sim=Y_sim,
        dates_sim=dates_sim,
        preds_map=preds_map,
        retrain_dates_map=retrain_dates_map,
        coef_snapshots=periodic_recency_res["coef_snapshots"],
        selected_features=selected_features,
        out_dir=out_dir,
    )

    print("\n=== Stage 2 Adaptation Overall Summary (Jan 2015 – Oct 2026, Max 6 Retrains) ===")
    disp_cols = [
        "strategy", "n_retrains", "mape_2015_2024", "mape_2025_2026",
        "overall_mape_2015_2026", "overall_mae_2015_2026", "midday_09_16_mape", "improv_vs_frozen_pct",
    ]
    print(overall_df[disp_cols].round(3).to_string(index=False))

    print("\n=== Yearly Demand MAPE (%) Across Key Adaptation Strategies ===")
    yr_pivot = yearly_df.pivot(index="year", columns="strategy", values="mape")[ci_keys]
    print(yr_pivot.round(3).to_string())

    print("\n=== Window & Half-Life Sensitivity (6-Retrain Periodic) ===")
    print(sens_df.round(3).to_string(index=False))
    print(f"\nCompleted Step 2 (Stage 2 Adaptation Comparison) in {time.time() - t0:.1f}s. Saved outputs to {out_dir}/.")


if __name__ == "__main__":
    main()

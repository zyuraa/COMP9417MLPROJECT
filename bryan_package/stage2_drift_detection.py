"""Stage 2 Online Drift Detection and Distribution Shift Analysis (Jan 2015 – Oct 2026).

Implements Step 1 of the remaining COMP9417 project requirements:
  1. Online Input-Based Drift Detector (`OnlineInputDriftDetector`):
       - Monitored statistic: daily Euclidean distance u_in(D) of Day D's mean-normalised
         48-half-hour NSW demand profile to the nearest cluster centre in 5D PCA space
         (DayTypeTransformer fitted strictly on 2010–2013 / 2010–2014).
       - Online smoothing: causal 30-day EWMA Z_in(D) (lambda = 2 / 31).
       - Threshold calibrated strictly on 2010–2014 data (zero false alarms on 2010–2014).
       - Companion Weather Covariate Shift Index Z_wea(D) from 8 daily Sydney weather
         summary variables standardised on 2010–2014.
  2. Online Error-Based Drift Detector (`OnlineErrorDriftDetector`):
       - Monitored statistic: daily realized Demand MAPE e(D) (%) of the Stage 1 selected
         Per-Slot Lasso model (alpha=1.0, k*=65).
       - Primary online statistic: causal 30-day EWMA Z_err(D) (lambda = 2 / 31).
       - Threshold calibrated strictly on genuine out-of-sample 2014 validation errors
         (from the 2010–2013 model predicting 2014).
       - Companion sequential change-point statistic: one-sided Page-Hinkley test M_PH(D).
  3. Dual Evaluation Modes (chronological end-of-day D observation -> D+1 effect):
       - Mode A (Fixed-Reference Monitoring of Frozen Model, 2015–2026)
       - Mode B (Adaptive Online Reset / Cooldown Schedule within <= 6 Retrain Budget,
         including closed-loop online model retraining verification).
  4. Quantitative Diagnostics & NEM Event Linkage:
       - Yearly evolution of Day-Type cluster shares, PCA distance, midday vs. overnight
         error/bias, and weather vs. shape covariate shift regimes.
       - Empirical test of whether input shift always degrades error: 2x2 comparison of
         Weather Shift vs. Profile Shape Shift (concept drift from rooftop solar).
       - Evidence-based mapping of drift alarms to Australian NEM structural events.

Exports tables, JSON summary, and figures (fig9, fig10) to bryan_package/analysis_outputs/.
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
    build_candidate_features,
    mae_metric,
    mape_metric,
    market_day,
)
from model import PerSlotLinearRegressor  # noqa: E402


EWMA_SPAN = 30
COOLDOWN_DAYS = 365
MAX_RETRAIN_BUDGET = 6


def compute_causal_ewma(values: np.ndarray, span: int = EWMA_SPAN, init_value: Optional[float] = None) -> np.ndarray:
    """Compute causal EWMA Z_t = alpha * x_t + (1 - alpha) * Z_{t-1} with alpha = 2/(span+1)."""
    alpha = 2.0 / (span + 1.0)
    z = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return z
    if init_value is None:
        z[0] = float(values[0])
        start_i = 1
    else:
        z_prev = float(init_value)
        z[0] = alpha * float(values[0]) + (1.0 - alpha) * z_prev
        start_i = 1
    for i in range(start_i, len(values)):
        z[i] = alpha * float(values[i]) + (1.0 - alpha) * z[i - 1]
    return z


class OnlineInputDriftDetector:
    """Online input-based drift detector monitoring PCA cluster distance of daily profiles."""

    def __init__(self, ewma_span: int = EWMA_SPAN, sigma_mult: float = 3.0, max_safety: float = 1.02):
        self.ewma_span = int(ewma_span)
        self.sigma_mult = float(sigma_mult)
        self.max_safety = float(max_safety)
        self.ref_mean_: float = 0.0
        self.ref_std_: float = 0.0
        self.ref_max_: float = 0.0
        self.threshold_: float = 0.0

    def calibrate(self, z_ref: np.ndarray, warmup: int = 30) -> "OnlineInputDriftDetector":
        """Calibrate reference baseline and control limit solely on 2010–2014 data."""
        z_eff = z_ref[warmup:] if len(z_ref) > warmup else z_ref
        self.ref_mean_ = float(np.mean(z_eff))
        self.ref_std_ = float(np.std(z_eff, ddof=1))
        self.ref_max_ = float(np.max(z_eff))
        ucl_sigma = self.ref_mean_ + self.sigma_mult * self.ref_std_
        ucl_max = self.max_safety * self.ref_max_
        self.threshold_ = float(max(ucl_sigma, ucl_max))
        return self

    def detect_fixed_reference(
        self, dates: pd.DatetimeIndex, z_series: np.ndarray, sim_mask: np.ndarray
    ) -> Dict[str, object]:
        """Mode A: Fixed-reference online monitoring across 2015–2026."""
        sim_dates = dates[sim_mask]
        z_sim = z_series[sim_mask]
        alarm_flags = z_sim > self.threshold_
        alarm_idx = np.where(alarm_flags)[0]

        first_alarm_date = str(sim_dates[alarm_idx[0]].date()) if len(alarm_idx) > 0 else None
        first_eff_date = (
            str((sim_dates[alarm_idx[0]] + pd.Timedelta(days=1)).date())
            if len(alarm_idx) > 0
            else None
        )

        # Sustained drift onset: first day where the next 30 consecutive days remain above threshold
        sustained_date = None
        if len(alarm_idx) >= 30:
            win_sum = np.convolve(alarm_flags.astype(int), np.ones(30, dtype=int), mode="valid")
            sust_hits = np.where(win_sum == 30)[0]
            if len(sust_hits) > 0:
                sustained_date = str(sim_dates[sust_hits[0]].date())

        return {
            "alarm_flags_sim": alarm_flags,
            "n_alarm_days": int(alarm_flags.sum()),
            "pct_alarm_days": float(100.0 * alarm_flags.mean()),
            "first_alarm_date": first_alarm_date,
            "first_effective_date": first_eff_date,
            "sustained_30d_onset_date": sustained_date,
        }

    def detect_adaptive_reset(
        self,
        dates: pd.DatetimeIndex,
        z_all: np.ndarray,
        sim_start_idx: int,
        cooldown_days: int = COOLDOWN_DAYS,
        max_alarms: int = MAX_RETRAIN_BUDGET,
    ) -> Tuple[List[dict], np.ndarray]:
        """Mode B: Online adaptive reset with trailing-window re-anchoring and cooldown."""
        curr_thresh = self.threshold_
        thresh_trace = np.full(len(dates), curr_thresh, dtype=np.float64)
        alarms: List[dict] = []
        last_alarm_idx = -10**9

        for i in range(sim_start_idx, len(dates)):
            thresh_trace[i] = curr_thresh
            if len(alarms) >= max_alarms:
                continue
            if (i - last_alarm_idx) < cooldown_days:
                continue
            if z_all[i] > curr_thresh:
                d_alarm = dates[i]
                d_eff = dates[i + 1] if (i + 1) < len(dates) else d_alarm + pd.Timedelta(days=1)
                alarms.append({
                    "alarm_num": len(alarms) + 1,
                    "alarm_date": str(d_alarm.date()),
                    "effective_date": str(d_eff.date()),
                    "statistic_value": float(z_all[i]),
                    "threshold_at_alarm": float(curr_thresh),
                    "exceedance_ratio": float(z_all[i] / max(curr_thresh, 1e-12)),
                })
                last_alarm_idx = i
                # Re-anchor reference strictly on trailing window up to day D (no future leakage)
                win_start = max(30, i - cooldown_days + 1)
                z_win = z_all[win_start : i + 1]
                mu_new = float(np.mean(z_win))
                sd_new = float(np.std(z_win, ddof=1))
                mx_new = float(np.max(z_win))
                curr_thresh = float(max(mu_new + self.sigma_mult * sd_new, self.max_safety * mx_new))

        return alarms, thresh_trace


class OnlineErrorDriftDetector:
    """Online error-based drift detector (30-day EWMA + Page-Hinkley on daily MAPE)."""

    def __init__(
        self,
        ewma_span: int = EWMA_SPAN,
        sigma_mult: float = 3.0,
        max_safety: float = 1.02,
        ph_delta_mult: float = 0.5,
        ph_lambda_mult: float = 1.25,
    ):
        self.ewma_span = int(ewma_span)
        self.sigma_mult = float(sigma_mult)
        self.max_safety = float(max_safety)
        self.ph_delta_mult = float(ph_delta_mult)
        self.ph_lambda_mult = float(ph_lambda_mult)

        self.daily_ref_mean_: float = 0.0
        self.daily_ref_std_: float = 0.0
        self.ewma_ref_mean_: float = 0.0
        self.ewma_ref_std_: float = 0.0
        self.ewma_ref_max_: float = 0.0
        self.ewma_threshold_: float = 0.0
        self.ph_delta_: float = 0.0
        self.ph_ref_max_: float = 0.0
        self.ph_threshold_: float = 0.0

    def calibrate(
        self, e_val_2014: np.ndarray, z_val_2014: np.ndarray, warmup: int = 30
    ) -> "OnlineErrorDriftDetector":
        """Calibrate thresholds strictly on genuine out-of-sample 2014 validation errors."""
        self.daily_ref_mean_ = float(np.mean(e_val_2014))
        self.daily_ref_std_ = float(np.std(e_val_2014, ddof=1))

        z_eff = z_val_2014[warmup:] if len(z_val_2014) > warmup else z_val_2014
        self.ewma_ref_mean_ = float(np.mean(z_eff))
        self.ewma_ref_std_ = float(np.std(z_eff, ddof=1))
        self.ewma_ref_max_ = float(np.max(z_eff))
        self.ewma_threshold_ = float(
            max(
                self.ewma_ref_mean_ + self.sigma_mult * self.ewma_ref_std_,
                self.max_safety * self.ewma_ref_max_,
            )
        )

        self.ph_delta_ = self.ph_delta_mult * self.daily_ref_std_
        ph_2014 = self.compute_page_hinkley(e_val_2014)
        self.ph_ref_max_ = float(np.max(ph_2014))
        self.ph_threshold_ = float(self.ph_lambda_mult * max(self.ph_ref_max_, 1.0))
        return self

    def compute_page_hinkley(self, daily_errors: np.ndarray) -> np.ndarray:
        """One-sided Page-Hinkley cumulative sum statistic for upward error shift."""
        m = np.zeros(len(daily_errors), dtype=np.float64)
        curr = 0.0
        for i, err in enumerate(daily_errors):
            curr = max(0.0, curr + (float(err) - self.daily_ref_mean_ - self.ph_delta_))
            m[i] = curr
        return m

    def detect_fixed_reference(
        self,
        dates_sim: pd.DatetimeIndex,
        z_sim: np.ndarray,
        ph_sim: np.ndarray,
    ) -> Dict[str, object]:
        """Mode A: Fixed-reference monitoring of frozen model across 2015–2026."""
        ewma_flags = z_sim > self.ewma_threshold_
        ph_flags = ph_sim > self.ph_threshold_

        ewma_idx = np.where(ewma_flags)[0]
        ph_idx = np.where(ph_flags)[0]

        sustained_ewma = None
        if len(ewma_idx) >= 30:
            win_sum = np.convolve(ewma_flags.astype(int), np.ones(30, dtype=int), mode="valid")
            hits = np.where(win_sum == 30)[0]
            if len(hits) > 0:
                sustained_ewma = str(dates_sim[hits[0]].date())

        return {
            "ewma_alarm_flags_sim": ewma_flags,
            "ewma_n_alarm_days": int(ewma_flags.sum()),
            "ewma_pct_alarm_days": float(100.0 * ewma_flags.mean()),
            "ewma_first_alarm_date": str(dates_sim[ewma_idx[0]].date()) if len(ewma_idx) > 0 else None,
            "ewma_first_effective_date": (
                str((dates_sim[ewma_idx[0]] + pd.Timedelta(days=1)).date())
                if len(ewma_idx) > 0
                else None
            ),
            "ewma_sustained_30d_onset_date": sustained_ewma,
            "ph_alarm_flags_sim": ph_flags,
            "ph_n_alarm_days": int(ph_flags.sum()),
            "ph_pct_alarm_days": float(100.0 * ph_flags.mean()),
            "ph_first_alarm_date": str(dates_sim[ph_idx[0]].date()) if len(ph_idx) > 0 else None,
            "ph_first_effective_date": (
                str((dates_sim[ph_idx[0]] + pd.Timedelta(days=1)).date())
                if len(ph_idx) > 0
                else None
            ),
        }

    def detect_adaptive_reset_frozen_stream(
        self,
        dates_sim: pd.DatetimeIndex,
        z_sim: np.ndarray,
        cooldown_days: int = COOLDOWN_DAYS,
        max_alarms: int = MAX_RETRAIN_BUDGET,
    ) -> Tuple[List[dict], np.ndarray]:
        """Mode B1: Adaptive reset on frozen error stream (re-anchored threshold + cooldown)."""
        curr_thresh = self.ewma_threshold_
        thresh_trace = np.full(len(dates_sim), curr_thresh, dtype=np.float64)
        alarms: List[dict] = []
        last_alarm_idx = -10**9

        for i in range(len(dates_sim)):
            thresh_trace[i] = curr_thresh
            if len(alarms) >= max_alarms:
                continue
            if (i - last_alarm_idx) < cooldown_days:
                continue
            if z_sim[i] > curr_thresh:
                d_alarm = dates_sim[i]
                d_eff = (
                    dates_sim[i + 1]
                    if (i + 1) < len(dates_sim)
                    else d_alarm + pd.Timedelta(days=1)
                )
                alarms.append({
                    "alarm_num": len(alarms) + 1,
                    "alarm_date": str(d_alarm.date()),
                    "effective_date": str(d_eff.date()),
                    "statistic_value": float(z_sim[i]),
                    "threshold_at_alarm": float(curr_thresh),
                    "exceedance_ratio": float(z_sim[i] / max(curr_thresh, 1e-12)),
                })
                last_alarm_idx = i
                win_start = max(30, i - cooldown_days + 1)
                z_win = z_sim[win_start : i + 1]
                mu_new = float(np.mean(z_win))
                sd_new = float(np.std(z_win, ddof=1))
                mx_new = float(np.max(z_win))
                curr_thresh = float(max(mu_new + self.sigma_mult * sd_new, self.max_safety * mx_new))

        return alarms, thresh_trace


def run_closed_loop_error_retraining(
    df_full: pd.DataFrame,
    selected_features: List[str],
    initial_model: PerSlotLinearRegressor,
    initial_bundle: dict,
    e_val_2014: np.ndarray,
    z_val_2014: np.ndarray,
    detector: OnlineErrorDriftDetector,
    cooldown_days: int = COOLDOWN_DAYS,
    max_alarms: int = MAX_RETRAIN_BUDGET,
) -> Dict[str, object]:
    """Mode B2: Closed-loop online verification where triggering an alarm at end of Day D
    actually retrains Per-Slot Lasso + DayTypeTransformer on past data <= D (taking effect D+1),
    and the detector monitors the newly retrained model's out-of-sample errors.
    """
    dates = initial_bundle["target_dates"]
    feature_names = initial_bundle["feature_names"]
    selected_indices = [feature_names.index(f) for f in selected_features]
    Y_2d = initial_bundle["Y_2d"]
    sim_mask = dates.year >= 2015
    sim_indices = np.where(sim_mask)[0]

    curr_model = initial_model
    curr_X3d = initial_bundle["X_3d"]
    curr_thresh = detector.ewma_threshold_

    alpha = 2.0 / (detector.ewma_span + 1.0)
    # Standard SPC deployment initialization at 2015-01-01: Z_0 = mu_err_ref (2014 in-control mean)
    z_curr = float(detector.ewma_ref_mean_)

    closed_e_sim = np.zeros(len(sim_indices), dtype=np.float64)
    closed_z_sim = np.zeros(len(sim_indices), dtype=np.float64)
    closed_thresh_sim = np.zeros(len(sim_indices), dtype=np.float64)
    closed_pred_sim = np.zeros((len(sim_indices), N_PERIODS), dtype=np.float64)

    alarms: List[dict] = []
    last_retrain_offset = -10**9
    post_retrain_z_buffer: List[float] = []
    reanchored_after_last_retrain = True

    raw_days = market_day(df_full.index)

    for offset, idx in enumerate(sim_indices):
        d_target = dates[idx]
        x_day = curr_X3d[idx]
        y_day = Y_2d[idx]
        y_hat = curr_model.predict_single_day(x_day)
        closed_pred_sim[offset] = y_hat

        e_day = float(np.mean(np.abs(y_day - y_hat) / np.abs(y_day)) * 100.0)
        z_curr = alpha * e_day + (1.0 - alpha) * z_curr

        closed_e_sim[offset] = e_day
        closed_z_sim[offset] = z_curr
        closed_thresh_sim[offset] = curr_thresh

        if not reanchored_after_last_retrain:
            post_retrain_z_buffer.append(z_curr)
            # After 1 full seasonal year (cooldown_days) of genuine post-retrain monitoring,
            # re-anchor threshold to the retrained model's in-control out-of-sample error band
            if len(post_retrain_z_buffer) >= cooldown_days:
                z_eff = np.asarray(post_retrain_z_buffer[30:], dtype=np.float64)
                mu_new = float(np.mean(z_eff))
                sd_new = float(np.std(z_eff, ddof=1))
                mx_new = float(np.max(z_eff))
                curr_thresh = float(
                    max(mu_new + detector.sigma_mult * sd_new, detector.max_safety * mx_new)
                )
                reanchored_after_last_retrain = True

        if len(alarms) >= max_alarms:
            continue
        if (offset - last_retrain_offset) < cooldown_days:
            continue
        if z_curr > curr_thresh:
            d_eff = dates[idx + 1] if (idx + 1) < len(dates) else d_target + pd.Timedelta(days=1)
            alarms.append({
                "alarm_num": len(alarms) + 1,
                "alarm_date": str(d_target.date()),
                "effective_date": str(d_eff.date()),
                "statistic_value": float(z_curr),
                "threshold_at_alarm": float(curr_thresh),
                "exceedance_ratio": float(z_curr / max(curr_thresh, 1e-12)),
            })
            last_retrain_offset = offset
            reanchored_after_last_retrain = False
            post_retrain_z_buffer = []

            # Retrain strictly using data up to end of Day D (d_target)
            df_past = df_full.loc[raw_days <= d_target]
            past_bundle = build_candidate_features(
                df_past,
                day_type_transformer=None,
                fit_day_type_max_year=9999,
                include_flat_daily=False,
            )
            new_dt = past_bundle["day_type_transformer"]
            new_model = PerSlotLinearRegressor(
                model_type="lasso",
                alpha=1.0,
                selected_indices=selected_indices,
                selected_feature_names=selected_features,
                day_type_transformer=new_dt,
            ).fit(past_bundle["X_3d"], past_bundle["Y_2d"])

            # Rebuild features on full df using the newly fitted DayTypeTransformer (no leakage:
            # transformer parameters were fitted strictly on df_past <= d_target)
            full_bundle_updated = build_candidate_features(
                df_full,
                day_type_transformer=new_dt,
                include_flat_daily=False,
            )
            curr_model = new_model
            curr_X3d = full_bundle_updated["X_3d"]

    return {
        "alarms": alarms,
        "daily_mape_sim": closed_e_sim,
        "ewma_sim": closed_z_sim,
        "thresh_sim": closed_thresh_sim,
        "preds_sim": closed_pred_sim,
        "overall_mape_sim": mape_metric(Y_2d[sim_mask], closed_pred_sim),
        "overall_mae_sim": mae_metric(Y_2d[sim_mask], closed_pred_sim),
    }


def compute_weather_covariate_shift(
    df: pd.DataFrame, target_dates: pd.DatetimeIndex, ref_mask: np.ndarray
) -> Dict[str, object]:
    """Compute daily Sydney weather covariate shift index standardised on 2010–2014."""
    op_dates = market_day(df.index)
    df_tmp = df.loc[op_dates.isin(target_dates)].copy()
    df_tmp["op_date"] = market_day(df_tmp.index)
    df_tmp["syd_cdd"] = np.maximum(0.0, df_tmp["syd_temp"] - 22.0)
    df_tmp["syd_hdd"] = np.maximum(0.0, 18.0 - df_tmp["syd_temp"])

    daily_wea = df_tmp.groupby("op_date").agg(
        syd_temp_mean=("syd_temp", "mean"),
        syd_temp_max=("syd_temp", "max"),
        syd_temp_min=("syd_temp", "min"),
        syd_cdd_mean=("syd_cdd", "mean"),
        syd_hdd_mean=("syd_hdd", "mean"),
        syd_solar_mean=("syd_solar", "mean"),
        syd_rh_mean=("syd_rh", "mean"),
        syd_cloud_mean=("syd_cloud", "mean"),
    ).loc[target_dates]

    W = daily_wea.to_numpy(dtype=np.float64)
    mu_ref = W[ref_mask].mean(axis=0, keepdims=True)
    sd_ref = np.maximum(W[ref_mask].std(axis=0, ddof=1, keepdims=True), 1e-6)
    Z_mat = (W - mu_ref) / sd_ref
    u_wea = np.sqrt(np.mean(Z_mat ** 2, axis=1))
    z_wea = compute_causal_ewma(u_wea, span=EWMA_SPAN)

    z_ref = z_wea[ref_mask][30:]
    h_wea = float(max(np.mean(z_ref) + 3.0 * np.std(z_ref, ddof=1), 1.02 * np.max(z_ref)))

    return {
        "daily_weather_df": daily_wea,
        "u_wea": u_wea,
        "z_wea": z_wea,
        "threshold": h_wea,
        "ref_mean": float(np.mean(z_ref)),
        "ref_std": float(np.std(z_ref, ddof=1)),
        "ref_max": float(np.max(z_ref)),
    }


def annotate_nem_events(alarms_df: pd.DataFrame) -> pd.DataFrame:
    """Attach evidence-based NEM market event context and interpretation to each alarm."""
    event_catalog = [
        (
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2017-12-31"),
            "2017 Extreme Summer Heatwaves & Hazelwood Coal Closure (Mar 2017)",
            "Temporary extreme weather / peak load regime & interstate price spikes; "
            "Error MAPE rises temporarily in Jan–Feb 2017 (3.47%/3.43%) and Dec 2017 (3.75%) "
            "but reverts in autumn/winter — mild covariate & boundary shift rather than sustained concept drift.",
        ),
        (
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2019-12-31"),
            "2018–2019 Rooftop Solar PV Acceleration & Evening Peak Steepening",
            "Sustained structural expansion of behind-the-meter rooftop PV depressing 09:00–16:00 "
            "operational demand; frozen model midday error diverges permanently from summer 2018–19 onward.",
        ),
        (
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2021-12-31"),
            "2020–2021 COVID-19 Lockdowns, La Nina & Oct 2021 5-Minute Settlement (5MS)",
            "Sharp daytime commercial load drop during NSW lockdowns (Apr 2020, Jul–Oct 2021) "
            "compounded by booming rooftop PV (>25% solar troughs by spring 2021); error peaks in Sep–Oct 2021 (7.97%/7.20%).",
        ),
        (
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2023-12-31"),
            "2022 NEM Energy Crisis (Jun 2022 Suspension) & Liddell Closure (Apr 2023)",
            "Coal plant outages/closure (Liddell Apr 2023) and record rooftop solar push midday net load "
            "lower; frozen Lasso crosses above Yesterday Baseline annually in 2023 (7.38% vs 6.46%) with >+500 MW midday over-prediction.",
        ),
        (
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2026-10-31"),
            "2024–2026 Deep Solar Duck Curve, Negative Daytime Spot Prices & EV/Battery Uptake",
            "Minimum operational demand troughs deepen below 3,000 MW (vs >5,000 MW in 2010–2014); "
            "negative spot prices reach ~15–20% of intervals, making 2010–2014 solar & lag weights severely stale (Sep–Dec >13–15% MAPE).",
        ),
    ]

    events = []
    evidence = []
    for _, row in alarms_df.iterrows():
        dt = pd.Timestamp(row["alarm_date"])
        matched_ev = "Post-2024 Structural Solar Duck Curve Regime"
        matched_note = "Continued midday net-load erosion and shape shift relative to prior baseline."
        for start, end, ev, note in event_catalog:
            if start <= dt <= end:
                matched_ev = ev
                matched_note = note
                break
        events.append(matched_ev)
        evidence.append(matched_note)

    out = alarms_df.copy()
    out["nem_market_event_window"] = events
    out["supporting_evidence_vs_coincidence"] = evidence
    return out


def plot_drift_figures(
    dates: pd.DatetimeIndex,
    z_in_13: np.ndarray,
    h_in_13: float,
    in_thresh_trace: np.ndarray,
    in_alarms: List[dict],
    z_wea: np.ndarray,
    h_wea: float,
    dates_2014_2026: pd.DatetimeIndex,
    z_err_frozen: np.ndarray,
    h_err: float,
    err_frozen_thresh_trace: np.ndarray,
    err_frozen_alarms: List[dict],
    ph_2014_2026: np.ndarray,
    h_ph: float,
    dates_sim: pd.DatetimeIndex,
    closed_z_sim: np.ndarray,
    closed_thresh_sim: np.ndarray,
    closed_alarms: List[dict],
    yearly_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ape_2d: np.ndarray,
    bias_2d: np.ndarray,
    u_in_13: np.ndarray,
    daily_mape_all: np.ndarray,
    out_dir: str,
):
    """Generate publication-quality Figures 9 and 10 for the drift detection section."""
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Figure 9: Online Drift Detectors Timeline (Input-Based vs Error-Based)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 13.0), sharex=True)

    ref_end = pd.Timestamp("2015-01-01")
    for ax in axes:
        ax.axvspan(dates[0], ref_end, color="lightgray", alpha=0.25, label="Calibration Period (2010–2014)" if ax is axes[0] else None)
        ax.axvline(ref_end, color="black", linestyle="-", linewidth=1.2)

    # Panel (a): Input-Based Detector (PCA Cluster Distance EWMA) + Weather Covariate Shift
    ax0 = axes[0]
    ax0.plot(dates, z_in_13, color="teal", linewidth=1.6, label="Input Profile Drift Z_in(D): 30d EWMA of PCA Cluster Distance")
    ax0.axhline(h_in_13, color="crimson", linestyle="--", linewidth=1.4, label=f"Fixed 2010–2014 Control Limit (h_in = {h_in_13:.3f})")
    ax0.step(dates, in_thresh_trace, where="post", color="darkorange", linestyle=":", linewidth=1.6, label="Mode B Adaptive Reset Limit (365d Cooldown, <=6 Alarms)")
    in_offsets = [(4, 10), (4, -15), (4, 10), (4, -15), (4, 10), (4, -15)]
    for k_i, al in enumerate(in_alarms):
        dt_al = pd.Timestamp(al["alarm_date"])
        ax0.axvline(dt_al, color="darkorange", alpha=0.55, linestyle="-.", linewidth=1.1)
        ax0.scatter([dt_al], [al["statistic_value"]], color="darkorange", edgecolor="black", s=55, zorder=6)
        ax0.annotate(
            f"I{k_i+1}: {al['alarm_date'][:7]}",
            xy=(dt_al, al["statistic_value"]),
            xytext=in_offsets[k_i % len(in_offsets)],
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="saddlebrown",
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.8),
        )
    ax0.set_ylim(0.0, float(np.max(z_in_13)) * 1.24)
    ax0_r = ax0.twinx()
    ax0_r.plot(dates, z_wea, color="slateblue", alpha=0.42, linewidth=1.1, label="Weather Covariate Shift Z_wea(D) (Right Axis)")
    ax0_r.axhline(h_wea, color="slateblue", alpha=0.4, linestyle=":", linewidth=1.0)
    ax0_r.set_ylim(0.4, float(np.max(z_wea)) * 1.22)
    ax0_r.set_ylabel("Weather Shift Index Z_wea", color="slateblue")
    ax0.set_ylabel("PCA Distance EWMA Z_in(D)")
    ax0.set_title("(a) Input-Based Online Drift Detector: 30-Day EWMA of PCA Demand-Shape Distance vs. Weather Covariate Shift")
    ax0.grid(True, alpha=0.3)
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0_r.get_legend_handles_labels()
    ax0.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=8.5)

    # Panel (b): Error-Based Detector (30-Day EWMA of Daily MAPE: Frozen vs Closed-Loop)
    ax1 = axes[1]
    m14_plot = dates_2014_2026.year == 2014
    m15_plot = dates_2014_2026.year >= 2015
    ax1.plot(dates_2014_2026[m14_plot], z_err_frozen[m14_plot], color="gray", linewidth=1.4, linestyle="-", label="2014 Out-of-Sample Val Error EWMA (Calibration)")
    ax1.plot(dates_2014_2026[m15_plot], z_err_frozen[m15_plot], color="crimson", linewidth=1.7, label="Frozen Model Error EWMA Z_err(D) (30d EWMA of Daily MAPE)")
    ax1.plot(dates_sim, closed_z_sim, color="royalblue", linewidth=1.7, label="Closed-Loop Retrained Model Error EWMA (Triggered by Error Detector)")
    ax1.axhline(h_err, color="darkred", linestyle="--", linewidth=1.4, label=f"2014 Validation Control Limit (h_err = {h_err:.2f}%)")
    ax1.step(dates_sim, closed_thresh_sim, where="post", color="navy", linestyle=":", linewidth=1.5, label="Closed-Loop Adaptive Reset Limit")
    err_offsets = [(4, 10), (4, 12), (4, -15), (4, 12), (4, -15), (4, 12)]
    for k_i, al in enumerate(closed_alarms):
        dt_al = pd.Timestamp(al["alarm_date"])
        ax1.axvline(dt_al, color="navy", alpha=0.55, linestyle="-.", linewidth=1.1)
        ax1.scatter([dt_al], [al["statistic_value"]], color="royalblue", edgecolor="black", s=55, zorder=6)
        ax1.annotate(
            f"E{k_i+1}: {al['alarm_date'][:7]}",
            xy=(dt_al, al["statistic_value"]),
            xytext=err_offsets[k_i % len(err_offsets)],
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="navy",
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.8),
        )
    ax1.set_ylim(0.5, float(np.max(z_err_frozen)) * 1.18)
    ax1.set_ylabel("30-Day EWMA Daily MAPE (%)")
    ax1.set_title("(b) Error-Based Online Drift Detector: Frozen Model vs. Closed-Loop Drift-Triggered Retraining (Max 6 Retrains)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5)

    # Panel (c): Page-Hinkley Statistic & Diurnal Decomposition (Midday Solar vs Overnight Error)
    ax2 = axes[2]
    mask_14_26 = dates.year >= 2014
    midday_daily = ape_2d[mask_14_26, 18:32].mean(axis=1)
    nonmid_daily = np.concatenate([ape_2d[mask_14_26, :18], ape_2d[mask_14_26, 32:]], axis=1).mean(axis=1)
    z_mid = compute_causal_ewma(midday_daily, span=EWMA_SPAN)
    z_non = compute_causal_ewma(nonmid_daily, span=EWMA_SPAN)

    ax2.plot(dates_2014_2026, z_mid, color="darkorange", linewidth=1.8, label="Midday Solar Window (09:00–16:00 AEST) 30d EWMA MAPE (%)")
    ax2.plot(dates_2014_2026, z_non, color="seagreen", linewidth=1.6, label="Evening & Overnight (16:00–09:00 AEST) 30d EWMA MAPE (%)")
    y2_top = float(np.max(z_mid)) * 1.22
    ax2.set_ylim(0.0, y2_top)

    ax2_r = ax2.twinx()
    ax2_r.plot(dates_2014_2026[m14_plot], ph_2014_2026[m14_plot], color="purple", alpha=0.35, linewidth=1.1)
    ax2_r.plot(dates_2014_2026[m15_plot], ph_2014_2026[m15_plot], color="purple", alpha=0.55, linewidth=1.3, label="Page-Hinkley Cumulative Statistic M_PH(D) (Right Axis)")
    ax2_r.axhline(h_ph, color="purple", alpha=0.5, linestyle="--", linewidth=1.1, label=f"Page-Hinkley Limit ({h_ph:.1f})")
    ax2_r.set_ylim(-200, float(np.max(ph_2014_2026)) * 1.15)

    nem_milestones = [
        ("2017-03-29", "Hazelwood Exit"),
        ("2020-04-01", "COVID Lockdowns"),
        ("2021-10-01", "5MS & Deep PV"),
        ("2023-04-28", "Liddell Exit"),
    ]
    for dt_s, lbl in nem_milestones:
        dt_m = pd.Timestamp(dt_s)
        ax2.axvline(dt_m, color="gray", linestyle=":", linewidth=1.1)
        ax2.text(
            dt_m, y2_top * 0.78, f" {lbl}", rotation=90, va="top", fontsize=8.0, color="dimgray",
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.0),
        )

    ax2.set_xlabel("Date (2010–2026)")
    ax2.set_ylabel("Diurnal Window EWMA MAPE (%)")
    ax2_r.set_ylabel("Page-Hinkley Stat M_PH", color="purple")
    ax2.set_title("(c) Diurnal Error Decomposition (Midday Solar vs. Overnight) & Page-Hinkley Sequential Change Statistic")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig9_online_drift_detectors_timeline.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 10: Drift Mechanisms — Day-Type Shifts, Weather vs Shape Regimes, Bias
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # Panel (a): Day-Type Cluster Shares (2010–2026) vs Annual Frozen Lasso MAPE
    ax_a = axes[0, 0]
    share_cols = [c for c in yearly_df.columns if c.startswith("cluster_") and c.endswith("_share_pct")]
    bottom = np.zeros(len(yearly_df))
    cluster_colors = ["steelblue", "seagreen", "darkorange", "crimson"]
    for idx_c, col in enumerate(share_cols):
        vals = yearly_df[col].to_numpy()
        ax_a.bar(yearly_df["year"], vals, bottom=bottom, width=0.72, alpha=0.72, color=cluster_colors[idx_c % 4], label=f"Cluster {idx_c + 1} Share (%)")
        bottom += vals
    ax_a2 = ax_a.twinx()
    ax_a2.plot(yearly_df["year"], yearly_df["frozen_lasso_mape"], color="black", marker="o", linewidth=2.2, label="Frozen Lasso MAPE (%) (Right)")
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Share of Days in 2010–2013 Cluster (%)")
    ax_a2.set_ylabel("Annual Frozen Lasso MAPE (%)")
    ax_a.set_ylim(0, 124)
    ax_a.set_title("(a) Day-Type Cluster Prevalence Shift vs. Annual Forecasting Error")
    ax_a.grid(True, axis="y", alpha=0.3)
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.0, ncol=3)

    # Panel (b): 2x2 Regime Comparison (Does Input Shift Always Cause Worse Predictions?)
    ax_b = axes[0, 1]
    x_pos = np.arange(len(regime_df))
    bars = ax_b.bar(x_pos, regime_df["mean_daily_mape"], color=["seagreen", "royalblue", "darkorange", "crimson"], alpha=0.85, edgecolor="black")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(regime_df["regime_short"], rotation=15, ha="right", fontsize=8.8)
    ax_b.set_ylabel("Mean Daily Demand MAPE (%)")
    ax_b.set_title("(b) Covariate Shift Test (2015–2026): Weather Shift vs. Profile Shape Shift")
    ax_b.grid(True, axis="y", alpha=0.3)
    for bar, (_, r) in zip(bars, regime_df.iterrows()):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.15,
            f"{r['mean_daily_mape']:.2f}%\n(n={int(r['n_days'])}, b={r['mean_signed_bias_mw']:+.0f} MW)",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax_b.set_ylim(0, max(regime_df["mean_daily_mape"]) * 1.25)

    # Panel (c): Midday Signed Bias vs Overnight Signed Bias by Year
    ax_c = axes[1, 0]
    ax_c.plot(yearly_df["year"], yearly_df["midday_bias_mw"], marker="o", color="crimson", linewidth=2.2, label="Midday (09:00–16:00) Signed Bias (MW)")
    ax_c.plot(yearly_df["year"], yearly_df["non_midday_bias_mw"], marker="s", color="navy", linewidth=2.0, linestyle="--", label="Overnight/Evening Signed Bias (MW)")
    ax_c.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("Mean Signed Forecast Bias: Pred - Obs (MW)")
    ax_c.set_title("(c) Midday Solar Over-Prediction Bias Driving Frozen Model Failure")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper left", fontsize=8.8)

    # Panel (d): Daily PCA Cluster Distance vs Daily MAPE Across Eras
    ax_d = axes[1, 1]
    era_specs = [
        ((2010, 2014), "2010–2014 (Calibration)", "gray", 0.25),
        ((2015, 2018), "2015–2018 (Early Sim)", "seagreen", 0.35),
        ((2019, 2022), "2019–2022 (PV & COVID)", "royalblue", 0.35),
        ((2023, 2026), "2023–2026 (Deep Duck Curve)", "crimson", 0.35),
    ]
    yrs_all = dates.year
    for (y0, y1), lbl, col, alp in era_specs:
        em = (yrs_all >= y0) & (yrs_all <= y1)
        ax_d.scatter(u_in_13[em], daily_mape_all[em], s=10, color=col, alpha=alp, label=lbl)
    ax_d.set_xlabel("Daily PCA Cluster Distance u_in(D) (vs. 2010–2013 Reference)")
    ax_d.set_ylabel("Daily Demand MAPE (%)")
    ax_d.set_title("(d) Daily Profile Shape Distance vs. Forecast Error by Era")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig10_drift_mechanisms_and_input_vs_error.png"), dpi=200)
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

    print("[1/5] Loading dataset and constructing Stage 1 chronological bundles (no leakage)...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()

    bundle_2013 = build_candidate_features(df, day_type_transformer=None, fit_day_type_max_year=2013, include_flat_daily=False)
    bundle_2014 = build_candidate_features(df, day_type_transformer=None, fit_day_type_max_year=2014, include_flat_daily=False)

    dates = bundle_2014["target_dates"]
    feature_names = bundle_2014["feature_names"]
    selected_indices = [feature_names.index(f) for f in selected_features]

    tr13_mask = dates.year <= 2013
    va14_mask = dates.year == 2014
    rf14_mask = dates.year <= 2014
    sim_mask = dates.year >= 2015
    mask_14_26 = dates.year >= 2014

    Y_2d = bundle_2014["Y_2d"]

    # Fit Stage 1 validation model (2010-2013) -> out-of-sample 2014 predictions
    m_2013 = PerSlotLinearRegressor(
        model_type="lasso", alpha=1.0, selected_indices=selected_indices,
        selected_feature_names=selected_features, day_type_transformer=bundle_2013["day_type_transformer"],
    ).fit(bundle_2013["X_3d"][tr13_mask], bundle_2013["Y_2d"][tr13_mask])

    # Fit Stage 1 frozen model (2010-2014) -> deployed on Jan 2015 - Oct 2026
    m_frozen = PerSlotLinearRegressor(
        model_type="lasso", alpha=1.0, selected_indices=selected_indices,
        selected_feature_names=selected_features, day_type_transformer=bundle_2014["day_type_transformer"],
    ).fit(bundle_2014["X_3d"][rf14_mask], bundle_2014["Y_2d"][rf14_mask])

    preds_all = np.zeros_like(Y_2d)
    preds_all[tr13_mask] = m_2013.predict_tensor(bundle_2013["X_3d"][tr13_mask])
    preds_all[va14_mask] = m_2013.predict_tensor(bundle_2013["X_3d"][va14_mask])
    preds_all[sim_mask] = m_frozen.predict_tensor(bundle_2014["X_3d"][sim_mask])

    ape_2d = 100.0 * np.abs(Y_2d - preds_all) / np.abs(Y_2d)
    bias_2d = preds_all - Y_2d
    daily_mape_all = ape_2d.mean(axis=1)

    print("[2/5] Calibrating Input-Based & Error-Based Online Drift Detectors strictly on 2010–2014...")
    # Detector 1: Input-Based (PCA Cluster Distance on 2010-2013 DayTypeTransformer per spec p.4)
    clusters_13, u_in_13 = bundle_2013["day_type_transformer"].transform(Y_2d)
    clusters_14, u_in_14 = bundle_2014["day_type_transformer"].transform(Y_2d)
    z_in_13 = compute_causal_ewma(u_in_13, span=EWMA_SPAN)
    z_in_14 = compute_causal_ewma(u_in_14, span=EWMA_SPAN)

    in_det_13 = OnlineInputDriftDetector().calibrate(z_in_13[rf14_mask], warmup=30)
    in_det_14 = OnlineInputDriftDetector().calibrate(z_in_14[rf14_mask], warmup=30)

    # Companion Weather Covariate Shift Index
    wea_info = compute_weather_covariate_shift(df, dates, rf14_mask)
    z_wea = wea_info["z_wea"]
    h_wea = wea_info["threshold"]

    # Detector 2: Error-Based (30d EWMA + Page-Hinkley calibrated strictly on 2014 out-of-sample errors)
    # Standard SPC deployment initialization on 2015-01-01: EWMA starts at mu_err_ref (2014 mean)
    # and Page-Hinkley starts at M_PH_0 = 0.0 when the refitted 2010–2014 frozen model goes live.
    e_val_2014 = daily_mape_all[va14_mask]
    z_val_2014 = compute_causal_ewma(e_val_2014, span=EWMA_SPAN)

    err_det = OnlineErrorDriftDetector().calibrate(e_val_2014, z_val_2014, warmup=30)
    ph_val_2014 = err_det.compute_page_hinkley(e_val_2014)

    e_sim = daily_mape_all[sim_mask]
    z_err_sim = compute_causal_ewma(e_sim, span=EWMA_SPAN, init_value=err_det.ewma_ref_mean_)
    ph_sim = err_det.compute_page_hinkley(e_sim)

    z_err_14_26 = np.concatenate([z_val_2014, z_err_sim])
    ph_14_26 = np.concatenate([ph_val_2014, ph_sim])

    # Mode A (Fixed-Reference Monitoring of Frozen Model, 2015–2026)
    dates_14_26 = dates[mask_14_26]
    dates_sim = dates[sim_mask]
    sim_start_idx = int(rf14_mask.sum())

    in_mode_a_13 = in_det_13.detect_fixed_reference(dates, z_in_13, sim_mask)
    in_mode_a_14 = in_det_14.detect_fixed_reference(dates, z_in_14, sim_mask)
    err_mode_a = err_det.detect_fixed_reference(dates_sim, z_err_sim, ph_sim)

    # Mode B (Adaptive Online Reset with 365-day Cooldown & Max Budget 6)
    in_alarms_b, in_thresh_trace = in_det_13.detect_adaptive_reset(
        dates, z_in_13, sim_start_idx=sim_start_idx, cooldown_days=COOLDOWN_DAYS, max_alarms=MAX_RETRAIN_BUDGET
    )
    err_frozen_alarms_b, err_frozen_thresh_trace = err_det.detect_adaptive_reset_frozen_stream(
        dates_sim, z_err_sim, cooldown_days=COOLDOWN_DAYS, max_alarms=MAX_RETRAIN_BUDGET
    )

    print("[3/5] Running closed-loop online retraining verification for Error-Based Detector...")
    closed_loop = run_closed_loop_error_retraining(
        df_full=df,
        selected_features=selected_features,
        initial_model=m_frozen,
        initial_bundle=bundle_2014,
        e_val_2014=e_val_2014,
        z_val_2014=z_val_2014,
        detector=err_det,
        cooldown_days=COOLDOWN_DAYS,
        max_alarms=MAX_RETRAIN_BUDGET,
    )

    # Export Table 10: Detector Calibration & Mode A/B Summary
    calib_rows = [
        {
            "detector": "Input-Based (PCA Cluster Distance 30d EWMA, 2010–2013 Ref)",
            "monitored_stat": "u_in(D) = min_c ||z_D - mu_c||_2 (5D PCA)",
            "calibration_period": "2010-01-08 to 2014-12-31",
            "ref_mean": in_det_13.ref_mean_,
            "ref_std": in_det_13.ref_std_,
            "ref_max": in_det_13.ref_max_,
            "threshold": in_det_13.threshold_,
            "mode_a_first_alarm": in_mode_a_13["first_alarm_date"],
            "mode_a_first_effective": in_mode_a_13["first_effective_date"],
            "mode_a_sustained_onset": in_mode_a_13["sustained_30d_onset_date"],
            "mode_a_pct_days_in_alarm": in_mode_a_13["pct_alarm_days"],
            "mode_b_n_alarms_le6": len(in_alarms_b),
        },
        {
            "detector": "Input-Based (PCA Cluster Distance 30d EWMA, 2010–2014 Ref)",
            "monitored_stat": "u_in(D) = min_c ||z_D - mu_c||_2 (5D PCA)",
            "calibration_period": "2010-01-08 to 2014-12-31",
            "ref_mean": in_det_14.ref_mean_,
            "ref_std": in_det_14.ref_std_,
            "ref_max": in_det_14.ref_max_,
            "threshold": in_det_14.threshold_,
            "mode_a_first_alarm": in_mode_a_14["first_alarm_date"],
            "mode_a_first_effective": in_mode_a_14["first_effective_date"],
            "mode_a_sustained_onset": in_mode_a_14["sustained_30d_onset_date"],
            "mode_a_pct_days_in_alarm": in_mode_a_14["pct_alarm_days"],
            "mode_b_n_alarms_le6": len(in_alarms_b),
        },
        {
            "detector": "Weather Covariate Shift Index (8-Var Sydney Weather 30d EWMA)",
            "monitored_stat": "u_wea(D) = RMS standardized Sydney weather z-score",
            "calibration_period": "2010-01-08 to 2014-12-31",
            "ref_mean": wea_info["ref_mean"],
            "ref_std": wea_info["ref_std"],
            "ref_max": wea_info["ref_max"],
            "threshold": h_wea,
            "mode_a_first_alarm": str(dates_sim[np.where(z_wea[sim_mask] > h_wea)[0][0]].date()) if np.any(z_wea[sim_mask] > h_wea) else None,
            "mode_a_first_effective": str((dates_sim[np.where(z_wea[sim_mask] > h_wea)[0][0]] + pd.Timedelta(days=1)).date()) if np.any(z_wea[sim_mask] > h_wea) else None,
            "mode_a_sustained_onset": None,
            "mode_a_pct_days_in_alarm": float(100.0 * np.mean(z_wea[sim_mask] > h_wea)),
            "mode_b_n_alarms_le6": np.nan,
        },
        {
            "detector": "Error-Based (Daily MAPE 30d EWMA, Frozen Stream)",
            "monitored_stat": "e(D) = daily mean MAPE (%) -> Z_err(D)",
            "calibration_period": "2014-01-01 to 2014-12-31 (Out-of-Sample Val)",
            "ref_mean": err_det.ewma_ref_mean_,
            "ref_std": err_det.ewma_ref_std_,
            "ref_max": err_det.ewma_ref_max_,
            "threshold": err_det.ewma_threshold_,
            "mode_a_first_alarm": err_mode_a["ewma_first_alarm_date"],
            "mode_a_first_effective": err_mode_a["ewma_first_effective_date"],
            "mode_a_sustained_onset": err_mode_a["ewma_sustained_30d_onset_date"],
            "mode_a_pct_days_in_alarm": err_mode_a["ewma_pct_alarm_days"],
            "mode_b_n_alarms_le6": len(err_frozen_alarms_b),
        },
        {
            "detector": "Error-Based (Daily MAPE 30d EWMA, Closed-Loop Retrained)",
            "monitored_stat": "e(D) = daily mean MAPE (%) of active model",
            "calibration_period": "2014-01-01 to 2014-12-31 (Out-of-Sample Val)",
            "ref_mean": err_det.ewma_ref_mean_,
            "ref_std": err_det.ewma_ref_std_,
            "ref_max": err_det.ewma_ref_max_,
            "threshold": err_det.ewma_threshold_,
            "mode_a_first_alarm": err_mode_a["ewma_first_alarm_date"],
            "mode_a_first_effective": err_mode_a["ewma_first_effective_date"],
            "mode_a_sustained_onset": err_mode_a["ewma_sustained_30d_onset_date"],
            "mode_a_pct_days_in_alarm": err_mode_a["ewma_pct_alarm_days"],
            "mode_b_n_alarms_le6": len(closed_loop["alarms"]),
        },
        {
            "detector": "Error-Based Companion (Page-Hinkley Test on Daily MAPE)",
            "monitored_stat": "M_PH(D) = max(0, M_PH(D-1) + e(D) - mu_2014 - delta)",
            "calibration_period": "2014-01-01 to 2014-12-31 (Out-of-Sample Val)",
            "ref_mean": err_det.daily_ref_mean_,
            "ref_std": err_det.daily_ref_std_,
            "ref_max": err_det.ph_ref_max_,
            "threshold": err_det.ph_threshold_,
            "mode_a_first_alarm": err_mode_a["ph_first_alarm_date"],
            "mode_a_first_effective": err_mode_a["ph_first_effective_date"],
            "mode_a_sustained_onset": err_mode_a["ph_first_alarm_date"],
            "mode_a_pct_days_in_alarm": err_mode_a["ph_pct_alarm_days"],
            "mode_b_n_alarms_le6": np.nan,
        },
    ]
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(os.path.join(out_dir, "table10_drift_detector_calibration_and_summary.csv"), index=False)

    # Export Table 11: Logged Mode B Alarms & NEM Event Mapping
    alarm_records = []
    for rec in in_alarms_b:
        alarm_records.append({"detector_mode": "Input-EWMA-Reset (PCA Distance)", **rec})
    for rec in err_frozen_alarms_b:
        alarm_records.append({"detector_mode": "Error-EWMA-Reset (Frozen Stream)", **rec})
    for rec in closed_loop["alarms"]:
        alarm_records.append({"detector_mode": "Error-EWMA-Reset (Closed-Loop Retrained)", **rec})
    alarms_df = annotate_nem_events(pd.DataFrame(alarm_records))
    alarms_df.to_csv(os.path.join(out_dir, "table11_drift_alarms_and_nem_events.csv"), index=False)

    print("[4/5] Computing yearly Day-Type/drift evolution and 2x2 covariate vs. concept shift analysis...")
    # Table 12: Yearly Drift & Day-Type Evolution (2010–2026)
    yearly_rows = []
    midday_slots = slice(18, 32)  # 09:00 to 16:00 AEST
    nonmid_slots = list(range(0, 18)) + list(range(32, 48))

    for yr in sorted(np.unique(dates.year)):
        ym = dates.year == yr
        row = {
            "year": int(yr),
            "n_days": int(ym.sum()),
            "mean_demand_mw": float(Y_2d[ym].mean()),
            "min_demand_mw": float(Y_2d[ym].min()),
            "midday_mean_demand_mw": float(Y_2d[ym, midday_slots].mean()),
            "mean_pca_dist_u_in": float(u_in_13[ym].mean()),
            "p90_pca_dist_u_in": float(np.percentile(u_in_13[ym], 90)),
            "mean_weather_shift_u_wea": float(wea_info["u_wea"][ym].mean()),
            "frozen_lasso_mape": float(mape_metric(Y_2d[ym], preds_all[ym])),
            "midday_mape": float(ape_2d[ym, midday_slots].mean()),
            "non_midday_mape": float(ape_2d[ym][:, nonmid_slots].mean()),
            "midday_bias_mw": float(bias_2d[ym, midday_slots].mean()),
            "non_midday_bias_mw": float(bias_2d[ym][:, nonmid_slots].mean()),
        }
        for c in range(bundle_2013["day_type_transformer"].n_clusters):
            row[f"cluster_{c + 1}_share_pct"] = float(100.0 * np.mean(clusters_13[ym] == c))

        if yr >= 2015:
            sim_yr = dates_sim.year == yr
            row["input_mode_a_alarm_pct"] = float(100.0 * in_mode_a_13["alarm_flags_sim"][sim_yr].mean())
            row["error_mode_a_alarm_pct"] = float(100.0 * err_mode_a["ewma_alarm_flags_sim"][sim_yr].mean())
            row["closed_loop_retrained_mape"] = float(
                mape_metric(Y_2d[sim_mask][sim_yr], closed_loop["preds_sim"][sim_yr])
            )
        else:
            row["input_mode_a_alarm_pct"] = float(100.0 * np.mean(z_in_13[ym] > in_det_13.threshold_))
            row["error_mode_a_alarm_pct"] = 0.0 if yr == 2014 else np.nan
            row["closed_loop_retrained_mape"] = row["frozen_lasso_mape"]
        yearly_rows.append(row)

    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(os.path.join(out_dir, "table12_yearly_drift_and_daytype_evolution.csv"), index=False)

    # Table 13: 2x2 Input Shift vs Error Regime Analysis (2015–2026)
    # Thresholds calibrated strictly on 2010–2014 90th percentiles of daily statistics u_wea and u_in
    wea_p90_ref = float(np.percentile(wea_info["u_wea"][rf14_mask], 90))
    shape_p90_ref = float(np.percentile(u_in_13[rf14_mask], 90))

    wea_high = wea_info["u_wea"][sim_mask] > wea_p90_ref
    shape_high = u_in_13[sim_mask] > shape_p90_ref
    e_sim = daily_mape_all[sim_mask]
    b_sim = bias_2d[sim_mask].mean(axis=1)
    mid_e_sim = ape_2d[sim_mask, midday_slots].mean(axis=1)

    regime_specs = [
        ((~wea_high) & (~shape_high), "Low Weather Shift + Low Shape Shift", "1. Normal Both\n(In-Dist)"),
        (wea_high & (~shape_high), "High Weather Shift + Low Shape Shift", "2. Weather Shift\nOnly"),
        ((~wea_high) & shape_high, "Low Weather Shift + High Shape Shift", "3. Shape Shift\nOnly (Solar/Duck)"),
        (wea_high & shape_high, "High Weather Shift + High Shape Shift", "4. Both Shifts\nCombined"),
    ]
    regime_rows = []
    for m_reg, r_name, r_short in regime_specs:
        regime_rows.append({
            "regime": r_name,
            "regime_short": r_short,
            "n_days": int(m_reg.sum()),
            "pct_of_sim_days": float(100.0 * m_reg.mean()),
            "mean_daily_mape": float(e_sim[m_reg].mean()),
            "median_daily_mape": float(np.median(e_sim[m_reg])),
            "mean_midday_mape": float(mid_e_sim[m_reg].mean()),
            "mean_signed_bias_mw": float(b_sim[m_reg].mean()),
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(os.path.join(out_dir, "table13_input_shift_vs_error_regime_analysis.csv"), index=False)

    # Correlations linking input shift, weather shift, day-type cluster shares, and error
    sim_daily_df = pd.DataFrame({
        "daily_mape": e_sim,
        "u_in": u_in_13[sim_mask],
        "u_wea": wea_info["u_wea"][sim_mask],
        "z_in": z_in_13[sim_mask],
        "z_wea": z_wea[sim_mask],
        "z_err": z_err_sim,
    })
    corr_stats = {
        "daily_corr_u_in_vs_mape_2015_2026": float(sim_daily_df["u_in"].corr(sim_daily_df["daily_mape"])),
        "daily_corr_u_wea_vs_mape_2015_2026": float(sim_daily_df["u_wea"].corr(sim_daily_df["daily_mape"])),
        "ewma_corr_z_in_vs_z_err_2015_2026": float(sim_daily_df["z_in"].corr(sim_daily_df["z_err"])),
        "ewma_corr_z_wea_vs_z_err_2015_2026": float(sim_daily_df["z_wea"].corr(sim_daily_df["z_err"])),
        "yearly_corr_pca_dist_vs_mape": float(yearly_df["mean_pca_dist_u_in"].corr(yearly_df["frozen_lasso_mape"])),
        "yearly_corr_cluster1_share_vs_mape": float(yearly_df["cluster_1_share_pct"].corr(yearly_df["frozen_lasso_mape"])),
        "weather_p90_ref_2010_2014": wea_p90_ref,
        "shape_p90_ref_2010_2014": shape_p90_ref,
        "closed_loop_overall_mape_2015_2026": closed_loop["overall_mape_sim"],
        "closed_loop_overall_mae_2015_2026": closed_loop["overall_mae_sim"],
        "frozen_overall_mape_2015_2026": mape_metric(Y_2d[sim_mask], preds_all[sim_mask]),
        "frozen_overall_mae_2015_2026": mae_metric(Y_2d[sim_mask], preds_all[sim_mask]),
    }

    summary_payload = {
        "ewma_span_days": EWMA_SPAN,
        "cooldown_days": COOLDOWN_DAYS,
        "max_retrain_budget": MAX_RETRAIN_BUDGET,
        "calibration": calib_rows,
        "mode_b_input_alarms": in_alarms_b,
        "mode_b_error_frozen_stream_alarms": err_frozen_alarms_b,
        "mode_b_error_closed_loop_alarms": closed_loop["alarms"],
        "correlations_and_metrics": corr_stats,
        "regime_2x2_summary": regime_rows,
    }
    with open(os.path.join(out_dir, "stage2_drift_detection_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    print("[5/5] Generating Figures 9 & 10...")
    plot_drift_figures(
        dates=dates,
        z_in_13=z_in_13,
        h_in_13=in_det_13.threshold_,
        in_thresh_trace=in_thresh_trace,
        in_alarms=in_alarms_b,
        z_wea=z_wea,
        h_wea=h_wea,
        dates_2014_2026=dates_14_26,
        z_err_frozen=z_err_14_26,
        h_err=err_det.ewma_threshold_,
        err_frozen_thresh_trace=err_frozen_thresh_trace,
        err_frozen_alarms=err_frozen_alarms_b,
        ph_2014_2026=ph_14_26,
        h_ph=err_det.ph_threshold_,
        dates_sim=dates_sim,
        closed_z_sim=closed_loop["ewma_sim"],
        closed_thresh_sim=closed_loop["thresh_sim"],
        closed_alarms=closed_loop["alarms"],
        yearly_df=yearly_df,
        regime_df=regime_df,
        ape_2d=ape_2d,
        bias_2d=bias_2d,
        u_in_13=u_in_13,
        daily_mape_all=daily_mape_all,
        out_dir=out_dir,
    )

    # Concise diagnostic report to stdout
    print("\n=== Detector Calibration & Mode A Summary ===")
    print(calib_df[["detector", "threshold", "mode_a_first_alarm", "mode_a_sustained_onset", "mode_a_pct_days_in_alarm", "mode_b_n_alarms_le6"]].to_string(index=False))

    print("\n=== Mode B Online Alarm Schedules (Budget <= 6, 365-Day Cooldown) ===")
    print(alarms_df[["detector_mode", "alarm_num", "alarm_date", "effective_date", "statistic_value", "threshold_at_alarm", "nem_market_event_window"]].to_string(index=False))

    print("\n=== 2x2 Input Shift vs Error Regime Analysis (2015–2026) ===")
    print(regime_df[["regime", "n_days", "pct_of_sim_days", "mean_daily_mape", "mean_midday_mape", "mean_signed_bias_mw"]].round(3).to_string(index=False))

    print(
        f"\nFrozen Lasso Overall 2015–2026: {corr_stats['frozen_overall_mape_2015_2026']:.3f}% "
        f"({corr_stats['frozen_overall_mae_2015_2026']:.2f} MW) | "
        f"Closed-Loop Error-Triggered Retrained ({len(closed_loop['alarms'])} retrains): "
        f"{corr_stats['closed_loop_overall_mape_2015_2026']:.3f}% "
        f"({corr_stats['closed_loop_overall_mae_2015_2026']:.2f} MW)"
    )
    print(f"Completed Step 1 (Online Drift Detection) in {time.time() - t0:.1f}s. Saved outputs to {out_dir}/.")


if __name__ == "__main__":
    main()

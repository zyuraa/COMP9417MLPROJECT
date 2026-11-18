"""Stage 3 Final Model Evaluation, 4-Year vs. Recency Comparison, and Live-Week Forecast (7–13 Dec 2026).

Implements Step 3 of the COMP9417 requirements:
  1. Compares the teammate's Sliding-4Y (1460d) proposal against Exponential Recency (H=365d),
     their hybrid (Exp. Recency H=365d on Last 4Y), Sliding-2Y/3Y, Expanding, and Frozen Stage 1.
  2. Performs a walk-forward historical analogue of the exact live-week protocol:
     for each recent year Y in {2021, 2022, 2023, 2024, 2025}, fits strictly on data
     <= Y-10-31 and evaluates out-of-sample on Dec 7–13 Y (37–43 days ahead) and Dec 1–31 Y.
  3. Builds 7-day live-week prediction intervals (80% & 95% PIs) for 7–13 Dec 2026 using:
       - Day-block bootstrap (7 whole 48-hh days, B = 5,000) on out-of-sample Nov 15 – Dec 31 (2023–2025)
       - Contiguous 7-day moving-block bootstrap (preserving across-day autocorrelation)
       - 2025–2026 adapted out-of-sample cross-check & empirical Dec 7–13 analogues.
  4. Exports table19–table21, stage3_live_week_summary.json, and fig13 to analysis_outputs/.
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
from stage2_adaptation import (  # noqa: E402
    build_daytype_column_map,
    extract_daily_demand_matrix,
    get_periodic_retrain_indices,
    update_selected_daytype_features,
)
from stage2_drift_detection import MAX_RETRAIN_BUDGET  # noqa: E402


STAGE3_HALFLIFE_DAYS = 365.0
STAGE3_DAYTYPE_WINDOW_DAYS = 1095


def fit_variant_at_day(
    retrain_sample_idx: int,
    target_dates: pd.DatetimeIndex,
    X_sub_buf: np.ndarray,
    Y_2d: np.ndarray,
    nsw_dem_all: np.ndarray,
    idx_valid: np.ndarray,
    daytype_col_specs: List[Tuple[int, str, Optional[int]]],
    selected_features: List[str],
    mode: str,
    window_days: Optional[int] = None,
    halflife_days: Optional[float] = None,
    alpha: float = 1.0,
) -> PerSlotLinearRegressor:
    """Fit DayTypeTransformer + Per-Slot Lasso strictly on data <= target_dates[retrain_sample_idx]."""
    raw_day_idx = int(idx_valid[retrain_sample_idx])
    d_retrain = target_dates[retrain_sample_idx]

    if mode == "expanding":
        dt_start = 0
        tr_start = 0
        sample_weight = None
    elif mode == "sliding":
        assert window_days is not None
        dt_start = max(0, raw_day_idx - window_days + 1)
        tr_start = max(0, retrain_sample_idx - window_days + 1)
        sample_weight = None
    elif mode == "recency":
        assert halflife_days is not None
        dt_start = max(0, raw_day_idx - STAGE3_DAYTYPE_WINDOW_DAYS + 1)
        tr_start = max(0, retrain_sample_idx - window_days + 1) if window_days is not None else 0
        ages_days = (d_retrain - target_dates[tr_start : retrain_sample_idx + 1]).days.to_numpy(dtype=np.float64)
        raw_w = np.power(2.0, -ages_days / float(halflife_days))
        sample_weight = raw_w / np.mean(raw_w)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

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


def run_periodic_variant(
    periodic_indices: np.ndarray,
    initial_model: PerSlotLinearRegressor,
    X_sub_initial: np.ndarray,
    Y_2d: np.ndarray,
    target_dates: pd.DatetimeIndex,
    sim_indices: np.ndarray,
    nsw_dem_all: np.ndarray,
    idx_valid: np.ndarray,
    daytype_col_specs: List[Tuple[int, str, Optional[int]]],
    selected_features: List[str],
    mode: str,
    window_days: Optional[int] = None,
    halflife_days: Optional[float] = None,
) -> np.ndarray:
    """Run 6-retrain Periodic strategy across 2015–2026 for a Stage 3 candidate variant."""
    n_sim = len(sim_indices)
    sim_start = int(sim_indices[0])
    sim_end = int(sim_indices[-1]) + 1
    preds_sim = np.zeros((n_sim, N_PERIODS), dtype=np.float64)
    X_sub_buf = X_sub_initial.copy()
    curr_model = initial_model

    seg_start = sim_start
    for r_idx in sorted(int(x) for x in periodic_indices):
        seg_end = r_idx + 1
        if seg_end > seg_start:
            preds_sim[seg_start - sim_start : seg_end - sim_start] = curr_model.predict_tensor(
                X_sub_buf[seg_start:seg_end]
            )
        curr_model = fit_variant_at_day(
            retrain_sample_idx=r_idx,
            target_dates=target_dates,
            X_sub_buf=X_sub_buf,
            Y_2d=Y_2d,
            nsw_dem_all=nsw_dem_all,
            idx_valid=idx_valid,
            daytype_col_specs=daytype_col_specs,
            selected_features=selected_features,
            mode=mode,
            window_days=window_days,
            halflife_days=halflife_days,
        )
        seg_start = seg_end

    if sim_end > seg_start:
        preds_sim[seg_start - sim_start : sim_end - sim_start] = curr_model.predict_tensor(
            X_sub_buf[seg_start:sim_end]
        )
    return preds_sim


def score_demand_marks(improvement_pct: float) -> int:
    """Official COMP9417 live-evaluation rubric for Demand MAPE improvement I (%)."""
    if improvement_pct >= 40.0:
        return 5
    if improvement_pct >= 30.0:
        return 4
    if improvement_pct >= 20.0:
        return 3
    if improvement_pct >= 10.0:
        return 2
    if improvement_pct > 0.0:
        return 1
    return 0


def bootstrap_7day_week_intervals(
    ape_model_days: np.ndarray,
    ae_model_days: np.ndarray,
    ape_b1_days: np.ndarray,
    block_list_model_ape: Optional[List[np.ndarray]] = None,
    block_list_model_ae: Optional[List[np.ndarray]] = None,
    block_list_b1_ape: Optional[List[np.ndarray]] = None,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[dict, np.ndarray]:
    """Compute 7-day prediction intervals via day-block bootstrap (and optional contiguous 7-day blocks)."""
    rng = np.random.default_rng(seed)
    n_days = len(ape_model_days)

    # 1. Day-Block Bootstrap drawing 7 whole 48-hh days
    draw_idx = rng.integers(0, n_days, size=(n_boot, 7))
    daily_mape_m = ape_model_days.mean(axis=1)
    daily_mae_m = ae_model_days.mean(axis=1)
    daily_mape_b1 = ape_b1_days.mean(axis=1)

    boot_mape_m = daily_mape_m[draw_idx].mean(axis=1)
    boot_mae_m = daily_mae_m[draw_idx].mean(axis=1)
    boot_mape_b1 = daily_mape_b1[draw_idx].mean(axis=1)
    boot_imp = 100.0 * (1.0 - boot_mape_m / boot_mape_b1)

    res_dayblock = {
        "n_pool_days": int(n_days),
        "mape_mean": float(np.mean(boot_mape_m)),
        "mape_median": float(np.median(boot_mape_m)),
        "mape_pi80_low": float(np.percentile(boot_mape_m, 10.0)),
        "mape_pi80_high": float(np.percentile(boot_mape_m, 90.0)),
        "mape_pi95_low": float(np.percentile(boot_mape_m, 2.5)),
        "mape_pi95_high": float(np.percentile(boot_mape_m, 97.5)),
        "mae_mean": float(np.mean(boot_mae_m)),
        "mae_pi80_low": float(np.percentile(boot_mae_m, 10.0)),
        "mae_pi80_high": float(np.percentile(boot_mae_m, 90.0)),
        "mae_pi95_low": float(np.percentile(boot_mae_m, 2.5)),
        "mae_pi95_high": float(np.percentile(boot_mae_m, 97.5)),
        "b1_mape_mean": float(np.mean(boot_mape_b1)),
        "b1_mape_pi80_low": float(np.percentile(boot_mape_b1, 10.0)),
        "b1_mape_pi80_high": float(np.percentile(boot_mape_b1, 90.0)),
        "b1_mape_pi95_low": float(np.percentile(boot_mape_b1, 2.5)),
        "b1_mape_pi95_high": float(np.percentile(boot_mape_b1, 97.5)),
        "improv_pct_mean": float(np.mean(boot_imp)),
        "improv_pct_pi80_low": float(np.percentile(boot_imp, 10.0)),
        "improv_pct_pi80_high": float(np.percentile(boot_imp, 90.0)),
        "improv_pct_pi95_low": float(np.percentile(boot_imp, 2.5)),
        "improv_pct_pi95_high": float(np.percentile(boot_imp, 97.5)),
        "prob_improv_ge_40_pct": float(100.0 * np.mean(boot_imp >= 40.0)),
        "prob_beat_baseline_pct": float(100.0 * np.mean(boot_imp > 0.0)),
    }

    res_dict = {"day_block_7d": res_dayblock}
    boot_dist = boot_mape_m

    if block_list_model_ape is not None and len(block_list_model_ape) > 0:
        n_blk = len(block_list_model_ape)
        blk_mape_m = np.asarray([b.mean() for b in block_list_model_ape], dtype=np.float64)
        blk_mae_m = np.asarray([b.mean() for b in block_list_model_ae], dtype=np.float64)
        blk_mape_b1 = np.asarray([b.mean() for b in block_list_b1_ape], dtype=np.float64)

        b_idx = rng.integers(0, n_blk, size=n_boot)
        c_mape_m = blk_mape_m[b_idx]
        c_mae_m = blk_mae_m[b_idx]
        c_mape_b1 = blk_mape_b1[b_idx]
        c_imp = 100.0 * (1.0 - c_mape_m / c_mape_b1)

        res_dict["contiguous_7d_block"] = {
            "n_pool_days": int(n_blk),
            "mape_mean": float(np.mean(c_mape_m)),
            "mape_median": float(np.median(c_mape_m)),
            "mape_pi80_low": float(np.percentile(c_mape_m, 10.0)),
            "mape_pi80_high": float(np.percentile(c_mape_m, 90.0)),
            "mape_pi95_low": float(np.percentile(c_mape_m, 2.5)),
            "mape_pi95_high": float(np.percentile(c_mape_m, 97.5)),
            "mae_mean": float(np.mean(c_mae_m)),
            "mae_pi80_low": float(np.percentile(c_mae_m, 10.0)),
            "mae_pi80_high": float(np.percentile(c_mae_m, 90.0)),
            "mae_pi95_low": float(np.percentile(c_mae_m, 2.5)),
            "mae_pi95_high": float(np.percentile(c_mae_m, 97.5)),
            "b1_mape_mean": float(np.mean(c_mape_b1)),
            "b1_mape_pi80_low": float(np.percentile(c_mape_b1, 10.0)),
            "b1_mape_pi80_high": float(np.percentile(c_mape_b1, 90.0)),
            "b1_mape_pi95_low": float(np.percentile(c_mape_b1, 2.5)),
            "b1_mape_pi95_high": float(np.percentile(c_mape_b1, 97.5)),
            "improv_pct_mean": float(np.mean(c_imp)),
            "improv_pct_pi80_low": float(np.percentile(c_imp, 10.0)),
            "improv_pct_pi80_high": float(np.percentile(c_imp, 90.0)),
            "improv_pct_pi95_low": float(np.percentile(c_imp, 2.5)),
            "improv_pct_pi95_high": float(np.percentile(c_imp, 97.5)),
            "prob_improv_ge_40_pct": float(100.0 * np.mean(c_imp >= 40.0)),
            "prob_beat_baseline_pct": float(100.0 * np.mean(c_imp > 0.0)),
        }
        boot_dist = c_mape_m

    return res_dict, boot_dist


def plot_stage3_figure(
    comp_df: pd.DataFrame,
    analogue_df: pd.DataFrame,
    boot_dist_contig: np.ndarray,
    contig_summary: dict,
    dec25_week_dates: pd.DatetimeIndex,
    dec25_obs: np.ndarray,
    dec25_stage3_pred: np.ndarray,
    dec25_frozen_pred: np.ndarray,
    out_dir: str,
) -> None:
    """Generate Figure 13: Stage 3 4Y vs Recency Comparison, Walk-Forward Dec 7-13 & Live-Week PI."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    # Panel (a): 4-Year Window vs. Exponential Recency Weighting Comparison
    ax_a = axes[0, 0]
    order_keys = [
        "Expanding (Uniform, All Data)",
        "Sliding-4Y (Uniform, 1460d) [Teammate 4Y]",
        "Sliding-2Y (Uniform, 730d)",
        "Exp. Recency (H=548d, All Data)",
        "Exp. Recency (H=365d, Last 4Y=1460d)",
        "Exp. Recency (H=365d, All Data) [Stage 3 Selected]",
    ]
    short_lbls = [
        "Expanding\n(Uniform)",
        "Sliding-4Y\n(Teammate)",
        "Sliding-2Y\n(Uniform)",
        "Recency\nH=548d",
        "Recency H=365d\n+ Last 4Y",
        "Recency H=365d\n(Stage 3)",
    ]
    sub_a = comp_df.set_index("variant").loc[order_keys]
    x_pos = np.arange(len(order_keys))
    w_bar = 0.36
    b1 = ax_a.bar(x_pos - w_bar / 2, sub_a["mape_2025_2026"], width=w_bar, color="royalblue", alpha=0.88, edgecolor="black", label="2025–2026 Holdout MAPE (%)")
    b2 = ax_a.bar(x_pos + w_bar / 2, sub_a["walkfwd_dec7_13_2022_2025_mape"], width=w_bar, color="darkorange", alpha=0.88, edgecolor="black", label="Walk-Fwd Dec 7–13 (2022–2025) MAPE (%)")
    for bars in (b1, b2):
        for bar in bars:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=7.8)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(short_lbls, fontsize=8.4)
    ax_a.set_ylabel("Demand MAPE (%)")
    ax_a.set_ylim(0, max(sub_a["mape_2025_2026"].max(), sub_a["walkfwd_dec7_13_2022_2025_mape"].max()) * 1.24)
    ax_a.set_title("(a) Teammate 4-Year Cutoff vs. Exponential Recency Weighting (H=365d)")
    ax_a.grid(True, axis="y", alpha=0.3)
    ax_a.legend(loc="upper right", fontsize=8.3)

    # Panel (b): Walk-Forward Dec 7–13 Historical Analogues (Trained strictly <= Oct 31 of Year Y)
    ax_b = axes[0, 1]
    yrs = analogue_df["year"].to_numpy()
    xb = np.arange(len(yrs))
    wb = 0.25
    ax_b.bar(xb - wb, analogue_df["frozen_dec7_13_mape"], width=wb, color="crimson", alpha=0.82, edgecolor="black", label="Frozen Stage 1 (2010–2014)")
    ax_b.bar(xb, analogue_df["b1_dec7_13_mape"], width=wb, color="gray", alpha=0.75, edgecolor="black", label="Baseline 1: Same HH Yesterday")
    bars_s3 = ax_b.bar(xb + wb, analogue_df["stage3_dec7_13_mape"], width=wb, color="navy", alpha=0.90, edgecolor="black", label="Stage 3 (Exp. Recency H=365d <= Oct 31)")
    for bar, (_, r) in zip(bars_s3, analogue_df.iterrows()):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.18,
            f"{r['stage3_dec7_13_mape']:.2f}%\n(I={r['improv_vs_b1_pct']:.0f}%)",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="navy",
        )
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([f"Dec 7–13,\n{int(y)}" for y in yrs], fontsize=9.0)
    ax_b.set_ylabel("7-Day Demand MAPE (%)")
    ax_b.set_ylim(0, max(analogue_df["frozen_dec7_13_mape"].max(), analogue_df["b1_dec7_13_mape"].max()) * 1.18)
    ax_b.set_title("(b) Exact Walk-Forward Analogues: Trained <= Oct 31, Tested on Dec 7–13")
    ax_b.grid(True, axis="y", alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=8.3)

    # Panel (c): Live-Week (7–13 Dec 2026) Predictive MAPE Distribution & Intervals
    ax_c = axes[1, 0]
    ax_c.hist(boot_dist_contig, bins=32, density=True, color="steelblue", alpha=0.65, edgecolor="white", label="7-Day Contiguous Block Bootstrap (B=5,000)")
    ax_c.axvline(contig_summary["mape_mean"], color="navy", linewidth=2.3, label=f"Point Forecast Mean = {contig_summary['mape_mean']:.2f}% (Med {contig_summary['mape_median']:.2f}%)")
    ax_c.axvspan(contig_summary["mape_pi80_low"], contig_summary["mape_pi80_high"], color="cornflowerblue", alpha=0.28, label=f"80% PI: [{contig_summary['mape_pi80_low']:.2f}%, {contig_summary['mape_pi80_high']:.2f}%]")
    ax_c.axvline(contig_summary["mape_pi95_low"], color="darkorange", linestyle="--", linewidth=1.6, label=f"95% PI: [{contig_summary['mape_pi95_low']:.2f}%, {contig_summary['mape_pi95_high']:.2f}%]")
    ax_c.axvline(contig_summary["mape_pi95_high"], color="darkorange", linestyle="--", linewidth=1.6)
    ax_c.axvline(contig_summary["b1_mape_mean"], color="crimson", linestyle=":", linewidth=2.0, label=f"Expected Yesterday Baseline = {contig_summary['b1_mape_mean']:.2f}%")
    ax_c.set_xlabel("Expected 7-Day Demand MAPE (%) on 7–13 Dec 2026")
    ax_c.set_ylabel("Bootstrap Probability Density")
    ax_c.set_title("(c) Live-Week (7–13 Dec 2026) Predictive Distribution & Uncertainty Intervals")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=8.0)

    # Panel (d): Half-Hourly Trajectory on Most Recent Walk-Forward Analogue (7–13 Dec 2025)
    ax_d = axes[1, 1]
    t_axis = pd.date_range(dec25_week_dates[0] + pd.Timedelta("30min"), periods=len(dec25_obs), freq="30min")
    ax_d.plot(t_axis, dec25_obs, color="black", linewidth=2.0, label="Observed NSW Demand (MW)")
    ax_d.plot(t_axis, dec25_stage3_pred, color="navy", linewidth=1.7, label="Stage 3 Forecast (Trained <= 2025-10-31)")
    ax_d.plot(t_axis, dec25_frozen_pred, color="crimson", linestyle="--", linewidth=1.3, alpha=0.80, label="Frozen Stage 1 Forecast (2010–2014)")
    ax_d.set_xlabel("Date (7–13 December 2025, Half-Hourly AEST)")
    ax_d.set_ylabel("NSW Operational Demand (MW)")
    ax_d.set_title("(d) Half-Hourly Forecast Tracking on 7–13 Dec 2025 (37–43d Post-Oct 31 Cutoff)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="lower left", fontsize=8.3)
    ax_d.xaxis.set_major_locator(mdates.DayLocator())
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig13_stage3_final_model_and_live_week_forecast.png"), dpi=200)
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

    print("[1/4] Loading dataset & building base tensor...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    bundle_2014 = build_candidate_features(df, day_type_transformer=None, fit_day_type_max_year=2014, include_flat_daily=False)
    dates = bundle_2014["target_dates"]
    feature_names = bundle_2014["feature_names"]
    selected_indices = [feature_names.index(f) for f in selected_features]
    Y_2d = bundle_2014["Y_2d"]

    X_sub_initial = bundle_2014["X_3d"][:, :, selected_indices].copy()
    nsw_dem_all, idx_valid, _ = extract_daily_demand_matrix(df)
    daytype_col_specs = build_daytype_column_map(selected_features)

    rf14_mask = dates.year <= 2014
    sim_mask = dates.year >= 2015
    sim_indices = np.where(sim_mask)[0]
    dates_sim = dates[sim_mask]
    Y_sim = Y_2d[sim_mask]

    dem_d_idx = feature_names.index("dem_d_t")
    dem_d6_idx = feature_names.index("dem_d6_t")
    b1_all = bundle_2014["X_3d"][:, :, dem_d_idx]
    b7_all = bundle_2014["X_3d"][:, :, dem_d6_idx]

    initial_model = PerSlotLinearRegressor(
        model_type="lasso", alpha=1.0, selected_indices=None,
        selected_feature_names=selected_features, day_type_transformer=bundle_2014["day_type_transformer"],
    ).fit(X_sub_initial[rf14_mask], Y_2d[rf14_mask])
    frozen_preds_sim = initial_model.predict_tensor(X_sub_initial[sim_mask])

    periodic_6_indices = get_periodic_retrain_indices(sim_indices, MAX_RETRAIN_BUDGET)

    # Variants to compare (Teammate's 4Y vs Exponential Recency H=365d vs Hybrid vs Sliding-2Y/3Y)
    variants = [
        ("Frozen Stage 1 (2010–2014)", "frozen", None, None),
        ("Expanding (Uniform, All Data)", "expanding", None, None),
        ("Sliding-4Y (Uniform, 1460d) [Teammate 4Y]", "sliding", 1460, None),
        ("Sliding-3Y (Uniform, 1095d)", "sliding", 1095, None),
        ("Sliding-2Y (Uniform, 730d)", "sliding", 730, None),
        ("Exp. Recency (H=548d, All Data)", "recency", None, 548.0),
        ("Exp. Recency (H=365d, Last 4Y=1460d)", "recency", 1460, 365.0),
        ("Exp. Recency (H=365d, All Data) [Stage 3 Selected]", "recency", None, 365.0),
    ]

    print("[2/4] Running 6-retrain Stage 2 comparison & walk-forward Oct 31 -> Dec 7–13 tests (2021–2025)...")
    stage2_preds: Dict[str, np.ndarray] = {}
    for v_name, mode, w_days, h_days in variants:
        if mode == "frozen":
            stage2_preds[v_name] = frozen_preds_sim
        else:
            stage2_preds[v_name] = run_periodic_variant(
                periodic_6_indices, initial_model, X_sub_initial, Y_2d, dates, sim_indices,
                nsw_dem_all, idx_valid, daytype_col_specs, selected_features, mode, w_days, h_days,
            )

    # Walk-forward Oct 31 cutoff across Y in [2021, 2022, 2023, 2024, 2025]
    analogue_years = [2021, 2022, 2023, 2024, 2025]
    wf_preds_by_variant: Dict[str, Dict[int, np.ndarray]] = {v[0]: {} for v in variants}
    wf_dates_by_year: Dict[int, pd.DatetimeIndex] = {}
    wf_truth_by_year: Dict[int, np.ndarray] = {}
    wf_b1_by_year: Dict[int, np.ndarray] = {}
    wf_b7_by_year: Dict[int, np.ndarray] = {}

    # Also compute exact price-direction baseline reference on Dec 7-13 of each year
    op_days_full = market_day(df.index)
    price_mat_full = df["nsw_price"].to_numpy(dtype=np.float64).reshape(-1, N_PERIODS)
    unique_days_full = pd.DatetimeIndex(sorted(op_days_full.unique()))
    price_by_date = {d: price_mat_full[i] for i, d in enumerate(unique_days_full)}

    analogue_rows = []
    for yr in analogue_years:
        cutoff = pd.Timestamp(f"{yr}-10-31")
        cutoff_idx = int(np.where(dates <= cutoff)[0][-1])
        nov_dec_mask = (dates.year == yr) & (dates.month >= 11)
        nov_dec_dates = dates[nov_dec_mask]
        wf_dates_by_year[yr] = nov_dec_dates
        wf_truth_by_year[yr] = Y_2d[nov_dec_mask]
        wf_b1_by_year[yr] = b1_all[nov_dec_mask]
        wf_b7_by_year[yr] = b7_all[nov_dec_mask]

        for v_name, mode, w_days, h_days in variants:
            if mode == "frozen":
                wf_preds_by_variant[v_name][yr] = initial_model.predict_tensor(X_sub_initial[nov_dec_mask])
            else:
                X_buf = X_sub_initial.copy()
                m_wf = fit_variant_at_day(
                    cutoff_idx, dates, X_buf, Y_2d, nsw_dem_all, idx_valid,
                    daytype_col_specs, selected_features, mode, w_days, h_days,
                )
                wf_preds_by_variant[v_name][yr] = m_wf.predict_tensor(X_buf[nov_dec_mask])

        # Evaluate Dec 7-13 and full Dec 1-31 for this year
        w7_mask = (nov_dec_dates.month == 12) & (nov_dec_dates.day >= 7) & (nov_dec_dates.day <= 13)
        dec_mask = nov_dec_dates.month == 12
        y_w7 = wf_truth_by_year[yr][w7_mask]
        y_dec = wf_truth_by_year[yr][dec_mask]

        s3_name = "Exp. Recency (H=365d, Last 4Y=1460d)"
        all_rec_name = "Exp. Recency (H=365d, All Data) [Stage 3 Selected]"
        s4_name = "Sliding-4Y (Uniform, 1460d) [Teammate 4Y]"
        s2_name = "Sliding-2Y (Uniform, 730d)"
        frz_name = "Frozen Stage 1 (2010–2014)"

        p_s3_w7 = wf_preds_by_variant[s3_name][yr][w7_mask]
        p_s3_dec = wf_preds_by_variant[s3_name][yr][dec_mask]
        p_b1_w7 = wf_b1_by_year[yr][w7_mask]
        p_b7_w7 = wf_b7_by_year[yr][w7_mask]

        m_s3_w7 = mape_metric(y_w7, p_s3_w7)
        m_b1_w7 = mape_metric(y_w7, p_b1_w7)
        imp_w7 = 100.0 * (1.0 - m_s3_w7 / m_b1_w7)

        # Price baseline accuracy & balanced accuracy on Dec 7-13 Y
        c_true_list, c_b1_list = [], []
        for d_w in nov_dec_dates[w7_mask]:
            p_next = price_by_date[d_w]
            p_d = price_by_date[d_w - pd.Timedelta(days=1)]
            p_bar = float(np.mean(p_d))
            c_true_list.append((p_next > p_bar).astype(int))
            c_b1_list.append((p_d > p_bar).astype(int))
        c_true_arr = np.concatenate(c_true_list)
        c_b1_arr = np.concatenate(c_b1_list)
        b1_price_acc = float(100.0 * np.mean(c_true_arr == c_b1_arr))
        recalls = [np.mean(c_b1_arr[c_true_arr == k] == k) for k in np.unique(c_true_arr)]
        b1_price_bal_acc = float(100.0 * np.mean(recalls))

        analogue_rows.append({
            "year": yr,
            "cutoff_date": f"{yr}-10-31",
            "stage3_dec7_13_mape": m_s3_w7,
            "stage3_dec7_13_mae": mae_metric(y_w7, p_s3_w7),
            "stage3_dec7_13_midday_mape": mape_metric(y_w7[:, 18:32], p_s3_w7[:, 18:32]),
            "stage3_dec7_13_bias_mw": float(np.mean(p_s3_w7 - y_w7)),
            "all_data_recency_dec7_13_mape": mape_metric(y_w7, wf_preds_by_variant[all_rec_name][yr][w7_mask]),
            "sliding_4y_dec7_13_mape": mape_metric(y_w7, wf_preds_by_variant[s4_name][yr][w7_mask]),
            "sliding_2y_dec7_13_mape": mape_metric(y_w7, wf_preds_by_variant[s2_name][yr][w7_mask]),
            "frozen_dec7_13_mape": mape_metric(y_w7, wf_preds_by_variant[frz_name][yr][w7_mask]),
            "b1_dec7_13_mape": m_b1_w7,
            "b1_dec7_13_mae": mae_metric(y_w7, p_b1_w7),
            "b7_dec7_13_mape": mape_metric(y_w7, p_b7_w7),
            "improv_vs_b1_pct": imp_w7,
            "demand_live_marks_est": score_demand_marks(imp_w7),
            "stage3_full_dec_mape": mape_metric(y_dec, p_s3_dec),
            "b1_dec7_13_price_acc_pct": b1_price_acc,
            "b1_dec7_13_price_bal_acc_pct": b1_price_bal_acc,
        })

    analogue_df = pd.DataFrame(analogue_rows)
    analogue_df.to_csv(os.path.join(out_dir, "table20_live_week_dec7_13_historical_analogues.csv"), index=False)

    # Export Table 19: 4Y vs Exponential Recency Comparison
    m_sel = dates_sim.year <= 2024
    m_hold = dates_sim.year >= 2025
    comp_rows = []
    for v_name, mode, w_days, h_days in variants:
        p_sim = stage2_preds[v_name]
        w7_22_25 = []
        w7_21_25 = []
        for yr in analogue_years:
            d_yr = wf_dates_by_year[yr]
            w7_m = (d_yr.month == 12) & (d_yr.day >= 7) & (d_yr.day <= 13)
            val_m = mape_metric(wf_truth_by_year[yr][w7_m], wf_preds_by_variant[v_name][yr][w7_m])
            w7_21_25.append(val_m)
            if yr >= 2022:
                w7_22_25.append(val_m)
        comp_rows.append({
            "variant": v_name,
            "mape_2015_2024": mape_metric(Y_sim[m_sel], p_sim[m_sel]),
            "mape_2025_2026": mape_metric(Y_sim[m_hold], p_sim[m_hold]),
            "mae_2025_2026": mae_metric(Y_sim[m_hold], p_sim[m_hold]),
            "overall_mape_2015_2026": mape_metric(Y_sim, p_sim),
            "overall_mae_2015_2026": mae_metric(Y_sim, p_sim),
            "walkfwd_dec7_13_2021_2025_mape": float(np.mean(w7_21_25)),
            "walkfwd_dec7_13_2022_2025_mape": float(np.mean(w7_22_25)),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(os.path.join(out_dir, "table19_stage3_4y_vs_recency_comparison.csv"), index=False)

    print("[3/4] Computing 7-day Live-Week Prediction Intervals (7–13 Dec 2026)...")
    s3_name = "Exp. Recency (H=365d, Last 4Y=1460d)"

    # Pool A: Out-of-sample Nov 15 - Dec 31 across 2023, 2024, 2025 (from Oct 31 cutoff models)
    es_ape_m_list, es_ae_m_list, es_ape_b1_list = [], [], []
    blk_ape_m, blk_ae_m, blk_ape_b1 = [], [], []
    for yr in [2023, 2024, 2025]:
        d_yr = wf_dates_by_year[yr]
        es_m = (d_yr >= pd.Timestamp(f"{yr}-11-15")) & (d_yr <= pd.Timestamp(f"{yr}-12-31"))
        y_es = wf_truth_by_year[yr][es_m]
        p_es = wf_preds_by_variant[s3_name][yr][es_m]
        b1_es = wf_b1_by_year[yr][es_m]

        ape_m = 100.0 * np.abs(y_es - p_es) / np.abs(y_es)
        ae_m = np.abs(y_es - p_es)
        ape_b1 = 100.0 * np.abs(y_es - b1_es) / np.abs(y_es)
        es_ape_m_list.append(ape_m)
        es_ae_m_list.append(ae_m)
        es_ape_b1_list.append(ape_b1)
        for s_i in range(0, len(y_es) - 6):
            blk_ape_m.append(ape_m[s_i : s_i + 7])
            blk_ae_m.append(ae_m[s_i : s_i + 7])
            blk_ape_b1.append(ape_b1[s_i : s_i + 7])

    pool_es_ape_m = np.vstack(es_ape_m_list)
    pool_es_ae_m = np.vstack(es_ae_m_list)
    pool_es_ape_b1 = np.vstack(es_ape_b1_list)

    pi_es, boot_dist_contig = bootstrap_7day_week_intervals(
        pool_es_ape_m, pool_es_ae_m, pool_es_ape_b1,
        blk_ape_m, blk_ae_m, blk_ape_b1, n_boot=5000, seed=42,
    )

    # Pool B: Full 2025–2026 Adapted Out-of-Sample Cross-Check
    p_s3_25_26 = stage2_preds[s3_name][m_hold]
    y_25_26 = Y_sim[m_hold]
    b1_25_26 = b1_all[sim_mask][m_hold]
    ape_25_26 = 100.0 * np.abs(y_25_26 - p_s3_25_26) / np.abs(y_25_26)
    ae_25_26 = np.abs(y_25_26 - p_s3_25_26)
    b1_ape_25_26 = 100.0 * np.abs(y_25_26 - b1_25_26) / np.abs(y_25_26)
    blk_25_m = [ape_25_26[i : i + 7] for i in range(len(y_25_26) - 6)]
    blk_25_ae = [ae_25_26[i : i + 7] for i in range(len(y_25_26) - 6)]
    blk_25_b1 = [b1_ape_25_26[i : i + 7] for i in range(len(y_25_26) - 6)]
    pi_25_26, _ = bootstrap_7day_week_intervals(
        ape_25_26, ae_25_26, b1_ape_25_26, blk_25_m, blk_25_ae, blk_25_b1, n_boot=5000, seed=42,
    )

    interval_rows = [
        {"pool": "Early-Summer Nov 15–Dec 31 (2023–2025 Walk-Fwd)", "method": "7-Day Day-Block Bootstrap", **pi_es["day_block_7d"]},
        {"pool": "Early-Summer Nov 15–Dec 31 (2023–2025 Walk-Fwd)", "method": "7-Day Contiguous Moving-Block Bootstrap", **pi_es["contiguous_7d_block"]},
        {"pool": "Full Holdout 2025–2026 (Periodic H=365d)", "method": "7-Day Day-Block Bootstrap", **pi_25_26["day_block_7d"]},
        {"pool": "Full Holdout 2025–2026 (Periodic H=365d)", "method": "7-Day Contiguous Moving-Block Bootstrap", **pi_25_26["contiguous_7d_block"]},
    ]
    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(os.path.join(out_dir, "table21_live_week_forecast_and_intervals.csv"), index=False)

    # Save JSON summary
    summary_payload = {
        "stage3_strategy": "Hybrid: Last 4Y (1460d) + Exponential Recency Weighting (H=365d) + 3Y DayType Clustering",
        "teammate_4y_vs_recency": comp_rows,
        "historical_dec7_13_analogues": analogue_rows,
        "live_week_prediction_intervals": interval_rows,
    }
    with open(os.path.join(out_dir, "stage3_live_week_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    print("[4/4] Generating Figure 13...")
    d25 = wf_dates_by_year[2025]
    w7_25 = (d25.month == 12) & (d25.day >= 7) & (d25.day <= 13)
    plot_stage3_figure(
        comp_df=comp_df,
        analogue_df=analogue_df,
        boot_dist_contig=boot_dist_contig,
        contig_summary=pi_es["contiguous_7d_block"],
        dec25_week_dates=d25[w7_25],
        dec25_obs=wf_truth_by_year[2025][w7_25].reshape(-1),
        dec25_stage3_pred=wf_preds_by_variant[s3_name][2025][w7_25].reshape(-1),
        dec25_frozen_pred=wf_preds_by_variant["Frozen Stage 1 (2010–2014)"][2025][w7_25].reshape(-1),
        out_dir=out_dir,
    )

    print("\n=== Table 19: Teammate 4-Year Window vs. Exponential Recency Weighting ===")
    print(comp_df.round(3).to_string(index=False))

    print("\n=== Table 20: Walk-Forward Live-Week Analogues (Trained <= Oct 31, Tested Dec 7–13) ===")
    cols_20 = [
        "year", "stage3_dec7_13_mape", "stage3_dec7_13_mae", "sliding_4y_dec7_13_mape",
        "frozen_dec7_13_mape", "b1_dec7_13_mape", "improv_vs_b1_pct", "demand_live_marks_est",
    ]
    print(analogue_df[cols_20].round(3).to_string(index=False))

    print("\n=== Table 21: 7–13 Dec 2026 Live-Week MAPE Forecast & Prediction Intervals ===")
    cols_21 = [
        "pool", "method", "mape_mean", "mape_pi80_low", "mape_pi80_high",
        "mape_pi95_low", "mape_pi95_high", "b1_mape_mean", "improv_pct_mean", "prob_improv_ge_40_pct",
    ]
    print(interval_df[cols_21].round(3).to_string(index=False))
    print(f"\nCompleted Stage 3 live-week evaluation in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

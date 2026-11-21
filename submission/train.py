"""Stage 1 Feature Selection, Hyperparameter Tuning, and Model Training Pipeline (COMP9417).

Usage (from the project kit root):
    python bryan_package/train.py data/nem_weather_halfhourly_2010_2026.csv

Experimental Protocol (Stage 1):
  1. Train on 2010-2013 and validate on 2014 in strict chronological order.
  2. Perform Feature Group Ablation and Hybrid L1 + Residual-F Feature Selection.
  3. Tune hyperparameters across Ridge (L2), Lasso (L1), and ElasticNet (L1+L2),
     and compare against the Flat Daily Multi-Output Ridge model and both baselines.
  4. Once the optimal algorithm, feature subset, and hyperparameters are selected on 2014,
     refit the frozen Stage 1 model on the full 2010-2014 period and save to my_package/artifacts/.
"""
import json
import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
FEATURES_DIR = os.path.join(HERE, "artifacts")
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features import (  # noqa: E402
    DayTypeTransformer,
    FeatureSelector,
    N_PERIODS,
    WINDOW_DAYS,
    build_candidate_features,
    mae_metric,
    mape_metric,
    market_day,
)
from model import PerSlotLinearRegressor, market_day as price_market_day, price_features_from_components  # noqa: E402


def evaluate_hyperparameter_grid(
    X_3d: np.ndarray,
    Y_2d: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    selected_indices: list,
    selected_names: list,
) -> tuple:
    """Evaluate Ridge, Lasso, and ElasticNet across hyperparameter grids on 2014 validation."""
    X_tr, Y_tr = X_3d[train_mask], Y_2d[train_mask]
    X_va, Y_va = X_3d[val_mask], Y_2d[val_mask]

    records = []
    best_preds_by_family = {}
    best_overall = {"val_mape": float("inf")}

    # 1. Ridge (L2) grid
    ridge_alphas = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0, 50.0]
    for alpha in ridge_alphas:
        model = PerSlotLinearRegressor(
            model_type="ridge",
            alpha=alpha,
            selected_indices=selected_indices,
            selected_feature_names=selected_names,
        )
        model.fit(X_tr, Y_tr)
        preds = model.predict_tensor(X_va)
        mape_v = mape_metric(Y_va, preds)
        mae_v = mae_metric(Y_va, preds)
        rec = {
            "model_family": "Ridge (L2)",
            "model_type": "ridge",
            "alpha": alpha,
            "l1_ratio": 0.0,
            "val_mape": mape_v,
            "val_mae": mae_v,
        }
        records.append(rec)
        if "Ridge (L2)" not in best_preds_by_family or mape_v < best_preds_by_family["Ridge (L2)"]["mape"]:
            best_preds_by_family["Ridge (L2)"] = {"mape": mape_v, "mae": mae_v, "preds": preds, "params": rec}
        if mape_v < best_overall["val_mape"]:
            best_overall = {**rec, "preds": preds}

    # 2. Lasso (L1) grid
    lasso_alphas = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 5.00]
    for alpha in lasso_alphas:
        model = PerSlotLinearRegressor(
            model_type="lasso",
            alpha=alpha,
            selected_indices=selected_indices,
            selected_feature_names=selected_names,
        )
        model.fit(X_tr, Y_tr)
        preds = model.predict_tensor(X_va)
        mape_v = mape_metric(Y_va, preds)
        mae_v = mae_metric(Y_va, preds)
        rec = {
            "model_family": "Lasso (L1)",
            "model_type": "lasso",
            "alpha": alpha,
            "l1_ratio": 1.0,
            "val_mape": mape_v,
            "val_mae": mae_v,
        }
        records.append(rec)
        if "Lasso (L1)" not in best_preds_by_family or mape_v < best_preds_by_family["Lasso (L1)"]["mape"]:
            best_preds_by_family["Lasso (L1)"] = {"mape": mape_v, "mae": mae_v, "preds": preds, "params": rec}
        if mape_v < best_overall["val_mape"]:
            best_overall = {**rec, "preds": preds}

    # 3. ElasticNet (L1 + L2) grid
    enet_alphas = [0.005, 0.010, 0.020, 0.050, 0.100]
    enet_l1_ratios = [0.2, 0.5, 0.8, 0.9]
    for alpha in enet_alphas:
        for l1_ratio in enet_l1_ratios:
            model = PerSlotLinearRegressor(
                model_type="elasticnet",
                alpha=alpha,
                l1_ratio=l1_ratio,
                selected_indices=selected_indices,
                selected_feature_names=selected_names,
            )
            model.fit(X_tr, Y_tr)
            preds = model.predict_tensor(X_va)
            mape_v = mape_metric(Y_va, preds)
            mae_v = mae_metric(Y_va, preds)
            rec = {
                "model_family": "ElasticNet (L1+L2)",
                "model_type": "elasticnet",
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "val_mape": mape_v,
                "val_mae": mae_v,
            }
            records.append(rec)
            if "ElasticNet (L1+L2)" not in best_preds_by_family or mape_v < best_preds_by_family["ElasticNet (L1+L2)"]["mape"]:
                best_preds_by_family["ElasticNet (L1+L2)"] = {"mape": mape_v, "mae": mae_v, "preds": preds, "params": rec}
            if mape_v < best_overall["val_mape"]:
                best_overall = {**rec, "preds": preds}

    return pd.DataFrame(records), best_overall, best_preds_by_family


def save_stage1_plots_and_tables(
    out_dir: str,
    selector: FeatureSelector,
    hp_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_preds_by_family: dict,
    flat_preds: np.ndarray,
    b1_preds: np.ndarray,
    b7_preds: np.ndarray,
    Y_va: np.ndarray,
    val_dates: pd.DatetimeIndex,
):
    """Export Stage 1 selection tables and high-resolution report diagnostic plots."""
    os.makedirs(out_dir, exist_ok=True)
    selector.ablation_df_.to_csv(os.path.join(out_dir, "table4_feature_group_ablation.csv"), index=False)
    selector.ranking_df_.to_csv(os.path.join(out_dir, "table5_top_features_ranking.csv"), index=False)
    hp_df.to_csv(os.path.join(out_dir, "table6_hyperparameter_search.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "table7_model_family_comparison_2014.csv"), index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Figure 5: Feature Selection & Ablation Diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel A: Progressive Group Addition
    prog_df = selector.ablation_df_[selector.ablation_df_["mode"] == "progressive_add"].copy()
    group_labels = [
        "+ NSW Lags (8)",
        "+ Rolling Stats (17)",
        "+ Syd Weather (31)",
        "+ Weather Deltas (40)",
        "+ Calendar (56)",
        "+ Regime Inter. (62)",
        "+ Interstate (70)",
        "+ Day-Type (79)",
    ]
    if len(group_labels) != len(prog_df):
        group_labels = [f"+ {g} ({n})" for g, n in zip(prog_df["group"], prog_df["n_features"])]
    y_pos = np.arange(len(prog_df))
    bars = axes[0].barh(y_pos, prog_df["val_mape"], color="steelblue", edgecolor="navy", alpha=0.85)
    axes[0].axvline(4.935, color="crimson", linestyle="--", linewidth=1.5, label="Yesterday Baseline (4.93%)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(group_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("2014 Validation Demand MAPE (%)")
    axes[0].set_title("(a) Progressive Feature Group Addition")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].legend(loc="lower right")
    for bar, val in zip(bars, prog_df["val_mape"]):
        axes[0].text(val + 0.06, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}%", va="center", fontsize=8.5)
    axes[0].set_xlim(0, 5.6)

    # Panel B: Top-K Feature Selection Curve
    k_df = selector.k_curve_df_
    axes[1].plot(k_df["k"], k_df["val_mape"], marker="o", linewidth=2.0, color="darkgreen", label="Per-Slot Ridge Val MAPE")
    best_k = selector.best_k_
    best_k_mape = float(k_df.loc[k_df["k"] == best_k, "val_mape"].iloc[0])
    axes[1].scatter([best_k], [best_k_mape], color="crimson", s=90, zorder=5, label=f"Selected k* = {best_k} ({best_k_mape:.2f}%)")
    axes[1].axvline(best_k, color="crimson", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Number of Top-Ranked Features Retained (k)")
    axes[1].set_ylabel("2014 Validation Demand MAPE (%)")
    axes[1].set_title("(b) Validation Error vs. Feature Subset Size k")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel C: Top 18 Features by Composite L1 + Residual-F Score
    top_df = selector.ranking_df_.head(18).iloc[::-1]
    axes[2].barh(top_df["feature_name"], top_df["l1_importance_mw"], color="teal", alpha=0.85)
    axes[2].set_xlabel("Mean Standardised |L1 Coefficient| Across 48 Slots (MW)")
    axes[2].set_title("(c) Top 18 Features by Standardised L1 Importance")
    axes[2].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_feature_selection_and_ablation.png"), dpi=200)
    plt.close(fig)

    # Figure 6: Hyperparameter Tuning & 2014 Validation Performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.0))

    # Panel A: Hyperparameter curves for Ridge, Lasso, ElasticNet
    r_sub = hp_df[hp_df["model_type"] == "ridge"]
    l_sub = hp_df[hp_df["model_type"] == "lasso"]
    e_sub = hp_df[(hp_df["model_type"] == "elasticnet") & (hp_df["l1_ratio"] == 0.8)]
    axes[0, 0].semilogx(r_sub["alpha"], r_sub["val_mape"], marker="o", linewidth=2, label="Ridge (L2)")
    axes[0, 0].semilogx(l_sub["alpha"], l_sub["val_mape"], marker="s", linewidth=2, label="Lasso (L1)")
    axes[0, 0].semilogx(e_sub["alpha"], e_sub["val_mape"], marker="^", linewidth=2, label="ElasticNet (l1_ratio=0.8)")
    axes[0, 0].set_xlabel("Regularisation Strength alpha (log scale)")
    axes[0, 0].set_ylabel("2014 Validation Demand MAPE (%)")
    axes[0, 0].set_title("(a) Regularisation Hyperparameter Tuning on 2014 Validation")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend()

    # Panel B: Diurnal MAPE across the 48 half-hours of the day on 2014
    hours = np.arange(N_PERIODS) / 2.0
    hh_b1 = np.mean(np.abs(Y_va - b1_preds) / Y_va, axis=0) * 100.0
    hh_flat = np.mean(np.abs(Y_va - flat_preds) / Y_va, axis=0) * 100.0
    hh_ridge = np.mean(np.abs(Y_va - best_preds_by_family["Ridge (L2)"]["preds"]) / Y_va, axis=0) * 100.0
    hh_lasso = np.mean(np.abs(Y_va - best_preds_by_family["Lasso (L1)"]["preds"]) / Y_va, axis=0) * 100.0

    axes[0, 1].plot(hours, hh_b1, color="gray", linestyle="--", linewidth=1.6, label="Yesterday Baseline (4.93%)")
    axes[0, 1].plot(hours, hh_flat, color="darkorange", linewidth=1.8, label="Flat Daily Ridge (1.92%)")
    axes[0, 1].plot(hours, hh_ridge, color="royalblue", linewidth=2.0, label="Per-Slot Ridge (1.75%)")
    axes[0, 1].plot(hours, hh_lasso, color="crimson", linewidth=2.0, label="Per-Slot Lasso (1.74%)")
    axes[0, 1].set_xlabel("Time of Day (Hours, AEST)")
    axes[0, 1].set_ylabel("2014 Validation MAPE (%)")
    axes[0, 1].set_title("(b) Diurnal Forecasting Error Profile Across 48 Half-Hours (2014)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel C: Monthly 2014 Validation MAPE Comparison
    months_2014 = val_dates.month
    m_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m_b1, m_b7, m_lasso, m_ridge = [], [], [], []
    for m_idx in range(1, 13):
        m_mask = months_2014 == m_idx
        m_b1.append(mape_metric(Y_va[m_mask], b1_preds[m_mask]))
        m_b7.append(mape_metric(Y_va[m_mask], b7_preds[m_mask]))
        m_ridge.append(mape_metric(Y_va[m_mask], best_preds_by_family["Ridge (L2)"]["preds"][m_mask]))
        m_lasso.append(mape_metric(Y_va[m_mask], best_preds_by_family["Lasso (L1)"]["preds"][m_mask]))

    x_m = np.arange(12)
    axes[1, 0].plot(x_m, m_b1, marker="o", linestyle="--", color="gray", label="Same HH Yesterday")
    axes[1, 0].plot(x_m, m_b7, marker="s", linestyle=":", color="slategray", label="Same HH Last Week")
    axes[1, 0].plot(x_m, m_ridge, marker="^", linewidth=2, color="royalblue", label="Per-Slot Ridge (L2)")
    axes[1, 0].plot(x_m, m_lasso, marker="d", linewidth=2, color="crimson", label="Per-Slot Lasso (L1)")
    axes[1, 0].set_xticks(x_m)
    axes[1, 0].set_xticklabels(m_names)
    axes[1, 0].set_xlabel("Month of 2014")
    axes[1, 0].set_ylabel("Monthly Demand MAPE (%)")
    axes[1, 0].set_title("(c) Monthly Demand MAPE Across 2014 Validation Set")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Panel D: Predicted vs Observed NSW Demand for a 7-Day Summer Heatwave Week (Jan 13-19, 2014)
    w_start, w_end = 12, 19  # Jan 13 to Jan 19, 2014
    t_axis = pd.date_range(val_dates[w_start], periods=(w_end - w_start) * N_PERIODS, freq="30min")
    obs_week = Y_va[w_start:w_end].reshape(-1)
    pred_week = best_preds_by_family["Lasso (L1)"]["preds"][w_start:w_end].reshape(-1)
    b1_week = b1_preds[w_start:w_end].reshape(-1)

    axes[1, 1].plot(t_axis, obs_week, color="black", linewidth=2.0, label="Observed NSW Demand")
    axes[1, 1].plot(t_axis, pred_week, color="crimson", linewidth=1.8, linestyle="-", label="Regularised Linear (Lasso)")
    axes[1, 1].plot(t_axis, b1_week, color="steelblue", linewidth=1.2, linestyle="--", alpha=0.75, label="Yesterday Baseline")
    axes[1, 1].set_xlabel("Date (January 2014)")
    axes[1, 1].set_ylabel("NSW Operational Demand (MW)")
    axes[1, 1].set_title("(d) Predicted vs. Observed Demand (13–19 Jan 2014)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_hyperparameter_tuning_and_validation.png"), dpi=200)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    t_start = time.time()

    if len(sys.argv) < 2:
        csv_path = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
    else:
        csv_path = sys.argv[1]

    print(f"[1/5] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()

    # Restrict to Stage 1 period (2010-2014) as specified in the protocol:
    # Train on 2010-2013, select features/algorithms/hyperparameters on 2014, then refit on 2010-2014.
    days = market_day(df.index)
    df_stage1 = df.loc[days.year <= 2014].copy()

    print("[2/5] Extracting candidate features (fitting Day-Type PCA+KMeans on 2010-2013 only)...")
    bundle = build_candidate_features(
        df_stage1,
        day_type_transformer=None,
        fit_day_type_max_year=2013,
        include_flat_daily=True,
    )
    X_3d = bundle["X_3d"]
    Y_2d = bundle["Y_2d"]
    X_flat = bundle["X_flat"]
    target_dates = bundle["target_dates"]
    feature_names = bundle["feature_names"]
    feature_groups = bundle["feature_groups"]

    train_mask = target_dates.year <= 2013
    val_mask = target_dates.year == 2014
    print(
        f"      Candidate tensor shape: {X_3d.shape} | "
        f"Train days (2010-2013): {int(train_mask.sum())} | Val days (2014): {int(val_mask.sum())}"
    )

    # Compute 2014 baselines and Flat Daily Multi-Output Ridge for comparison
    Y_va = Y_2d[val_mask]
    dem_d_idx = feature_names.index("dem_d_t")
    dem_d6_idx = feature_names.index("dem_d6_t")
    b1_preds = X_3d[val_mask, :, dem_d_idx]
    b7_preds = X_3d[val_mask, :, dem_d6_idx]
    b1_mape, b1_mae = mape_metric(Y_va, b1_preds), mae_metric(Y_va, b1_preds)
    b7_mape, b7_mae = mape_metric(Y_va, b7_preds), mae_metric(Y_va, b7_preds)

    sc_flat = StandardScaler()
    X_flat_tr = sc_flat.fit_transform(X_flat[train_mask])
    X_flat_va = sc_flat.transform(X_flat[val_mask])
    flat_model = Ridge(alpha=1.0).fit(X_flat_tr, Y_2d[train_mask])
    flat_preds = flat_model.predict(X_flat_va)
    flat_mape, flat_mae = mape_metric(Y_va, flat_preds), mae_metric(Y_va, flat_preds)

    print("[3/5] Running FeatureSelector (Group Ablation + Hybrid L1 & Residual-F Ranking)...")
    selector = FeatureSelector(lasso_alpha=0.30, l1_weight=0.85, f_weight=0.15, eval_ridge_alpha=5.0)
    selector.fit_and_select(X_3d, Y_2d, train_mask, val_mask, feature_names, feature_groups)
    print(f"      Selected optimal feature subset: k* = {selector.best_k_} / {len(feature_names)} features")
    for _, row in selector.ablation_df_[selector.ablation_df_["mode"] == "progressive_add"].iterrows():
        print(f"        + {row['group']:20s} ({int(row['n_features']):2d} feats) -> Val MAPE: {row['val_mape']:.4f}%, MAE: {row['val_mae']:.2f} MW")

    print("[4/5] Tuning hyperparameters across Ridge (L2), Lasso (L1), and ElasticNet (L1+L2) on 2014...")
    hp_df, best_overall, best_preds_by_family = evaluate_hyperparameter_grid(
        X_3d=X_3d,
        Y_2d=Y_2d,
        train_mask=train_mask,
        val_mask=val_mask,
        selected_indices=selector.selected_indices_,
        selected_names=selector.selected_feature_names_,
    )

    summary_rows = [
        {
            "Method": "Baseline 1: Same half-hour yesterday",
            "Features_Per_Target": 1,
            "Best_Hyperparameters": "lag=1 day",
            "Val_2014_MAPE_pct": b1_mape,
            "Val_2014_MAE_MW": b1_mae,
            "Relative_Improvement_vs_B1_pct": 0.0,
        },
        {
            "Method": "Baseline 2: Same half-hour last week",
            "Features_Per_Target": 1,
            "Best_Hyperparameters": "lag=7 days",
            "Val_2014_MAPE_pct": b7_mape,
            "Val_2014_MAE_MW": b7_mae,
            "Relative_Improvement_vs_B1_pct": 100.0 * (1.0 - b7_mape / b1_mape),
        },
        {
            "Method": "Flat Daily Multi-Output Ridge (L2)",
            "Features_Per_Target": X_flat.shape[1],
            "Best_Hyperparameters": "alpha=1.0",
            "Val_2014_MAPE_pct": flat_mape,
            "Val_2014_MAE_MW": flat_mae,
            "Relative_Improvement_vs_B1_pct": 100.0 * (1.0 - flat_mape / b1_mape),
        },
    ]
    for fam_name in ["Ridge (L2)", "ElasticNet (L1+L2)", "Lasso (L1)"]:
        info = best_preds_by_family[fam_name]
        p = info["params"]
        hp_str = f"alpha={p['alpha']}" if p["model_type"] != "elasticnet" else f"alpha={p['alpha']}, l1_ratio={p['l1_ratio']}"
        summary_rows.append({
            "Method": f"Per-Slot {fam_name} (Selected k={selector.best_k_})",
            "Features_Per_Target": selector.best_k_,
            "Best_Hyperparameters": hp_str,
            "Val_2014_MAPE_pct": info["mape"],
            "Val_2014_MAE_MW": info["mae"],
            "Relative_Improvement_vs_B1_pct": 100.0 * (1.0 - info["mape"] / b1_mape),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Stage 1 Validation Summary (2014) ===")
    print(summary_df.to_string(index=False))

    print(
        f"\n[5/5] Optimal Stage 1 model on 2014: {best_overall['model_family']} "
        f"(alpha={best_overall['alpha']}, l1_ratio={best_overall['l1_ratio']}) "
        f"with MAPE={best_overall['val_mape']:.4f}%, MAE={best_overall['val_mae']:.2f} MW."
    )
    print("      Fitting Stage 3 Final Model on Last 4Y (1460d) up to 2026-10-31 with Exponential Recency (H=365d)...")

    STAGE3_WINDOW_DAYS = 1460          # Hybrid: Last 4 years of operational days
    STAGE3_HALFLIFE_DAYS = 365.0       # Exponential decay half-life = 1 year
    STAGE3_DAYTYPE_WINDOW_DAYS = 1095  # Trailing 3 years for Day-Type PCA+KMeans

    df_stage3 = df.loc[days <= pd.Timestamp("2026-10-31")].copy()
    days_s3 = market_day(df_stage3.index)
    counts_s3 = pd.Series(1, index=df_stage3.index).groupby(days_s3).sum()
    valid_days_s3 = pd.DatetimeIndex(sorted(counts_s3[counts_s3 == N_PERIODS].index))
    df_s3_clean = df_stage3.loc[days_s3.isin(valid_days_s3)]
    nsw_dem_s3 = df_s3_clean["nsw_demand"].to_numpy(dtype=np.float64).reshape(len(valid_days_s3), N_PERIODS)

    stage3_dt = DayTypeTransformer(n_components=5, n_clusters=4, random_state=42)
    dt_start_idx = max(0, len(valid_days_s3) - STAGE3_DAYTYPE_WINDOW_DAYS)
    stage3_dt.fit(nsw_dem_s3[dt_start_idx:])

    stage3_bundle = build_candidate_features(
        df_s3_clean,
        day_type_transformer=stage3_dt,
        include_flat_daily=False,
    )
    X_s3 = stage3_bundle["X_3d"][-STAGE3_WINDOW_DAYS:]
    Y_s3 = stage3_bundle["Y_2d"][-STAGE3_WINDOW_DAYS:]
    stage3_dates = stage3_bundle["target_dates"][-STAGE3_WINDOW_DAYS:]
    d_end = stage3_dates[-1]
    ages_days = (d_end - stage3_dates).days.to_numpy(dtype=np.float64)
    raw_weights = np.power(2.0, -ages_days / STAGE3_HALFLIFE_DAYS)
    sample_weight = raw_weights / np.mean(raw_weights)

    final_model = PerSlotLinearRegressor(
        model_type=best_overall["model_type"],
        alpha=best_overall["alpha"],
        l1_ratio=best_overall["l1_ratio"],
        selected_indices=selector.selected_indices_,
        selected_feature_names=selector.selected_feature_names_,
        day_type_transformer=stage3_dt,
    )
    final_model.fit(X_s3, Y_s3, sample_weight=sample_weight)

    artifact_dir = os.path.join(HERE, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(artifact_dir, "demand_model.joblib"))

    config_data = {
        "window_days": WINDOW_DAYS,
        "stage1_train_years": "2010-2013",
        "stage1_val_year": "2014",
        "stage3_train_start": str(stage3_dates[0].date()),
        "stage3_train_end": str(d_end.date()),
        "stage3_n_train_days": int(len(stage3_dates)),
        "stage3_adaptation_strategy": "Hybrid: Last 4Y (1460d) + Exponential Recency Weighting (H=365d) + 3Y DayType Clustering",
        "stage3_window_days": STAGE3_WINDOW_DAYS,
        "stage3_halflife_days": STAGE3_HALFLIFE_DAYS,
        "stage3_daytype_window_days": STAGE3_DAYTYPE_WINDOW_DAYS,
        "selected_model_family": best_overall["model_family"],
        "selected_model_type": best_overall["model_type"],
        "alpha": best_overall["alpha"],
        "l1_ratio": best_overall["l1_ratio"],
        "n_candidate_features": len(feature_names),
        "n_selected_features": selector.best_k_,
        "selected_features": selector.selected_feature_names_,
        "val_2014_mape": round(best_overall["val_mape"], 4),
        "val_2014_mae": round(best_overall["val_mae"], 2),
        "baseline_yesterday_2014_mape": round(b1_mape, 4),
        "relative_improvement_pct": round(100.0 * (1.0 - best_overall["val_mape"] / b1_mape), 2),
    }
    with open(os.path.join(artifact_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_data, fh, indent=2)

    try:
        analysis_dir = os.path.join(HERE, "analysis_outputs")
        save_stage1_plots_and_tables(
            out_dir=analysis_dir,
            selector=selector,
            hp_df=hp_df,
            summary_df=summary_df,
            best_preds_by_family=best_preds_by_family,
            flat_preds=flat_preds,
            b1_preds=b1_preds,
            b7_preds=b7_preds,
            Y_va=Y_va,
            val_dates=target_dates[val_mask],
        )
        print(f"      Saved Stage 1 validation tables and figures to {analysis_dir}/")
    except OSError:
        pass

    elapsed = time.time() - t_start
    print(
        f"      Saved Stage 3 final model artifacts ({len(stage3_dates)} days, "
        f"{stage3_dates[0].date()} to {d_end.date()}, H={STAGE3_HALFLIFE_DAYS:.0f}d) "
        f"to {artifact_dir}/ in {elapsed:.1f}s."
    )


INITIAL_TRAIN_START = pd.Timestamp("2010-01-01")
INITIAL_TRAIN_END = pd.Timestamp("2014-01-01")
VALIDATION_END = pd.Timestamp("2015-01-01")
FINAL_KNN_WINDOW_YEARS = 4


def make_training_data(df):
    df = df.sort_index().copy()
    day = price_market_day(df.index)
    daily_mean = df.groupby(day)["nsw_price"].transform("mean")
    t = pd.DataFrame(index=df.index)
    t["day"] = day
    shifted = df.index - pd.Timedelta(minutes=30)
    t["period"] = shifted.hour * 2 + shifted.minute // 30
    t["y_up"] = (df["nsw_price"] > daily_mean.shift(48)).astype(int)
    t["demand_lag1"] = df["nsw_demand"].shift(48)
    t["demand_lag7"] = df["nsw_demand"].shift(48 * 7)
    t["price_lag1"] = df["nsw_price"].shift(48)
    t["previous_mean_price"] = daily_mean.shift(48)
    t["up_lag1"] = (df["nsw_price"].shift(48) > daily_mean.shift(48)).astype(int)
    t["up_lag7"] = (df["nsw_price"].shift(48 * 7) > daily_mean.shift(48)).astype(int)
    for col in ("syd_temp", "syd_solar", "syd_rh"):
        t[col] = df[col]
    t = t.iloc[48 * 7:].dropna()
    return price_features_from_components(t), t["y_up"], t["day"]


def new_model():
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=1),
    )


def train_price_model(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")
    X, y, day = make_training_data(df)

    # Initial model: 2010–2013 only.
    initial_fit = (day >= INITIAL_TRAIN_START) & (day < INITIAL_TRAIN_END)
    if not initial_fit.any():
        raise ValueError("No initial training data in 2010–2013")
    model = new_model()
    model.fit(X.loc[initial_fit], y.loc[initial_fit])
    print(f"Initial fit: {initial_fit.sum():,} rows through {day.loc[initial_fit].max().date()}")

    # Validation: 2014 only. Predicting does not change the trained model.
    validation = (day >= INITIAL_TRAIN_END) & (day < VALIDATION_END)
    if validation.any():
        predictions = model.predict(X.loc[validation])
        score = balanced_accuracy_score(y.loc[validation], predictions)
        print(f"2014 validation balanced accuracy: {score:.4f}")
    else:
        print("Warning: no 2014 validation data available")

    # Final submission model: refit on the latest four calendar years of
    # complete market days in the supplied dataset. No older rows are fitted.
    counts = day.value_counts()
    complete_days = pd.DatetimeIndex(counts[counts == N_PERIODS].index).sort_values()
    if complete_days.empty:
        raise ValueError("No complete 48-period market days available for KNN training")

    last_day = complete_days.max()
    window_start = last_day + pd.Timedelta(days=1) - pd.DateOffset(years=FINAL_KNN_WINDOW_YEARS)
    final_fit = (day >= window_start) & (day <= last_day) & day.isin(complete_days)
    if final_fit.sum() < 50:
        raise ValueError("Insufficient observations in final four-year KNN window")

    model = new_model()
    model.fit(X.loc[final_fit], y.loc[final_fit])
    first_day = day.loc[final_fit].min()
    print(f"Final KNN fit: {final_fit.sum():,} rows from {first_day.date()} "
          f"through {last_day.date()} (4-year window starts {window_start.date()})")

    out = Path(__file__).resolve().parent / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "price_model.joblib"
    joblib.dump(model, path)
    print(f"Saved latest-four-year KNN model: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()  # Demand pipeline; accepts optional CSV path from sys.argv[1].
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
    train_price_model(csv_path)

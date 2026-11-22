"""Train the demand regressor and price-direction KNN submission artifacts.

Run: python train.py path/to/nem_weather_halfhourly_2010_2026.csv
Requires features.py in artifacts/ and writes models to artifacts/.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "artifacts"))

from features import DayTypeTransformer, FeatureSelector, N_PERIODS, WINDOW_DAYS, build_candidate_features, market_day, mape_metric, mae_metric
from model import PerSlotLinearRegressor, market_day as price_market_day, price_features_from_components

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


def train_demand_model(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    days = market_day(df.index)

    # Retain 2014 selection because the final model depends on the selected
    # feature indices and hyperparameters. Remove report-only diagnostics.
    stage1 = build_candidate_features(
        df.loc[days.year <= 2014],
        day_type_transformer=None,
        fit_day_type_max_year=2013,
        include_flat_daily=False,
    )
    dates = stage1["target_dates"]
    selector = FeatureSelector(lasso_alpha=0.30, l1_weight=0.85,
                               f_weight=0.15, eval_ridge_alpha=5.0)
    selector.fit_and_select(
        stage1["X_3d"], stage1["Y_2d"], dates.year <= 2013,
        dates.year == 2014, stage1["feature_names"], stage1["feature_groups"]
    )
    _, best, _ = evaluate_hyperparameter_grid(
        stage1["X_3d"], stage1["Y_2d"], dates.year <= 2013,
        dates.year == 2014, selector.selected_indices_,
        selector.selected_feature_names_,
    )

    # Final demand model: latest 1460 complete days, one-year weight half-life,
    # and day-type transformer fitted on the latest 1095 complete days.
    stage3 = df.loc[days <= pd.Timestamp("2026-10-31")]
    days3 = market_day(stage3.index)
    counts = pd.Series(1, index=stage3.index).groupby(days3).sum()
    valid = pd.DatetimeIndex(sorted(counts[counts == N_PERIODS].index))
    if len(valid) < 1095:
        raise ValueError("Not enough complete days for demand training")
    clean = stage3.loc[days3.isin(valid)]
    demand = clean["nsw_demand"].to_numpy(dtype=np.float64).reshape(len(valid), N_PERIODS)
    transformer = DayTypeTransformer(n_components=5, n_clusters=4, random_state=42)
    transformer.fit(demand[-1095:])
    bundle = build_candidate_features(clean, day_type_transformer=transformer,
                                      include_flat_daily=False)
    X = bundle["X_3d"][-1460:]
    Y = bundle["Y_2d"][-1460:]
    training_dates = bundle["target_dates"][-1460:]
    ages = (training_dates[-1] - training_dates).days.to_numpy(dtype=np.float64)
    weights = np.power(2.0, -ages / 365.0)
    weights /= weights.mean()

    model = PerSlotLinearRegressor(
        model_type=best["model_type"], alpha=best["alpha"],
        l1_ratio=best["l1_ratio"],
        selected_indices=selector.selected_indices_,
        selected_feature_names=selector.selected_feature_names_,
        day_type_transformer=transformer,
    )
    model.fit(X, Y, sample_weight=weights)
    out = HERE / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "demand_model.joblib")
    with (out / "config.json").open("w", encoding="utf-8") as fh:
        json.dump({"window_days": WINDOW_DAYS}, fh, indent=2)
    print(f"Demand model saved: {len(training_dates)} days, ending {training_dates[-1].date()}")


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

    out = HERE / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "price_model.joblib"
    joblib.dump(model, path)
    print(f"Saved latest-four-year KNN model: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    train_demand_model(csv_path)
    train_price_model(csv_path)

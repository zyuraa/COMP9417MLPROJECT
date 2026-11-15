"""Select the demand model on 2014, then refit the winner through 2014."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from features import FEATURE_GROUPS, WEATHER, make_features, market_day
from model import DemandModel, demand_offset


PACKAGE = Path(__file__).resolve().parent
SEED = 9417
BASE_SETTINGS = (15, 150)


def metrics(observed, predicted):
    error = (observed - predicted).abs()
    return {"mape_pct": 100 * (error / observed).mean(), "mae_mw": error.mean()}


def fit_model(features, demand, group, target, settings):
    leaves, iterations = settings
    estimator = HistGradientBoostingRegressor(
        learning_rate=0.08, max_leaf_nodes=leaves, max_iter=iterations,
        min_samples_leaf=40, l2_regularization=1.0, max_bins=128,
        early_stopping=False, random_state=SEED,
    )
    with threadpool_limits(limits=2, user_api="openmp"):
        estimator.fit(features[FEATURE_GROUPS[group]], demand - demand_offset(features, target))
    return DemandModel(estimator, target)


def train(csv_path):
    df = pd.read_csv(csv_path, usecols=["timestamp", "nsw_demand", *WEATHER],
                     index_col="timestamp", parse_dates=["timestamp"]).sort_index()
    # Midnight closes the preceding market day, including at the year boundary.
    df = df.loc[market_day(df.index).year <= 2014]
    features = make_features(df).dropna()
    years = market_day(features.index).year
    demand = df.loc[features.index, "nsw_demand"]
    x_train, y_train = features.loc[years <= 2013], demand.loc[years <= 2013]
    x_val, y_val = features.loc[years == 2014], demand.loc[years == 2014]
    if x_train.empty or len(x_val) != 365 * 48:
        raise ValueError("Expected training data from 2010–2013 and all 48 slots of every day in 2014.")

    trials = []
    for name, lag in [("Yesterday", "demand_lag1"), ("Last week", "demand_lag7")]:
        trials.append({"model": name, "step": "baseline", "n_half_hours": len(y_val),
                       **metrics(y_val, x_val[lag])})

    def evaluate(step, group, target, settings):
        candidate = fit_model(x_train, y_train, group, target, settings)
        scores = metrics(y_val, candidate.predict(x_val))
        leaves, iterations = settings
        row = {"model": "Histogram gradient boosting", "step": step, "features": group,
               "target": target, "max_leaf_nodes": leaves, "max_iter": iterations,
               "n_half_hours": len(y_val), **scores}
        trials.append(row)
        print(f"{step:12} {group:8} {target:12} {leaves:2} leaves, {iterations} rounds"
              f"   MAPE {scores['mape_pct']:.3f}%   MAE {scores['mae_mw']:.1f} MW", flush=True)
        return row

    def rank(row):
        return row["mape_pct"], row["mae_mw"]

    # Change one decision at a time. All selection happens on the same 2014 timestamps.
    ablations = [evaluate("features", group, "demand_lag1", BASE_SETTINGS) for group in FEATURE_GROUPS]
    best_group = min(ablations, key=rank)
    group = best_group["features"]
    other_targets = ["demand", "demand_lag7"]
    if "matched_profile" in FEATURE_GROUPS[group]:
        other_targets.append("matched_profile")
    targets = [best_group] + [evaluate("target", group, target, BASE_SETTINGS)
                              for target in other_targets]
    best_target = min(targets, key=rank)
    target = best_target["target"]
    finalists = [best_target] + [evaluate("capacity", group, target, settings)
                                 for settings in ((15, 300), (31, 150), (31, 300))]
    best = min(finalists, key=rank)
    settings = best["max_leaf_nodes"], best["max_iter"]
    frozen = fit_model(features, demand, group, target, settings)

    out = PACKAGE / "artifacts"
    results = PACKAGE / "results"
    out.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    joblib.dump({"estimator": frozen.estimator, "target": target}, out / "demand_model.joblib", compress=3)
    pd.DataFrame([{**row, "selected": row is best} for row in trials]).to_csv(
        results / "stage1_2014.csv", index=False)
    print(f"Frozen model fitted on {len(demand):,} slots through 2014-12-31.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    train(parser.parse_args().csv_path)

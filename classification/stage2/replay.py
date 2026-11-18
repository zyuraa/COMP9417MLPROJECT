"""Stage 2 replay: run a classifier (kNN or logistic regression) over 2015-2026 under an adaptation strategy (when to retrain, and
on which data). Writes predictions/<name>.parquet in the shared format, so analysis.py puts
every strategy on the same tables and figures.

Timing (spec p.5): a retraining at the end of day D may only use rows whose target day is
<= D, and the new model forecasts from day D+1 onwards. Here a retrain date is that first
forecast day, so a retrain on 2017-01-01 is fitted on target days up to 2016-12-31.
The first model is the Stage 1 frozen one (fitted on 2010-2014) and does not count towards
the budget of six retrains.

Strategies:
    --every-years N      retrain on 1 January every N years (2017, 2019, ... for N=2)
    --alarms FILE        retrain the day after each alarm day; FILE is a csv with a "date"
                         column (the day D on which the detector signalled)
    --window-years W     train on the last W years only (sliding window); default is every
                         day since 2010 (expanding window)

Run from the repo root, e.g.:
    python classification/stage2/replay.py --every-years 2 --name knn_every2y_expanding
    python classification/stage2/replay.py --every-years 2 --window-years 3 --name knn_every2y_sliding3y
    python classification/stage2/replay.py --alarms alarms.csv --name knn_drift
    python classification/stage2/replay.py --model logreg --every-years 2 --name logreg_every2y_expanding
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluation as ev
import features as F

FEATURES = F.SHARED_FEATURES
START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-10-31")
BUDGET = 6

ap = argparse.ArgumentParser()
when = ap.add_mutually_exclusive_group(required=True)
when.add_argument("--every-years", type=int)
when.add_argument("--alarms", help="csv with a 'date' column of alarm days")
ap.add_argument("--window-years", type=int, default=None)
ap.add_argument("--model", default="knn", choices=["knn", "logreg"],
                help="Stage 1 settings: kNN k=50 distance; logistic regression C=1 balanced")
ap.add_argument("--name", required=True, help="predictions file name")
args = ap.parse_args()


SETTINGS = {
    "knn": ("kNN", {"k": 50, "weights": "distance"},
            lambda: KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)),
    "logreg": ("logistic regression", {"C": 1.0, "class_weight": "balanced"},
               lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
}
FAMILY, PARAMS, ESTIMATOR = SETTINGS[args.model]


def model():
    return make_pipeline(StandardScaler(), ESTIMATOR())


def retrain_dates():
    """First forecast day of each retrained model."""
    if args.every_years:
        dates = [pd.Timestamp(f"{y}-01-01")
                 for y in range(START.year + args.every_years, END.year + 1, args.every_years)]
    else:
        alarms = pd.to_datetime(pd.read_csv(args.alarms)["date"]).sort_values()
        dates = list(alarms + pd.Timedelta(days=1))
    dates = [d for d in dates if START < d <= END]
    if len(dates) > BUDGET:
        sys.exit(f"{len(dates)} retrains requested; the budget is {BUDGET} (spec p.5)")
    return dates


t = F.build_table(F.load(ev.DATA))
X, y, day = t[FEATURES], t["y_up"].astype(int), t["day"]
dates = retrain_dates()

# 2014 validation rows, identical to the Stage 1 model, so every file covers the same rows
segments = [("val", day.dt.year <= 2013, day.dt.year == 2014)]
bounds = [START] + dates + [END + pd.Timedelta(days=1)]
for i, (first, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
    fit = day < first
    if i > 0 and args.window_years:
        fit &= day >= first - pd.DateOffset(years=args.window_years)
    segments.append((f"model {i} from {first.date()}", fit, (day >= first) & (day < stop)))

parts = []
for label, fit, pred in segments:
    start = time.time()
    p = model().fit(X[fit], y[fit]).predict_proba(X[pred])[:, 1]
    parts.append((t.index[pred], y[pred], (p > 0.5).astype(int), p))
    print(f"{label}: fit {fit.sum():,} rows ({day[fit].min().date()} to {day[fit].max().date()}), "
          f"predict {pred.sum():,} rows ({time.time() - start:.0f}s)")

ev.save_predictions(args.name, *(np.concatenate([p[j] for p in parts]) for j in range(4)),
                    info={"model": FAMILY, "params": PARAMS,
                          "n_features": len(FEATURES),
                          "strategy": {"every_years": args.every_years, "alarms": args.alarms,
                                       "window_years": args.window_years,
                                       "retrains": [str(d.date()) for d in dates]}})

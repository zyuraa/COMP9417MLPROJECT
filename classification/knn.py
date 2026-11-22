"""kNN price-direction classifier. Writes predictions/knn.parquet for analysis.py.

  val    : fit on 2010-2013, predict 2014          (Stage 1 selection)
  frozen : refit on 2010-2014, predict 2015-2026   (Stage 2 frozen model, no updates)

FEATURES selects columns from features.py (currently the shared set). Run from the repo root:
    python classification/knn.py [--k 50] [--weights distance]
"""
import argparse
import os
import sys
import time

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev
import features as F

FEATURES = F.SHARED_FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--weights", default="distance", choices=["uniform", "distance"])
ap.add_argument("--name", default="knn", help="predictions file name")
args = ap.parse_args()


def model():
    return make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=args.k, weights=args.weights, n_jobs=-1))


t = F.build_table(F.load(ev.DATA))
X, y, year = t[FEATURES], t["y_up"].astype(int), t["year"]
parts = []
for split, fit_mask, pred_mask in [("val", year <= 2013, year == 2014),
                                   ("frozen", year <= 2013, year >= 2015)]:
    start = time.time()
    m = model().fit(X[fit_mask], y[fit_mask])
    p = m.predict_proba(X[pred_mask])[:, 1]
    parts.append((t.index[pred_mask], y[pred_mask], (p > 0.5).astype(int), p))
    print(f"{split}: fit {fit_mask.sum():,} rows, predict {pred_mask.sum():,} rows "
          f"({time.time() - start:.0f}s)")

ev.save_predictions(args.name,
                    np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                    np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]),
                    info={"model": "kNN", "params": {"k": args.k, "weights": args.weights},
                          "n_features": len(FEATURES)})

"""Frozen KNN error-CUSUM evaluation, 2015–2026.

Run from the repository root:
    python classification/stage2/run_frozen_cusum.py

No model retraining or detector recalibration during evaluation.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_DIR = Path(__file__).resolve().parents[1]
ROOT = CLASS_DIR.parent
sys.path.insert(0, str(CLASS_DIR))
sys.path.insert(0, str(ROOT))

import evaluation as ev
import features as F
from input_drift_detector.drift import build_error_detector

START = pd.Timestamp('2015-01-01')
END = pd.Timestamp('2026-10-31')
MARGIN = 1.1
COOLDOWN = 60


def new_model():
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1),
    )


def daily_errors(model, X, y, day, mask):
    pred = model.predict(X.loc[mask])
    return pd.Series(
        (pred != y.loc[mask].to_numpy()).astype(float),
        index=day.loc[mask].to_numpy(),
    ).groupby(level=0).mean().sort_index()


def main():
    t = F.build_table(F.load(ev.DATA))
    X = t[F.SHARED_FEATURES]
    y = t['y_up'].astype(int)
    day = pd.to_datetime(t['day'])

    train_mask = day.dt.year.between(2010, 2013)
    val_mask = day.dt.year == 2014
    test_mask = (day >= START) & (day <= END)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError('Missing training, validation, or evaluation observations')

    # Fit ONCE, only on 2010–2013 data.
    model = new_model().fit(X.loc[train_mask], y.loc[train_mask])

    # Calibrate the detector on frozen-model errors from 2014.
    val_err = daily_errors(model, X, y, day, val_mask)
    detector = build_error_detector(
        val_err, k=0.5, margin=MARGIN,
        cooldown=COOLDOWN, max_alarms=10**9,
    )
    if not np.isfinite(detector.sigma) or detector.sigma <= 0 or not np.isfinite(detector.h) or detector.h <= 0:
        raise ValueError('Invalid 2014 CUSUM calibration')

    print(f'2014-calibrated CUSUM threshold: {detector.h:.4f}')
    print(f'Reference daily error: {detector.mu:.4f}')

    # Predict using the SAME frozen model on all evaluation observations.
    pred = model.predict(X.loc[test_mask])
    test_day = day.loc[test_mask]
    test_y = y.loc[test_mask].to_numpy()
    err = pd.Series(
        (pred != test_y).astype(float),
        index=test_day.to_numpy(),
    ).groupby(level=0).mean().sort_index()

    trace = []
    alarms = []
    for d, daily_error in err.items():
        threshold = float(detector.h)
        pre_reset = max(0.0, detector.S + (float(daily_error) - detector.mu) / detector.sigma - detector.k)
        fired = bool(detector.update(float(daily_error)))
        trace.append({
            'date': d,
            'daily_error': float(daily_error),
            # Match replay_error.py: detector resets on alarm, so show h.
            'cusum': pre_reset,
            'threshold': threshold,
            'reference_mean': float(detector.mu),
            'reference_std': float(detector.sigma),
            'alarm': fired,
        })
        if fired:
            alarms.append(d)
            print(f'CUSUM alarm: {d.date()}')

    out_dir = ROOT / 'classification' / 'stage2' / 'alarms'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(out_dir / 'knn_frozen_cusum_trace.csv', index=False)
    pd.DataFrame({'date': pd.to_datetime(alarms)}).to_csv(
        out_dir / 'knn_frozen_cusum_alarms.csv', index=False
    )
    print(f'Saved frozen CUSUM trace and {len(alarms)} alarms to {out_dir}')


if __name__ == '__main__':
    main()
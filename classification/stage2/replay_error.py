"""Chronological error-CUSUM-triggered adaptation replay (Python 3.11).

Place in classification/stage2/ and run from repository root.
Recalibrates the CUSUM reference and threshold after each retraining using
out-of-sample errors from a historical chronological holdout.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
BUDGET = 6
MARGIN = 1.1

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logreg', 'knn'], required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--window-years', type=int, default=None)
parser.add_argument('--cooldown', type=int, default=60)
parser.add_argument('--validation-days', type=int, default=365)
args = parser.parse_args()


def new_model():
    if args.model == 'logreg':
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    else:
        clf = KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1)
    return make_pipeline(StandardScaler(), clf)


def historical_validation_errors(X, y, day, train_mask, window_days=365):
    """Daily out-of-sample errors from the most recent historical holdout."""
    available_days = day.loc[train_mask]
    if available_days.empty:
        return pd.Series(dtype=float)

    cutoff = available_days.max() - pd.Timedelta(days=window_days - 1)
    fit_mask = train_mask & (day < cutoff)
    val_mask = train_mask & (day >= cutoff)
    if not fit_mask.any() or not val_mask.any():
        return pd.Series(dtype=float)
    if y.loc[fit_mask].nunique() < 2:
        return pd.Series(dtype=float)

    temporary_model = new_model().fit(X.loc[fit_mask], y.loc[fit_mask])
    predictions = temporary_model.predict(X.loc[val_mask])
    return pd.Series(
        (predictions != y.loc[val_mask].to_numpy()).astype(float),
        index=day.loc[val_mask].to_numpy(),
    ).groupby(level=0).mean().sort_index()


def recalibrate_after_detection(detector, errors):
    """Update reference and threshold; retain old calibration if invalid.

    Uses drift.py's rebase() and calibrate(). The first half of the historical
    holdout estimates the reference; the second half calibrates the threshold.
    """
    split = len(errors) // 2
    reference = errors.iloc[:split]
    calibration = errors.iloc[split:]
    if (len(reference) < 30 or len(calibration) < 30
            or not np.isfinite(reference.std(ddof=1))
            or reference.std(ddof=1) <= 0):
        return False

    old_mu, old_sigma, old_h = detector.mu, detector.sigma, detector.h
    detector.rebase(reference)
    detector.calibrate(calibration, margin=MARGIN)

    if (not np.isfinite(detector.sigma) or detector.sigma <= 0
            or not np.isfinite(detector.h) or detector.h <= 0):
        detector.mu, detector.sigma, detector.h = old_mu, old_sigma, old_h
        detector.S = 0.0
        return False
    return True


print('Loading data and building features...', flush=True)
t = F.build_table(F.load(ev.DATA))
X = t[F.SHARED_FEATURES]
y = t['y_up'].astype(int)
day = pd.to_datetime(t['day'])

print('Calibrating initial CUSUM on 2014 validation errors...', flush=True)
val_model = new_model().fit(X.loc[day.dt.year <= 2013], y.loc[day.dt.year <= 2013])
val_mask = day.dt.year == 2014
val_pred = val_model.predict(X.loc[val_mask])
val_err = pd.Series(
    (val_pred != y.loc[val_mask].to_numpy()).astype(float),
    index=day.loc[val_mask].to_numpy(),
).groupby(level=0).mean().sort_index()

detector = build_error_detector(
    val_err, k=0.5, margin=MARGIN,
    cooldown=args.cooldown, max_alarms=BUDGET,
)
if (not np.isfinite(detector.sigma) or detector.sigma <= 0
        or not np.isfinite(detector.h) or detector.h <= 0):
    raise ValueError('Initial 2014 CUSUM calibration is invalid')
initial_threshold = float(detector.h)
print(f'Initial CUSUM threshold: {initial_threshold:.4f}', flush=True)

print('Training initial model on data before 2015...', flush=True)
initial_fit = day < START
current_model = new_model().fit(X.loc[initial_fit], y.loc[initial_fit])

parts = [(
    t.index[val_mask], y.loc[val_mask].to_numpy(),
    val_pred, val_model.predict_proba(X.loc[val_mask])[:, 1],
)]
alarms = []
retrain_days = []
trace = []
calibration_events = []

print('Starting chronological replay...', flush=True)
for d in sorted(day.loc[(day >= START) & (day <= END)].unique()):
    d = pd.Timestamp(d)
    if d.month == 1 and d.day == 1:
        print(f'Processing {d.year}...', flush=True)

    today = day == d
    x_today = X.loc[today]
    y_today = y.loc[today].to_numpy()

    # Predict before the day's labels are observed.
    p = current_model.predict_proba(x_today)[:, 1]
    pred = (p > 0.5).astype(int)
    parts.append((t.index[today], y_today, pred, p))

    # At the end of day d, labels and the day's error become available.
    daily_error = float(np.mean(pred != y_today))
    threshold_used = float(detector.h)
    mu_used = float(detector.mu)
    sigma_used = float(detector.sigma)
    fired = detector.update(daily_error)

    # update() resets S on alarm; record threshold crossing at h for plotting.
    trace.append({
        'date': d,
        'daily_error': daily_error,
        'cusum': threshold_used if fired else float(detector.S),
        'threshold': threshold_used,
        'reference_mean': mu_used,
        'reference_std': sigma_used,
        'alarm': bool(fired),
    })

    if not fired:
        continue

    alarms.append(d)
    print(f'CUSUM alarm: {d.date()} (daily error={daily_error:.3f})', flush=True)
    next_day = d + pd.Timedelta(days=1)
    if next_day > END:
        continue

    train = day <= d
    if args.window_years is not None:
        train &= day >= next_day - pd.DateOffset(years=args.window_years)

    # Historical holdout calibration uses no observations after detection day.
    errors = historical_validation_errors(
        X, y, day, train, window_days=args.validation_days,
    )
    updated = recalibrate_after_detection(detector, errors)
    calibration_events.append({
        'date': d,
        'updated': updated,
        'reference_mean': float(detector.mu),
        'reference_std': float(detector.sigma),
        'threshold': float(detector.h),
        'validation_days': int(len(errors)),
    })
    if updated:
        print(f'Updated CUSUM: mu={detector.mu:.4f}, '
              f'sigma={detector.sigma:.4f}, h={detector.h:.4f}', flush=True)
    else:
        print('CUSUM recalibration invalid/insufficient; retaining old parameters.',
              flush=True)

    # Actual deployed model trains on all allowed history, effective next day.
    current_model = new_model().fit(X.loc[train], y.loc[train])
    retrain_days.append(next_day)
    print(f'Retrained model for use from {next_day.date()}', flush=True)

print('Saving predictions and alarm traces...', flush=True)
arrays = [np.concatenate([part[j] for part in parts]) for j in range(4)]
ev.save_predictions(
    args.name, *arrays,
    info={
        'model': args.model,
        'strategy': {
            'type': 'online_error_cusum_post_detection_recalibration',
            'window_years': args.window_years,
            'validation_days': args.validation_days,
            'alarm_dates': [str(d.date()) for d in alarms],
            'retrains': [str(d.date()) for d in retrain_days],
            'reference': '2014 initially; historical holdout after alarms',
            'initial_threshold': initial_threshold,
            'final_threshold': float(detector.h),
        },
    },
)

out_dir = ROOT / 'classification' / 'stage2' / 'alarms'
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(trace).to_csv(out_dir / f'{args.name}_trace.csv', index=False)
pd.DataFrame({'date': alarms}).to_csv(out_dir / f'{args.name}_alarms.csv', index=False)
pd.DataFrame(calibration_events).to_csv(
    out_dir / f'{args.name}_calibrations.csv', index=False,
)
print('Alarm dates:', [str(d.date()) for d in alarms])
print('Retrain start dates:', [str(d.date()) for d in retrain_days])
print('Final CUSUM threshold:', detector.h)

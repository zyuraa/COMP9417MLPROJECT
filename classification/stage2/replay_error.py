"""Online error-CUSUM-triggered adaptation replay (Python 3.11).

Place at classification/stage2/replay_error_cusum.py and run from repository root.
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logreg', 'knn'], required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--window-years', type=int, default=None)
parser.add_argument('--cooldown', type=int, default=60)
args = parser.parse_args()


def new_model():
    if args.model == 'logreg':
        # Match the hyperparameters of your selected Stage 1 model.
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    else:
        clf = KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1)
    return make_pipeline(StandardScaler(), clf)


print('Loading data and building features...', flush=True)
t = F.build_table(F.load(ev.DATA))
X = t[F.SHARED_FEATURES]
y = t['y_up'].astype(int)
day = pd.to_datetime(t['day'])

print('Calibrating CUSUM on 2014 validation errors...', flush=True)
# 2014 validation predictions, generated without using 2014 for fitting.
val_model = new_model().fit(X.loc[day.dt.year <= 2013], y.loc[day.dt.year <= 2013])
val_mask = day.dt.year == 2014
val_pred = val_model.predict(X.loc[val_mask])
val_err = pd.Series(
    (val_pred != y.loc[val_mask].to_numpy()).astype(float),
    index=day.loc[val_mask].to_numpy(),
).groupby(level=0).mean().sort_index()

detector = build_error_detector(
    val_err, k=0.5, margin=1.1,
    cooldown=args.cooldown, max_alarms=BUDGET,
)
if not np.isfinite(detector.sigma) or detector.sigma <= 0:
    raise ValueError('2014 reference error has zero/invalid standard deviation')

print(f'CUSUM threshold: {detector.h:.4f}', flush=True)
print('Training initial model on 2010–2014...', flush=True)
# The original frozen model starts with all observations through 2014.
fit_mask = day < START
current_model = new_model().fit(X.loc[fit_mask], y.loc[fit_mask])

parts = [(
    t.index[val_mask], y.loc[val_mask].to_numpy(),
    val_pred, val_model.predict_proba(X.loc[val_mask])[:, 1],
)]
alarms = []
retrain_days = []
trace = []

# Each loop iteration represents a single target day, in chronological order.
print('Starting chronological replay...', flush=True)
for d in sorted(day.loc[(day >= START) & (day <= END)].unique()):
    d = pd.Timestamp(d)
    if d.month == 1 and d.day == 1:
        print(f'Processing {d.year}...', flush=True)
    today = day == d
    x_today = X.loc[today]
    y_today = y.loc[today].to_numpy()

    # Forecast before seeing the target day's labels.
    p = current_model.predict_proba(x_today)[:, 1]
    pred = (p > 0.5).astype(int)
    parts.append((t.index[today], y_today, pred, p))

    # At end of day D, true labels become available.
    daily_error = float(np.mean(pred != y_today))
    fired = detector.update(daily_error)
    trace.append({'date': d, 'daily_error': daily_error,
                  'cusum': detector.h if fired else detector.S,
                  'alarm': bool(fired)})

    if fired:
        alarms.append(d)
        print(f'CUSUM alarm: {d.date()} (daily error={daily_error:.3f})', flush=True)
        # Retrain at end of D, for forecasting D+1 onwards.
        next_day = d + pd.Timedelta(days=1)
        if next_day <= END:
            train = day <= d
            if args.window_years is not None:
                train &= day >= next_day - pd.DateOffset(years=args.window_years)
            current_model = new_model().fit(X.loc[train], y.loc[train])
            retrain_days.append(next_day)
            print(f'Retrained model for use from {next_day.date()}', flush=True)
        # Keep the original 2014 reference after retraining, so the detector
        # has a consistent historical benchmark. CUSUM resets on each alarm.

print('Saving predictions and alarm traces...', flush=True)
arrays = [np.concatenate([part[j] for part in parts]) for j in range(4)]
ev.save_predictions(
    args.name, *arrays,
    info={'model': args.model,
          'strategy': {'type': 'online_error_cusum',
                       'window_years': args.window_years,
                       'alarm_dates': [str(d.date()) for d in alarms],
                       'retrains': [str(d.date()) for d in retrain_days],
                       'reference': '2014 validation errors',
                       'threshold': float(detector.h)}}
)

out_dir = ROOT / 'classification' / 'stage2' / 'alarms'
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(trace).to_csv(out_dir / f'{args.name}_trace.csv', index=False)
pd.DataFrame({'date': alarms}).to_csv(out_dir / f'{args.name}_alarms.csv', index=False)
print('Alarm dates:', [str(d.date()) for d in alarms])
print('Retrain start dates:', [str(d.date()) for d in retrain_days])
print('CUSUM threshold:', detector.h)

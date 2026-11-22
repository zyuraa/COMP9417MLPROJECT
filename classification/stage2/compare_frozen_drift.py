"""Plot frozen KNN drift signals and monthly balanced accuracy.
Run: python classification/stage2/compare_frozen_drift.py --predictions YOUR_FROZEN_KNN_NAME
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

CLASS_DIR = Path(__file__).resolve().parents[1]
ROOT = CLASS_DIR.parent
sys.path.insert(0, str(CLASS_DIR))
sys.path.insert(0, str(ROOT))
import evaluation as ev
from input_drift_detector.drift import market_day

ap = argparse.ArgumentParser()
ap.add_argument('--predictions', required=True, help='Stem of classification/predictions/<name>.parquet')
args = ap.parse_args()
alarms = CLASS_DIR / 'stage2' / 'alarms'
figures = CLASS_DIR / 'figures'
figures.mkdir(parents=True, exist_ok=True)

input_trace = pd.read_csv(alarms / 'knn_frozen_input_trace.csv', parse_dates=['date'])
error_trace = pd.read_csv(alarms / 'knn_frozen_cusum_trace.csv', parse_dates=['date'])
pred_path = CLASS_DIR / 'predictions' / (args.predictions + '.parquet')
pred = pd.read_parquet(pred_path)
pred = pred.loc[pred['split'] == 'frozen'].copy()
pred['day'] = market_day(pred.index)
pred['month'] = pred['day'].dt.to_period('M')
monthly = []
for m, group in pred.groupby('month'):
    if group['y_true'].nunique() < 2:
        ba = np.nan
    else:
        ba = balanced_accuracy_score(group['y_true'], group['y_pred'])
    monthly.append({'month': m.to_timestamp(), 'balanced_accuracy': ba, 'n': len(group)})
monthly = pd.DataFrame(monthly)
monthly.to_csv(alarms / 'knn_frozen_monthly_balanced_accuracy.csv', index=False)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, trace, title in [(axes[0], input_trace, 'Input drift (seasonally adjusted score + CUSUM)'),
                         (axes[1], error_trace, 'Error drift (frozen KNN CUSUM)')]:
    ax.plot(trace.date, trace.cusum, label='CUSUM')
    ax.plot(trace.date, trace.threshold, '--', label='2014-calibrated threshold')
    events = trace.loc[trace.alarm.astype(bool), 'date']
    for d in events:
        ax.axvline(d, alpha=0.18)
    ax.set_title(f'{title} — {len(events)} alarms')
    ax.set_ylabel('CUSUM')
    ax.legend()
    ax.grid(alpha=0.2)
axes[2].plot(monthly.month, monthly.balanced_accuracy, label='Monthly balanced accuracy')
for d in error_trace.loc[error_trace.alarm.astype(bool), 'date']:
    axes[2].axvline(d, alpha=0.18)
axes[2].set_ylabel('Balanced accuracy')
axes[2].set_xlabel('Date')
axes[2].set_ylim(0, 1)
axes[2].grid(alpha=0.2)
fig.tight_layout()
output = figures / 'knn_frozen_drift_comparison.png'
fig.savefig(output, dpi=180)
plt.close(fig)
print('Saved', output)
print('Input alarms:', input_trace.loc[input_trace.alarm.astype(bool), 'date'].dt.strftime('%Y-%m-%d').tolist())
print('Error alarms:', error_trace.loc[error_trace.alarm.astype(bool), 'date'].dt.strftime('%Y-%m-%d').tolist())

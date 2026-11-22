"""Four-regime analysis for frozen KNN price-direction classification.

Run from repository root:
 python classification/stage2/analyse_knn_drift_regimes.py --predictions knn

Requires outputs from run_frozen_input.py and a frozen KNN parquet with
2014 validation and 2015-2026 frozen predictions.

Regimes use 30-day rolling means. Input threshold = 95th percentile of
2014 rolling input scores. Error threshold = 95th percentile of 2014
rolling daily classification errors. Error regime at day D uses errors
through D-1, not D, to avoid look-ahead in grouping performance.
These are descriptive regimes, not additional retraining alarms.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASS_DIR = Path(__file__).resolve().parents[1]
ROOT = CLASS_DIR.parent
sys.path.insert(0, str(CLASS_DIR))
import evaluation as ev

p = argparse.ArgumentParser()
p.add_argument('--predictions', default='knn')
p.add_argument('--window', type=int, default=30)
p.add_argument('--quantile', type=float, default=0.95)
args = p.parse_args()
if args.window < 2 or not 0 < args.quantile < 1:
    p.error('--window must be >=2 and --quantile between 0 and 1')

out = CLASS_DIR / 'stage2' / 'alarms'
fig_dir = CLASS_DIR / 'figures'
out.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# Use saved input score. The input score is model-independent and the
# original script saves only 2015 onward, so reconstruct 2014 using the
# same 2010-2013 fitted scorer (without future data).
from input_drift_detector.drift import daily_input_features, InputScore, market_day
market = ev.load_df()
features = daily_input_features(market).replace([np.inf, -np.inf], np.nan).dropna()
ref = features.loc['2010':'2013']
scorer = InputScore().fit(ref)
input_scores = scorer.score(features).rename('input_score').sort_index()

# 2014 validation and 2015-2026 frozen predictions from the same file.
pred_path = CLASS_DIR / 'predictions' / f'{args.predictions}.parquet'
if not pred_path.exists():
    raise FileNotFoundError(f'Missing {pred_path}. Run classification/knn.py first.')
pred = pd.read_parquet(pred_path).sort_index()
pred['day'] = market_day(pred.index)
pred = pred[(pred['day'] >= '2014-01-01') & (pred['day'] <= '2026-10-31')].copy()
if not {'y_true','y_pred','split'}.issubset(pred.columns):
    raise ValueError('Prediction file must contain y_true, y_pred, split')
val = pred[pred['day'].dt.year == 2014]
test = pred[(pred['day'] >= '2015-01-01') & (pred['day'] <= '2026-10-31')]
if val.empty or test.empty:
    raise ValueError('Need 2014 validation and 2015-2026 frozen predictions')
if not val['split'].eq('val').all() or not test['split'].eq('frozen').all():
    raise ValueError('Unexpected split labels in prediction file')

pred['wrong'] = (pred.y_true != pred.y_pred).astype(float)
daily_error = pred.groupby('day')['wrong'].mean().rename('daily_error')

# Rolling scores are computed causally. Input score for D is observed by
# end of D. The error statistic used to assign D is shifted one day.
input_roll = input_scores.rolling(args.window, min_periods=args.window).mean().rename('input_rolling')
error_roll = daily_error.rolling(args.window, min_periods=args.window).mean().rename('error_rolling')
input_threshold = float(input_roll.loc['2014'].dropna().quantile(args.quantile))
error_threshold = float(error_roll.loc['2014'].dropna().quantile(args.quantile))
if not np.isfinite(input_threshold) or not np.isfinite(error_threshold):
    raise ValueError('Insufficient 2014 data for rolling thresholds')

# For D use input history through D-1 as well, ensuring both regime flags
# are known BEFORE day D forecasts. This is stricter than end-of-day detection.
regime_days = pd.DataFrame(index=pd.date_range('2015-01-01', '2026-10-31', freq='D'))
regime_days.index.name = 'day'
regime_days['input_rolling'] = input_roll.shift(1).reindex(regime_days.index)
regime_days['error_rolling'] = error_roll.shift(1).reindex(regime_days.index)
regime_days['input_shift'] = regime_days.input_rolling > input_threshold
regime_days['error_shift'] = regime_days.error_rolling > error_threshold
regime_days['regime'] = np.select(
    [regime_days.input_shift & regime_days.error_shift,
     regime_days.input_shift & ~regime_days.error_shift,
     ~regime_days.input_shift & regime_days.error_shift],
    ['4. Both shifts', '2. Input shift only', '3. Error increase only'],
    default='1. Normal')
regime_days.loc[regime_days[['input_rolling','error_rolling']].isna().any(axis=1), 'regime'] = np.nan

# Attach regimes to individual half-hour predictions.
joined = test.join(regime_days[['regime']], on='day')
joined = joined.dropna(subset=['regime'])
order = ['1. Normal','2. Input shift only','3. Error increase only','4. Both shifts']
rows = []
for name in order:
    g = joined[joined.regime == name]
    if g.empty:
        rows.append(dict(regime=name, days=0, observations=0, accuracy=np.nan,
                         balanced_accuracy=np.nan, class1_share=np.nan))
        continue
    ba = balanced_accuracy_score(g.y_true, g.y_pred) if g.y_true.nunique() == 2 else np.nan
    rows.append(dict(regime=name, days=g.day.nunique(), observations=len(g),
                     accuracy=(g.y_true == g.y_pred).mean(),
                     balanced_accuracy=ba, class1_share=g.y_true.mean()))
summary = pd.DataFrame(rows)
summary['pct_days'] = 100 * summary.days / regime_days.regime.notna().sum()
summary.to_csv(out / 'knn_drift_regime_summary.csv', index=False)
regime_days.to_csv(out / 'knn_drift_regime_daily.csv')

# Monthly balanced accuracy within each regime. Empty/single-class
# groups are omitted, rather than pretending BA is meaningful.
monthly_rows = []
for (month, regime), g in joined.groupby([joined.day.dt.to_period('M'), 'regime']):
    monthly_rows.append({'month': month.to_timestamp(), 'regime': regime,
                         'balanced_accuracy': balanced_accuracy_score(g.y_true, g.y_pred)
                         if g.y_true.nunique() == 2 else np.nan,
                         'n': len(g)})
monthly = pd.DataFrame(monthly_rows)
monthly.to_csv(out / 'knn_drift_regime_monthly.csv', index=False)

# Figure 1: summary grouped bars.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(order))
axes[0].bar(x, summary.balanced_accuracy, label='Balanced accuracy')
axes[0].plot(x, summary.accuracy, 'ko', label='Ordinary accuracy')
axes[0].set_xticks(x, ['Normal', 'Input only', 'Error only', 'Both'])
axes[0].set_ylim(0.5, 1)
axes[0].set_ylabel('Score')
axes[0].set_title('Frozen KNN performance by drift regime')
axes[0].legend()
axes[0].grid(axis='y', alpha=.2)
axes[1].bar(x, summary.days)
axes[1].set_xticks(x, ['Normal', 'Input only', 'Error only', 'Both'])
axes[1].set_ylabel('Number of days')
axes[1].set_title('Time spent in each regime')
axes[1].grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(fig_dir / 'knn_drift_regime_summary.png', dpi=180)
plt.close(fig)

# Figure 2: timeline with monthly overall balanced accuracy and regimes.
monthly_all = []
for month, g in test.groupby(test.day.dt.to_period('M')):
    monthly_all.append((month.to_timestamp(), balanced_accuracy_score(g.y_true,g.y_pred)
                        if g.y_true.nunique() == 2 else np.nan))
monthly_all = pd.DataFrame(monthly_all, columns=['month','ba'])
fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                       gridspec_kw={'height_ratios':[1, 2]})
ax[0].plot(monthly_all.month, monthly_all.ba, linewidth=1.5)
ax[0].set_ylabel('Balanced accuracy')
ax[0].set_ylim(0.5,1)
ax[0].set_title('Frozen KNN performance and historical drift regimes')
ax[0].grid(alpha=.2)
code = regime_days.regime.map({r:i for i,r in enumerate(order)})
ax[1].scatter(regime_days.index, code, c=code, cmap='viridis', vmin=0, vmax=3,
              marker='|', s=30)
ax[1].set_yticks(range(4), ['Normal','Input only','Error only','Both'])
ax[1].set_ylabel('Regime (prior information)')
ax[1].set_xlabel('Date')
ax[1].grid(axis='x', alpha=.2)
fig.tight_layout()
fig.savefig(fig_dir / 'knn_drift_regime_timeline.png', dpi=180)
plt.close(fig)

print(f'Input threshold (2014, {args.window}d mean, q={args.quantile}): {input_threshold:.4f}')
print(f'Error threshold (2014, {args.window}d mean, q={args.quantile}): {error_threshold:.4f}')
print(summary.to_string(index=False, float_format=lambda v:f'{v:.4f}'))
print('Saved CSVs to:', out)
print('Saved plots to:', fig_dir)
print('CAUTION: Regimes are descriptive; error regime is defined from prior model errors.')

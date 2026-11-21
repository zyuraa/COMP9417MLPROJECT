"""Frozen input-based drift detection. Run from repository root."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

CLASS_DIR = Path(__file__).resolve().parents[1]
ROOT = CLASS_DIR.parent
sys.path.insert(0, str(CLASS_DIR))
sys.path.insert(0, str(ROOT))
import evaluation as ev
from input_drift_detector.drift import daily_input_features, build_input_detector

START, END = '2015-01-01', '2026-10-31'
OUT = CLASS_DIR / 'stage2' / 'alarms'


def main():
    df = ev.load_df()
    feats = daily_input_features(df).replace([np.inf, -np.inf], np.nan).dropna()
    scorer, detector, scores = build_input_detector(
        feats, k=0.5, margin=1.1, cooldown=60, max_alarms=10**9
    )
    if not np.isfinite(detector.h) or detector.h <= 0 or not np.isfinite(detector.sigma) or detector.sigma <= 0:
        raise ValueError('Invalid input drift calibration; inspect 2010–2014 data')
    print(f'Input CUSUM reference mean={detector.mu:.4f}, sd={detector.sigma:.4f}, threshold={detector.h:.4f}')
    rows = []
    for d, score in scores.loc[START:END].items():
        # Save pre-reset score for truthful visualisation.
        pre_reset = max(0., detector.S + (float(score)-detector.mu)/detector.sigma-detector.k)
        fired = detector.update(float(score))
        rows.append({'date': d, 'input_score': float(score), 'cusum': pre_reset,
                     'threshold': float(detector.h), 'alarm': bool(fired)})
    OUT.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(rows)
    trace.to_csv(OUT / 'knn_frozen_input_trace.csv', index=False)
    trace.loc[trace.alarm, ['date']].to_csv(OUT / 'knn_frozen_input_alarms.csv', index=False)
    z = scorer.zscores(feats.loc[START:END])
    z.pow(2).to_csv(OUT / 'knn_frozen_input_feature_contributions.csv', index_label='date')
    print(f'Saved {int(trace.alarm.sum())} input alarms to {OUT}')


if __name__ == '__main__':
    main()
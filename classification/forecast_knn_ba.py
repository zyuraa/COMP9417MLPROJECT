"""
Forecasts 7–13 Dec 2026 price-direction balanced accuracy.

"""
import argparse
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

from train import make_training_data

HERE = Path(__file__).resolve().parent
TARGET_START = pd.Timestamp('2026-12-07')
TARGET_END = pd.Timestamp('2026-12-13')
FINAL_CUTOFF = pd.Timestamp('2026-10-31')
LOOKBACKS = (4, 8, 12, 20)
MIN_CALIBRATION = 12


def complete_market_days(day):
    counts = day.value_counts()
    return pd.DatetimeIndex(counts[counts == 48].index).sort_values()


def training_mask(day, complete_days, cutoff):
    start = cutoff + pd.Timedelta(days=1) - pd.DateOffset(years=4)
    return (day >= start) & (day <= cutoff) & day.isin(complete_days)


def score_week(model, X, y, day, complete_days, start):
    end = start + pd.Timedelta(days=6)
    test_days = pd.date_range(start, end, freq='D')
    if not test_days.isin(complete_days).all():
        return None
    mask = (day >= start) & (day <= end)
    if mask.sum() != 7 * 48:
        return None
    truth = y.loc[mask].to_numpy()
    if len(np.unique(truth)) != 2:
        return None
    pred = model.predict(X.loc[mask])
    return float(balanced_accuracy_score(truth, pred))


def first_monday_after_gap(cutoff, gap_days=37):
    earliest = cutoff + pd.Timedelta(days=gap_days)
    return earliest + pd.Timedelta(days=(7 - earliest.weekday()) % 7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=HERE.parent / 'data' / 'nem_weather_halfhourly_2010_2026.csv')
    p.add_argument('--joblib', type=Path, default=HERE / 'artifacts' / 'price_model.joblib')
    p.add_argument('--out', type=Path, default=HERE / 'figures')
    p.add_argument('--step-months', type=int, default=2, help='Spacing of historical backtest cutoffs')
    args = p.parse_args()
    if args.step_months < 1:
        p.error('--step-months must be positive')
    if not args.csv.is_file() or not args.joblib.is_file():
        p.error(f'Missing CSV ({args.csv}) or trained KNN ({args.joblib})')

    final_model = joblib.load(args.joblib)
    if not hasattr(final_model, 'predict') or not hasattr(final_model, 'named_steps'):
        raise ValueError('Saved artifact must be a fitted sklearn KNN pipeline')
    if 'kneighborsclassifier' not in final_model.named_steps:
        raise ValueError('Saved pipeline is not the expected KNeighborsClassifier')
    print('Saved final KNN:', final_model)
    print('The joblib is used as the template; it is NOT retrained or overwritten.')

    df = pd.read_csv(args.csv, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    df = df.loc[(df.index - pd.Timedelta(minutes=30)).normalize() <= FINAL_CUTOFF]
    X, y, day = make_training_data(df)
    complete = complete_market_days(day)
    if FINAL_CUTOFF not in complete:
        raise ValueError('31 October 2026 is not a complete market day in this CSV')
    if len(X.columns) != len(final_model.feature_names_in_) or list(X.columns) != list(final_model.feature_names_in_):
        raise ValueError('CSV feature columns do not match the saved KNN; check train.py/model.py versions')

    final_mask = training_mask(day, complete, FINAL_CUTOFF)
    expected_n = int(final_mask.sum())
    actual_n = len(final_model.named_steps['kneighborsclassifier']._fit_X)
    if actual_n != expected_n:
        warnings.warn(f'Saved KNN has {actual_n} training rows but the expected Oct-2026 four-year window has {expected_n}. '
                      'Check whether the saved model was trained on this CSV and cutoff.')
    print(f'Expected final training window: {day.loc[final_mask].min().date()} to {FINAL_CUTOFF.date()} '
          f'({expected_n:,} half-hourly rows); saved model contains {actual_n:,} rows')

    # Every two months, recreate an earlier training cutoff and evaluate a full
    # week at least 37 days later. All test labels are strictly after training.
    cutoffs = pd.date_range('2014-10-31', '2026-09-30', freq='ME')
    cutoffs = cutoffs[::args.step_months]
    results = []
    for cutoff in cutoffs:
        test_start = first_monday_after_gap(cutoff)
        if test_start + pd.Timedelta(days=6) > FINAL_CUTOFF:
            continue
        fit = training_mask(day, complete, cutoff)
        if fit.sum() < 365 * 48 * 3:  # at least three years of valid rows
            continue
        model = clone(final_model)
        model.fit(X.loc[fit], y.loc[fit])
        ba = score_week(model, X, y, day, complete, test_start)
        if ba is not None:
            results.append({'cutoff': cutoff, 'test_start': test_start,
                            'test_end': test_start + pd.Timedelta(days=6),
                            'train_rows': int(fit.sum()), 'balanced_accuracy': ba})
            print(f'{cutoff.date()} -> {test_start.date()}: BA={ba:.4f}')
    hist = pd.DataFrame(results).sort_values('cutoff').reset_index(drop=True)
    if len(hist) < MIN_CALIBRATION + max(LOOKBACKS):
        raise ValueError(f'Only {len(hist)} backtests; need at least {MIN_CALIBRATION + max(LOOKBACKS)}. '
                         'Try --step-months 1 or check data completeness.')

    values = hist['balanced_accuracy'].to_numpy()
    evaluations = []
    for k in LOOKBACKS:
        errors = []
        for i in range(max(k, MIN_CALIBRATION), len(values)):
            estimate = values[i-k:i].mean()
            errors.append(values[i] - estimate)
        errors = np.asarray(errors)
        evaluations.append({'lookback': k, 'n_errors': len(errors),
                            'mae': float(np.mean(np.abs(errors)))})
    ranking = pd.DataFrame(evaluations).sort_values(['mae', 'lookback'])
    best_k = int(ranking.iloc[0]['lookback'])
    # Residuals for the selected lookback; quantiles provide approximate
    # forecast intervals, not guaranteed calibrated coverage.
    residuals = np.asarray([values[i] - values[i-best_k:i].mean()
                            for i in range(max(best_k, MIN_CALIBRATION), len(values))])
    point = float(values[-best_k:].mean())
    pi80 = np.clip(point + np.quantile(residuals, [0.10, 0.90]), 0, 1)
    pi95 = np.clip(point + np.quantile(residuals, [0.025, 0.975]), 0, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    hist.to_csv(args.out / 'knn_4year_backtests.csv', index=False)
    ranking.to_csv(args.out / 'knn_4year_lookback_comparison.csv', index=False)
    summary = pd.DataFrame([{'target_start': TARGET_START.date(),
                             'target_end': TARGET_END.date(), 'final_cutoff': FINAL_CUTOFF.date(),
                             'n_backtests': len(hist), 'lookback': best_k,
                             'forecast_ba': point, 'pi80_low': pi80[0], 'pi80_high': pi80[1],
                             'pi95_low': pi95[0], 'pi95_high': pi95[1]}])
    summary.to_csv(args.out / 'knn_4year_forecast.csv', index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist['test_start'], 100 * hist['balanced_accuracy'], 'o-', ms=3,
            label='Historical out-of-sample weekly BA')
    ax.errorbar([TARGET_START], [100 * point],
                yerr=[[100 * (point-pi80[0])], [100 * (pi80[1]-point)]],
                fmt='o', capsize=6, label='Dec 2026 forecast (80% PI)')
    ax.axhline(50, ls=':', color='gray')
    ax.set(xlabel='Evaluation week starting', ylabel='Balanced accuracy (%)',
           title='Four-year rolling KNN: historical BA and Dec 2026 forecast', ylim=(0, 100))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / 'knn_4year_ba_forecast.png', dpi=180)
    plt.close(fig)

    print('\nLookback comparisons (historical out-of-sample forecast MAE):')
    print(ranking.to_string(index=False))
    print(f'\nTarget: 7–13 Dec 2026 | historical backtests: {len(hist)}')
    print(f'Forecast BA: {point:.4f} ({100*point:.2f}%)')
    print(f'Approximate 80% PI: [{100*pi80[0]:.2f}%, {100*pi80[1]:.2f}%]')
    print(f'Approximate 95% PI: [{100*pi95[0]:.2f}%, {100*pi95[1]:.2f}%]')
    print('WARNING: historical tests use actual observed weather; live forecasts may use predicted weather.')
    print('WARNING: intervals are empirical and uncalibrated; lookback selection uses the same history.')
    print('Outputs:', args.out.resolve())


if __name__ == '__main__':
    main()

# NSW demand forecasting with gradient boosting

Run these commands from the repository root in the project's pinned Python 3.11 environment.

```powershell
python boosting_package/train.py data/nem_weather_halfhourly_2010_2026.csv
python boosting_package/evaluate.py data/nem_weather_halfhourly_2010_2026.csv --plots
```

## Stage 1

The training script fits histogram gradient boosting on 2010–2013 and ranks candidates by mean absolute percentage error (MAPE), with mean absolute error (MAE) as the tie-breaker. Every candidate and both course baselines use the same 17,520 half-hours of 2014. See `results/stage1_2014.csv` for the measured scores and the selected row.

Feature selection starts with yesterday's and last week's demand and the market calendar, then adds Sydney weather, recent demand summaries, and daily profile features. Each step keeps the previous features. The daily profile group uses the demand observed at the issue time and a same-slot average from previous working days or non-working days, whichever matches the forecast day. All forecast-day weather comes from the 48 supplied observations.

After choosing features, the script compares predicting demand directly with predicting a correction to a lagged demand forecast. For example, a target of `demand_lag1` means the trees learn today's demand minus yesterday's same-slot demand, and prediction adds yesterday's demand back. The final trials vary tree size and boosting rounds. The selected estimator is refitted once on 2010–2014 and saved in `artifacts/demand_model.joblib`.

## Frozen historical evaluation

`evaluate.py` uses that saved estimator throughout January 2015–October 2026. The monthly and yearly CSVs report MAPE (%) and MAE (MW) for boosting and both course baselines. Monthly MAPE includes a normal interval assuming independent half-hours and a percentile interval from 1,000 whole-day bootstrap resamples. The `--plots` option produces a monthly comparison and a seven-day observed-versus-predicted chart. The monthly comparison also shows Bryan's frozen Lasso when his published results are present in the repository.

| Demand forecast | 2014 MAPE (%) | 2014 MAE (MW) | Frozen 2015–Oct 2026 MAPE (%) | Frozen MAE (MW) |
|---|---:|---:|---:|---:|
| Gradient boosting | 2.005 | 160.6 | 3.623 | 269.1 |
| Bryan's per-slot Lasso | 1.741 | 140.9 | 5.162 | 357.5 |
| Same half-hour yesterday | 4.935 | 394.4 | 6.150 | 470.6 |
| Same half-hour last week | 4.934 | 401.3 | 7.422 | 569.7 |

The historical averages weight all 207,456 half-hours equally. The Lasso values come from tables 7 and 8 in `bryan_package/analysis_outputs/`.

## Using the demand model

`model.Forecaster` implements the kit's 48-row interface. Its price-direction column uses the same-half-hour-yesterday baseline. To integrate demand with the group's price classifier, load the demand artifact with `load_demand_model`, call `forecast_features(history, weather_next)`, and pass those features to the demand model's `predict` method.

Use the original course kit's checker when testing the standalone package. From the extracted original kit, run `python check_submission.py <path-to-boosting_package>`.

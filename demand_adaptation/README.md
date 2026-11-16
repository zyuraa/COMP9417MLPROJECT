# Demand retraining experiment

Run from the repository root using the Python 3.11 environment in the kit.

```powershell
python -m demand_adaptation.run data/nem_weather_halfhourly_2010_2026.csv
```

The command rebuilds the historical comparison, figure and demand candidate in `demand_adaptation/results/`. Use `--no-plots` to skip rendering the figure.

## Experimental choices

- The Stage 1 selection metric is 2014 demand MAPE. Bryan's per-slot Lasso scored 1.741% and Yan's histogram gradient boosting scored 2.005%. All three adaptation runs use the same Lasso, initial fit through 2014, 65 selected features and fixed regularisation.
- The frozen run uses the initial model for all 4,322 subsequent market days. Each adapted run refits its 48 scalers and 48 regressors on the preceding four calendar years at each replacement. Both retain the day-type transformation fitted through 2014 so the selected cluster indicators keep a consistent meaning.
- The fixed schedule deploys on 1 January of 2016, 2018, 2020, 2022, 2024 and 2026. Each refit includes data through the preceding market day. The alarm policy has the same six-replacement budget.
- The supplied **reference error alarm** observes the deployed model's MAPE after each day closes. Its statistic is the mean of the latest 28 daily MAPEs, seeded with the end of 2014. Its threshold of 2.174% is the 95th percentile of 28-day means from the 2014 validation forecasts. Those forecasts use a Lasso and preprocessing fitted through 2013. Stage 1 used 2014 to choose features and regularisation. The threshold and 365-day cooldown are fixed before the historical simulation.
- A signal on day D can affect the forecast for D+1. Once the six refits have been used, the log continues recording subsequent alarms. All candidate scores use the same 207,456 half-hours from January 2015 to October 2026.

## Saved results

| Strategy | Retrains | MAPE (%) | MAE (MW) |
|---|---:|---:|---:|
| Frozen Lasso | 0 | 5.162 | 357.5 |
| Scheduled Lasso | 6 | 2.983 | 222.0 |
| Reference alarm Lasso | 6 | 4.146 | 288.9 |
| Same half-hour yesterday | 0 | 6.150 | 470.6 |
| Same half-hour last week | 0 | 7.422 | 569.7 |

The reference alarm's sixth replacement took effect on 1 January 2020. The scheduled run still had three replacements available after that date. The decision, training and first forecast dates appear in `refits.csv`. Daily signals appear in `alarm_history.csv`. The monthly results include both 95% interval calculations, and the figure shows the whole-day bootstrap bands.

## Using the team's detector

To use independently generated end-of-day alarms, supply a CSV with a column named `alarm_date`, one unique ISO date per row in chronological order. An empty list with the header is valid. The first six eligible signals deploy a refit on the following day. An input detector can supply dates directly. For an error detector that depends on which model is active, generate the signals online from that policy's forecasts, as the reference implementation does in `simulate`.

```powershell
python -m demand_adaptation.run data/nem_weather_halfhourly_2010_2026.csv --alarm-dates leon_alarm_dates.csv
```

This run writes to `results_external/` so the reference comparison stays available.

## Final demand candidate

The exporter chooses the lowest overall simulation MAPE among the three Lasso strategies, breaking ties by MAE and then the number of replacements. The historical scores describe the data used to make that choice. For the selected scheduled strategy, the final model uses demand targets from 1 November 2022 through 31 October 2026. This deployment fit is separate from the six replacements in the historical simulation.

`results/demand_candidate.zip` contains the fitted demand estimator, its feature code and a standalone `train.py` that rebuilds the artifacts from the released CSV. From an unpacked candidate, run `python train.py <path-to-released-csv>`. Its price column uses the course's yesterday baseline. The group can integrate its selected price classifier when assembling the final joint submission.

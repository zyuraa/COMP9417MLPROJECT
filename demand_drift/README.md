# Demand drift experiment

Run the historical simulation from the repository root in the course Python environment.

```powershell
python -m demand_adaptation.run data/nem_weather_halfhourly_2010_2026.csv --profile-drift
```

The command writes to `demand_drift/results/`. Use `--no-plots` for tables only. All comparisons use the same 4,322 NSW market days from 1 January 2015 through 31 October 2026.

## What each detector sees

- **Input detector.** For a forecast on day D, take the 48 observed half-hours from D−1 and divide by that day's mean demand. This captures changes in the shape of the day and makes the score insensitive to uniform changes in level. Fit PCA (five components) and four K-means centres on 2010–2013 profiles, following the group's [day-type analysis](../bryan_package/analysis_outputs/fig2_day_type_clustering_and_drift.png). For each subsequent profile, measure the distance to the nearest centre and divide by the 2010–2013 mean distance for its calendar month. A ratio of 1 means a typical training-period distance for that month.
- **Error detector.** After D closes, compute mean absolute percentage error (MAPE) across its 48 demand forecasts. Use a fixed Lasso trained through 2014 for the diagnostic timeline. Keeping that model fixed lets later errors be compared on the same basis. The retraining experiment separately monitors the model deployed by its error-triggered policy, because a refit can change its future errors.
- **Threshold and timing.** Each statistic is a trailing 28-day mean. Set each threshold to the 95th percentile of the available 28-day means in 2014. For input calibration, 2 January through 31 December use profiles observed in 2014. For error calibration, 2014 predictions come from a Lasso fitted through 2013. Both rolling windows carry their recent 2014 history into January 2015. The fixed thresholds are **1.2319** for input distance and **2.1737%** for error. Each detector fires on a strict exceedance when at least 365 days have passed since its previous alarm. Both policies make the decision at the close of D, so a refit first affects D+1.

## Observed timing

The first five principal components capture 96.24% of variation in the normalised 2010–2013 training profiles. The four centres distinguish an afternoon peak, two evening peaks with different morning rises, and a flatter daytime profile. The group's four-cluster choice preserves these distinctions. A two-cluster fit had a higher training silhouette score (0.410 versus 0.307).

| Status of the two 28-day statistics | Market days |
|---|---:|
| Both above threshold | 3,342 |
| Only the input statistic above threshold | 400 |
| Only the frozen error statistic above threshold | 188 |
| Both at or below threshold | 392 |

The first error alarm was on 1 January 2015, when the 28-day MAPE was 3.04%. Its window included December 2014. The first input alarm was on 8 June 2015. Its distance ratio was 1.240 while the error statistic was still 1.73%, below its threshold. Both detectors issued 12 alarms. The long stretches above threshold and the 365-day cooldown account for the roughly annual spacing of later alarms. Interpreting every alarm as a separate change in the market would overstate the evidence.

Group day types also changed in frequency. The slow-morning, evening-peak profile (cluster 3) rose from 24.66% of forecast days in 2015 to 65.75% in 2025. The flatter daytime profile (cluster 4) fell from 25.21% to 0.27%. Frozen demand MAPE rose from 1.94% to 10.40% across the same two complete years. Changes in the mix of profiles can make an old model less accurate. Establishing that the relationship between the same inputs and demand changed would require further evidence.

The operator recorded a NSW operational demand minimum at noon on 16 February 2025 and attributed it to strong distributed solar output and low cooling demand. When this day's profile entered our detector on 17 February, its individual distance was 9.59 times the February reference average. The 28-day input ratio was 3.54 and frozen MAPE was 10.73% over that window. Both were already elevated, and neither detector issued a fresh alarm that day. The operator's account supplies a plausible explanation for unusual daytime shapes. The detector itself cannot identify their cause. [AEMO Quarterly Energy Dynamics, Q1 2025, p. 13](https://www.aemo.com.au/-/media/files/major-publications/qed/2025/qed-q1-2025.pdf#page=13)

## What happened when an alarm triggered retraining

Each adaptive policy started with the same Stage 1 Lasso, kept its selected features and regularisation, and could use six four-year-window refits. The scheduled policy deployed a new fit every two years from January 2016.

| Policy | Refits | MAPE (%) | MAE (MW) |
|---|---:|---:|---:|
| Frozen | 0 | 5.162 | 357.5 |
| Fixed schedule | 6 | 2.983 | 222.0 |
| Input alarm | 6 | 3.932 | 276.0 |
| Error alarm | 6 | 4.146 | 288.9 |

Both alarm policies improved on the frozen model. The error policy used its sixth refit on 1 January 2020 and the input policy on 23 June 2020 (effective dates). The schedule spread its six refits through January 2026 and had the lowest error on this historical comparison.

## Outputs

- [Detector timeline](results/drift_timeline.png) and [adaptation comparison](results/demand_adaptation.png)
- [Alarm dates and threshold values](results/alarm_dates.csv), [daily detector readings](results/daily_diagnostics.csv) and [monthly day-type shares and error](results/monthly_diagnostics.csv)
- [Input alarm dates in the shared retraining format](results/input_alarm_dates.csv), [actual refit dates](results/refits.csv) and [errors observed by the adaptive error policy](results/alarm_history.csv)
- [Overall policy results and both official baselines](results/overall.csv) and [monthly confidence intervals](results/monthly.csv)

# COMP9417 Project Handoff — `bryan_package/`

## Scope & Role
- **Focus:** NSW Day-Ahead Electricity **Demand** Forecasting (`nsw_demand_mw`, evaluated by MAPE % and MAE MW) using Regularised Linear Regression.
- **Price Direction (`nsw_price_up`):** Handled in `Forecaster.predict()` via the Same-Half-Hour-Yesterday baseline so `check_submission.py` passes end-to-end; teammates handle ML price-direction modelling.
- **Dataset:** `data/nem_weather_halfhourly_2010_2026.csv` (half-hourly 2010-01-01 to 2026-10-31; day $D$ = $D\text{ 00:30}$ to $(D+1)\text{ 00:00}$).

## Package Layout (`bryan_package/`)
- `features.py`: `DayTypeTransformer` (PCA `n_components=5`, K-Means `n_clusters=4` on mean-normalised profiles), `build_candidate_features()` (79 slot-aligned features across 8 groups), `build_single_day_features()`, and `FeatureSelector` (group ablation + hybrid $0.85\,L_1 + 0.15\,\log(1+F_{\text{residual}})$ ranking).
- `model.py`: `PerSlotLinearRegressor` (48 half-hour slot-specific `StandardScaler` + `Ridge`/`Lasso`/`ElasticNet`) and `Forecaster` (`__init__`, `predict`).
- `train.py`: Stage 1 pipeline (`2010–2013` train $\rightarrow$ `2014` val $\rightarrow$ refit on `2010–2014`), saves `artifacts/demand_model.joblib` and `artifacts/config.json`.
- `data_analysis.py`: Summary stats (`table1`, `table2`, `table2b`), duck curve & weather (`fig1`), day-type clustering (`fig2`), baseline CI (`fig3`), demand/price patterns (`fig4`).
- `stage2_frozen_simulation.py`: Stage 2 frozen simulation (`Jan 2015 – Oct 2026`, 4,322 days / 142 months) + Appendix A 95% CIs (Independent Normal vs. Day-Block Bootstrap, `n_boot=1000`). Exports `table8`, `table9`, `stage2_ci_dependence_stats.json`, `fig7`, `fig8`.
- `artifacts/` & `analysis_outputs/`: Pre-generated Stage 1 & Stage 2 outputs.

## Completed Results (Verified)
### Stage 1 (2014 Validation)
- **Baselines:** Same HH Yesterday = `4.9350%` MAPE (`394.42 MW`); Same HH Last Week = `4.9339%` MAPE (`401.26 MW`).
- **Feature Selection:** Pruned $79 \rightarrow k^*=65$ features (`table4`, `table5`, `fig5`).
- **Models (`table7`, `fig6`):**
  - Flat Daily Multi-Output Ridge (403 feats, $\alpha=1.0$): `1.9237%` (`154.65 MW`)
  - Per-Slot Ridge ($\alpha=5.0$): `1.7491%` (`141.46 MW`)
  - Per-Slot ElasticNet ($\alpha=0.005, l_1=0.5$): `1.7489%` (`141.47 MW`)
  - **Per-Slot Lasso ($\alpha=1.0$, Selected):** **`1.7408%`** (`140.90 MW`, $+64.73\%$ vs. Yesterday Baseline). Refitted on `2010–2014`.

### Stage 2 Frozen Simulation (Jan 2015 – Oct 2026)
- **Annual Degradation (`table8`, `fig7`, `fig8`):** Frozen Per-Slot Lasso beats Yesterday Baseline for `2015–2022` (`1.937%` in 2015 $\rightarrow$ `5.605%` vs `5.888%` in 2022), then crosses above in **2023** (`7.378%` vs `6.457%`) and peaks in **2025** (`10.396%`; overall 2015–2026 = `5.162%`). Degradation is concentrated at **08:00–16:00 AEST** (midday rooftop solar duck curve, $>+500\text{ MW}$ positive bias).
- **Appendix A 95% CIs (`table9`, `stage2_ci_dependence_stats.json`):** Mean CI width for Per-Slot Lasso is `0.587%` (Independent) vs `1.463%` (Day-Block Bootstrap, **$2.66\times$** wider) due to within-day error correlation ($\text{ICC}=0.272$, $\text{DEFF}=13.80 \implies \sim 3.48$ effective independent half-hours/day; $r(1\text{ hh})=0.966$). Across-day daily error ACF is $r(1\text{ day})=0.758$.

## Next Steps (Remaining Spec Requirements)
1. **Online Drift Detection (Stage 2):** Implement 1 input-based detector (e.g., PCA cluster distance / midday covariate shift) and 1 error-based detector (e.g., Page-Hinkley / EWMA on daily errors), calibrated strictly on `2010–2014`, logging alarm dates and linking to NEM events.
2. **Model Adaptation (Stage 2):** Compare $\ge 3$ strategies on `Per-Slot Lasso` within the **budget of $\le 6$ retrains** (`2015–2026`, using only past data up to Day $D$): no retraining, periodic, sliding window / recency weighting, and drift-triggered retraining.
3. **Stage 3 Final Model & Live Week:** Choose final retraining/weighting strategy from Stage 2, update `train.py` to fit on data up to `2026-10-31`, verify with `python check_submission.py bryan_package`, and compute live-week MAPE prediction + interval for **7–13 Dec 2026**.

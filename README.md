# Pearls AQI Predictor — report outline

Structure below, with the substance you already have for each section.
Numbers are from the final runs; re-check them against
`models/hourly_training_summary.csv` before submitting.

---

## 1. Problem and approach

Forecast Karachi AQI up to 72 hours ahead using a serverless ML pipeline:
feature pipeline → feature store → training pipeline → dashboard.

State two framing decisions up front, because everything downstream
follows from them:

- **Hourly, not daily.** The daily formulation collapsed 47,000 hourly
  readings into ~270 rows and could not beat a one-parameter baseline.
  Hourly modelling kept 8,200 usable rows over the same window.
- **Models predict the change, not the level.** Forecast =
  `current_AQI + model.predict(features)`. On a highly autocorrelated
  series, asking a model to predict the level means asking it to
  relearn the identity function, which a persistence baseline gets for
  free.

## 2. Data

**Sources.** Pollutants from OpenWeather's air pollution API (CAMS model
output, not station measurements). Weather from Open-Meteo's ERA5
archive. AQI computed locally from PM2.5 using the pre-2024 US EPA
breakpoint table — say which table, since EPA moved the 50-AQI boundary
from 12.0 to 9.0 µg/m³ in 2024.

**Why not AQICN.** Karachi's only AQICN station has been offline since
March 2025, confirmed by direct feed check rather than search-index lag.
Islamabad's was down too, so this is a regional gap, not one dead
sensor.

**Coverage.** 47,040 hourly rows, Feb 2021 – Aug 2026, 98.0% of hourly
slots. Worth noting: the worst single gap in five years (44% missing,
26 Mar – 9 Apr 2025) coincides with the AQICN outage — two independent
providers degrading in the same window suggests a regional ingest
disruption rather than coincidence.

**Known limitation.** OpenWeather paywalls historical weather, so
temperature/humidity/wind were absent from every backfilled row. Section
6 covers what happened when that was filled from Open-Meteo.

## 3. Pipeline

Short section, mostly architecture. Cover:

- `backfill.py` — chunked historical fetch with retries and
  checkpointing
- `fetch_openmeteo_history.py` — ERA5 weather, joined on the local hour
- `train_hourly.py` — feature construction, CV, model selection
- `hopsworks_sync.py` — feature group + model registry
- `hourly_update.py` — stateless hourly job, reads history from the
  feature store rather than local disk
- GitHub Actions — hourly features, daily retrain
- Streamlit dashboard

Say why engineered features go in the store rather than raw readings:
training and serving then consume identical rows, so the two feature
definitions cannot silently drift.

## 4. Evaluation method

This section earns trust for everything after it.

- **Expanding-window cross-validation**, four folds. Every test hour
  falls strictly after the data the model was fitted on.
- **Baselines that can win**, and did: persistence (no change),
  climatology (revert to the weekly mean), damped persistence (one
  fitted parameter blending the two), diurnal-naive (same hour
  yesterday). The winner ships even when it is a baseline.
- **Leakage control.** Fitting on shuffled targets gives R² = +0.003 on
  held-out data. If the feature matrix leaked, this would be positive.
- **Selection caveat.** Eight models were compared on the same folds, so
  the winner's score is optimistically biased. State it.

## 5. Results

Headline table — mean RMSE across the 1–72h curve:

| Model | Curve mean | 24/48/72h |
|---|---|---|
| Ensemble (Ridge + gradient boosting) | 18.20 | 28.78 |
| ElasticNet | 18.57 | 29.31 |
| Ridge | 18.78 | 29.14 |
| Persistence baseline | 20.66 | 31.46 |
| Diurnal-naive | 31.99 | 36.65 |

Per-horizon skill against persistence: +48% at 1h, +40% at 6h, +27% at
12h, +14% at 24h, +6% at 72h.

**Put this in context rather than leaving RMSE bare.** Post-regime AQI
standard deviation is ~43.5, so predicting the mean every time scores
~43.5. At +24h the model is at 19.88 — well under half the error of
knowing nothing. EPA health categories are 50 points wide, so most
next-day forecasts land in the correct category.

**Skill decays with horizon**, and that is the honest headline: a day
ahead the models clearly beat assuming nothing changes; three days ahead
they barely do.

## 6. Findings

The strongest part of the report. Five results, each with evidence.

### 6.1 The target definition mattered more than the model

Instantaneous hourly AQI vs AQI computed from the 24-hour rolling mean
concentration (the EPA definition, and what station feeds publish):

| | instant | epa24h |
|---|---|---|
| Curve mean | 24.33 | 18.20 |
| +6h margin over persistence | +1.7% | +40% |
| +12h margin | 0.0% | +27% |

Crucially the **margins widened**, not just the absolute errors — so
this is not merely a smoother target flattering every model equally.
Averaging cancels noise in the CAMS output while preserving the
pollution signal, leaving more learnable structure per unit variance.

### 6.2 An undetected regime change inflated apparent skill fourfold

Hour-to-hour AQI variability drops roughly ninefold around July 2025.
Matched-quarter comparison of the day-over-day change standard
deviation:

| Quarter | 2021 | 2022 | 2023 | 2024 | 2025/26 |
|---|---|---|---|---|---|
| Q4 | 37.3 | 37.9 | 34.9 | 31.7 | **3.9** |
| Q1 | — | 36.7 | 36.5 | 40.9 | **8.0** |
| Q3 | 9.7 | 6.2 | 5.5 | 4.1 | **1.6** |

Four consecutive winters near 35, then 3.9. Training across the break
produced margins of 14–20% over persistence; restricting to the current
regime gave 3–6%. The high-variance regime is easy to beat and no longer
exists, so the pooled figure was an artifact. All reported results use
the post-2025-07 window only.

### 6.3 Weather did not help, and the reason is instructive

Twelve ERA5 features were joined — wind speed and direction, gusts,
boundary layer height, a ventilation index (wind × mixing depth),
temperature, humidity, pressure, precipitation, cloud cover. Curve mean
moved from 24.28 to 24.33: no improvement.

The explanation is that the features are *contemporaneous*. Observed
wind at time *t* is used to predict AQI at *t+24*, but what actually
clears the airshed is the wind blowing during those 24 hours. Small
gains did appear at 1–3h, where current weather is still approximately
current — consistent with this reading. The correct next experiment is
forecast weather aligned to each target hour, not observed weather.

### 6.4 A flexible model found real structure and still lost

SHAP dependence plots for gradient boosting at +24h show genuine
nonlinearity — NO₂ and CO both saturate, rising steeply at low
concentrations then flattening (linearity R² = 0.58 and 0.48). So the
relationship is not linear, and the tree model detected that.

It still lost to ElasticNet at every horizon. The scatter around those
curves is large: SHAP values at the same CO concentration span roughly
−5 to +25. On 8,200 rows with heavily collinear inputs, fitting the
curve costs more in estimation variance than it recovers in bias. The
linear model is wrong about the shape and right about the trade-off.

Note also that SHAP's top features for the tree model (`aqi_lag3`,
`aqi_roll24`, `no2`, `co`) barely overlap with Ridge's at the same
horizon (`pm10`, `pm25`, `aqi_diff1`, `o3`). With collinear inputs, two
models can attribute the same signal to different columns and predict
equally well — a caution against over-reading any single feature
ranking.

### 6.5 Momentum gives way to mean reversion between one and two days

Ridge coefficients on the AQI change, by horizon:

- **+24h**: `pm10 (+6.66)`, `pm25 (+5.27)`, `aqi_diff1 (+4.27)` — all
  positive. Current levels and recent momentum push the forecast up.
- **+48h / +72h**: `aqi_minus_roll168 (−3.57, −4.11)`,
  `aqi_minus_hourly_norm (−3.17, −2.81)` — negative, and the deviation
  terms dominate. The further above the weekly norm today sits, the more
  AQI falls.

The daily pipeline found the same thing independently: the fitted
reversion coefficient *k* in the damped-persistence baseline rose 0.19 →
0.47 → 0.63 across day+1 to day+3. Two unrelated methods describing one
physical behaviour.

## 7. A bug worth its own section

The AQI calculator's breakpoint bands were compared against raw
concentrations rather than truncated ones. EPA defines the bands on
values truncated to one decimal, so readings falling between bands —
12.03, 35.44, 55.41, 150.47 — matched nothing and returned null.

**Scale:** 33 nulls in 9,482 post-regime rows, 0.4%.

**Effect:** measured +1h persistence RMSE of 36.09, against a true
hour-to-hour change of 4.36 — an eightfold inflation. Isolated valid
values surrounded by nulls create artificial cliffs, and forecasting
depends on differences between consecutive values rather than the values
themselves, so a rounding-level defect distorted every metric in the
project.

The bug survived eleven iterations of model tuning because the
calculator was the one component never unit-tested. It was found by
auditing every stage against first principles rather than by looking
harder at the models.

## 8. Engineering notes

Brief, but these are concrete and specific:

- **Timezone handling caused three separate bugs** — mixed tz-aware and
  tz-naive parsing in training, a silently empty weather join, and a
  sort failure in the hourly pipeline. One shared parsing helper used
  everywhere would have prevented all three.
- **Deployment on Windows** required working around six issues: a
  hardcoded `/tmp` certificate path, an MSVC-only dependency
  (`twofish`), a missing Delta Lake library, a 256-character metadata
  limit, Kerberos/HDFS unavailability, and a missing Kafka client.
- **Custom estimators create coupling.** Pickle stores a class path, so
  every consumer of a saved model must import the training module. A
  shared `models.py` would be cleaner.
- **Dashboard and CI need different dependency sets.** The Hopsworks
  client pins protobuf below 5.0, which breaks on newer Python; the
  dashboard needs none of it.

## 9. Limitations and next steps

- Pollutants are CAMS model output, not station measurements. The series
  has no meaningful diurnal cycle — `diurnal_naive` was the worst
  baseline at every horizon — which is consistent with modelled rather
  than sensed data.
- The deployed dashboard reads committed artifacts, so it goes stale
  between pushes.
- Eight models compared on shared folds; the winner's margin is
  optimistically biased.
- **Highest-value next experiment:** forecast weather from Open-Meteo,
  aligned per target hour, rather than observed weather.
- Prediction intervals from quantile regression would replace the
  ±1 RMSE band, which currently assumes symmetric normal errors.

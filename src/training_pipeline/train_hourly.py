"""
Hourly AQI forecasting out to 72 hours.

Why this exists alongside train_model.py
----------------------------------------
The daily pipeline aggregated 6938 hourly readings into ~270 daily rows
and then could not beat a one-parameter persistence baseline at any
horizon. Two reasons, both caused by the aggregation itself:

  1. It discarded 96% of the data. 270 rows cannot support a 14-feature
     model.
  2. It averaged away the diurnal cycle - the strongest genuinely
     learnable pattern in the series. Karachi AQI has repeating
     intraday structure (morning traffic peak, afternoon dispersion,
     nighttime inversion buildup). Daily means erase it, leaving a
     smooth slow-drifting series where nothing but persistence works.

This pipeline forecasts hourly across a 1-72h curve using every row.

An important honesty note on baselines
--------------------------------------
At horizons that are exact multiples of 24h, persistence is already
diurnally aligned: aqi[t+24] and aqi[t] are the same hour of day, so
"predict today's value" is implicitly "predict same hour yesterday" and
stays strong. Persistence is NOT a weak baseline at +24/+48/+72.

Where it fails is everything in between (+6, +12, +36), because it
carries the wrong phase of the daily cycle. A dashboard showing a 72h
curve needs all of those hours to be right, so evaluation runs across a
spread of horizons rather than only the flattering multiples of 24. The
multiples of 24 are still reported separately so the numbers stay
comparable to the daily pipeline's day+1/+2/+3.

AQI convention
--------------
AQI here is computed from the instantaneous hourly PM2.5 concentration.
The official EPA PM2.5 AQI is defined on a 24-hour average, so this is
the "current AQI" convention used by real-time dashboards (AirNow,
IQAir), not the regulatory figure. State this in the report - it is a
deliberate choice appropriate to hourly nowcasting, not an error.

Usage:
    python -m src.training_pipeline.train_hourly
    python -m src.training_pipeline.train_hourly --cv-folds 4
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25
except ImportError:
    calculate_aqi_from_pm25 = None

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

LOCAL_TZ = "Asia/Karachi"

# Horizons spanning the whole 72h curve, not just the multiples of 24
# where persistence is diurnally aligned and looks artificially good.
HORIZONS = [1, 3, 6, 12, 24, 36, 48, 60, 72]
REPORT_HORIZONS = [24, 48, 72]  # comparable to daily day+1/+2/+3

POLLUTANT_COLUMNS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

FEATURE_COLUMNS = [
    # Current conditions
    "pm25", "pm10", "o3", "no2", "so2", "co", "aqi",
    # Short-range momentum
    "aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_diff1", "aqi_diff3",
    # Diurnal structure: same hour on previous days. This is the signal
    # daily aggregation destroyed.
    "aqi_lag24", "aqi_lag48", "aqi_lag168",
    # Level and trend at multiple scales
    "aqi_roll24", "aqi_roll72", "aqi_roll168",
    # Mean reversion (the only thing that worked in the daily pipeline)
    "aqi_minus_roll24", "aqi_minus_roll168",
    # Diurnal anomaly: how far this hour sits from its own recent
    # same-hour norm, separating "high for 3pm" from "high overall".
    "aqi_minus_hourly_norm",
    # Cyclical time. sin/cos so 23:00 -> 00:00 is continuous.
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "day_of_week",
]

MIN_FOLD_TRAIN = 800  # hours (~33 days)


class ZeroDeltaBaseline(RegressorMixin, BaseEstimator):
    """Persistence: predict no change from the current hour."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class DiurnalNaive(RegressorMixin, BaseEstimator):
    """
    Seasonal-naive on the daily cycle: predict the value observed at the
    same hour of day, 24h ago, carried forward.

    delta = aqi_lag24 - aqi. At horizons that are multiples of 24 this
    is close to persistence; between them it supplies the correct phase
    of the diurnal cycle, which is exactly where persistence breaks.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X["aqi_lag24"] - X["aqi"], dtype=float)


class DampedReversion(RegressorMixin, BaseEstimator):
    """
    The daily pipeline's winner, ported over: delta = k * (roll24 - aqi),
    with k fit by least squares through the origin. Included so the
    hourly models have to beat the best daily-scale result, not just
    plain persistence.
    """

    def fit(self, X, y):
        x = -np.asarray(X["aqi_minus_roll24"], dtype=float)
        y = np.asarray(y, dtype=float)
        denom = float(np.dot(x, x))
        self.k_ = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        return self

    def predict(self, X):
        return self.k_ * -np.asarray(X["aqi_minus_roll24"], dtype=float)


class BoostedResidual(RegressorMixin, BaseEstimator):
    """
    Ridge first, then gradient boosting on what Ridge got wrong.

    The linear model captures the smooth autocorrelated trend, which is
    most of the signal on a 24h-averaged series. The booster then only
    has to explain the nonlinear remainder, which is a much easier
    target than the raw delta and much harder to overfit. On smooth
    series this usually beats either component alone.
    """

    def __init__(self, alphas=None, n_splits=3):
        self.alphas = alphas
        self.n_splits = n_splits

    def fit(self, X, y):
        alphas = self.alphas if self.alphas is not None else np.logspace(0, 4, 30)
        self.linear_ = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=self.n_splits))),
        ]).fit(X, y)
        residual = np.asarray(y, dtype=float) - self.linear_.predict(X)
        self.booster_ = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=25, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42,
        ).fit(X, residual)
        return self

    def predict(self, X):
        return self.linear_.predict(X) + self.booster_.predict(X)


class MeanEnsemble(RegressorMixin, BaseEstimator):
    """
    Averages several fitted models.

    Ridge wins the long horizons and boosting wins the short ones, so
    their errors are partly decorrelated - averaging decorrelated
    predictors reduces variance without adding bias. The plain mean is
    used rather than a learned weighting, because fitting weights on the
    same folds used for selection would overfit them.
    """

    def __init__(self, models=None):
        self.models = models

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for _, m in self.models]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.fitted_], axis=0)


def make_candidates(n_train: int) -> dict:
    """
    Fresh unfitted models per fold.

    HistGradientBoosting replaces RandomForest here: with ~5000 rows and
    strong interaction structure (hour x pollutant) it fits better and
    faster, and it handles the NaN weather columns natively if you later
    join Open-Meteo history in.
    """
    inner = max(3, min(5, n_train // 500))
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(0, 4, 30),
                          cv=TimeSeriesSplit(n_splits=inner))),
    ])
    candidates = {
        "persistence": ZeroDeltaBaseline(),
        "diurnal_naive": DiurnalNaive(),
        "damped_reversion": DampedReversion(),
        # Early stopping added: the previous fixed 300 iterations overfit
        # at long horizons, where hist_gbr was the worst model overall
        # despite winning at +1h.
        "hist_gbr": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=42,
        ),
        "ridge": ridge,
        # ElasticNet can zero out useless features entirely, where Ridge
        # only shrinks them. With 38 features including a heavily
        # collinear lag/rolling block, that selection often helps.
        "elastic_net": Pipeline([
            ("scaler", StandardScaler()),
            ("net", ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                cv=TimeSeriesSplit(n_splits=inner),
                max_iter=20000, tol=1e-3, random_state=42,
            )),
        ]),
        "boosted_residual": BoostedResidual(n_splits=inner),
    }

    # Ensemble must come last - it clones the candidates above.
    candidates["ensemble"] = MeanEnsemble(models=[
        ("ridge", candidates["ridge"]),
        ("hist_gbr", candidates["hist_gbr"]),
    ])
    return candidates


def _parse_timestamps_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)


def load_hourly(since: str | None = None, target_mode: str = "epa24h") -> pd.DataFrame:
    """Loads features.csv onto a continuous, gap-explicit hourly local-time grid."""
    df = pd.read_csv(FEATURES_PATH)

    missing = [c for c in POLLUTANT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"features.csv missing columns: {missing}")

    fetched = _parse_timestamps_utc(df["fetched_at"])
    if "station_timestamp" in df.columns:
        df["observed_at"] = _parse_timestamps_utc(df["station_timestamp"]).fillna(fetched)
    else:
        df["observed_at"] = fetched

    bad = int(df["observed_at"].isna().sum())
    if bad:
        print(f"  dropped {bad} rows with unparseable timestamps")
        df = df.dropna(subset=["observed_at"])
    if df.empty:
        raise ValueError("No rows left after timestamp parsing")

    local = df["observed_at"].dt.tz_convert(LOCAL_TZ)
    df["ts"] = local.dt.floor("h").dt.tz_localize(None)

    # Collapse duplicate readings for the same hour (backfill and the
    # live pipeline can both cover an hour).
    hourly = df.groupby("ts")[POLLUTANT_COLUMNS].mean().reset_index()

    print("Data coverage")
    print(f"  raw rows:                    {len(df)}")
    print(f"  distinct hours:              {len(hourly)}")

    # -9999 is OpenWeather's missing-data sentinel, not a concentration.
    # Only a handful of rows, but each one contaminates a full week of
    # roll168/dev168 downstream - and dev168 is the strongest single
    # correlate with the 24h-ahead change.
    n_sentinel = 0
    for c in POLLUTANT_COLUMNS:
        bad = hourly[c] < 0
        n_sentinel += int(bad.sum())
        hourly.loc[bad, c] = np.nan
    if n_sentinel:
        print(f"  negative sentinels masked:   {n_sentinel}")

    # Regime filter. Hourly variability drops ~9x around July 2025
    # (see audit check 5): every quarter of 2021-2025Q2 shows change_std
    # near 35 in winter, versus ~4 afterwards. Pooling across that break
    # inflates apparent model skill, because the high-variance regime is
    # easy to beat persistence on and no longer exists.
    if since:
        cutoff = pd.Timestamp(since)
        before = len(hourly)
        hourly = hourly[hourly["ts"] >= cutoff].reset_index(drop=True)
        print(f"  regime filter >= {cutoff.date()}:  {before} -> {len(hourly)} hours")

    full = pd.date_range(hourly["ts"].min(), hourly["ts"].max(), freq="h")
    n_gap = len(full) - len(hourly)
    hourly = (hourly.set_index("ts").reindex(full)
              .rename_axis("ts").reset_index())
    print(f"  hourly grid span:            {len(full)} hours ({n_gap} missing)")

    # Short gaps only. limit=3 hours, interior only - never invents data
    # at the edges or across a long outage.
    for c in POLLUTANT_COLUMNS:
        hourly[c] = hourly[c].interpolate(limit=3, limit_area="inside")

    if calculate_aqi_from_pm25 is None:
        raise ImportError("aqi_calculator required - cannot compute hourly AQI")

    def _aqi(v):
        """
        Caps at 500 instead of returning None above the breakpoint table.

        The calculator returns None for PM2.5 beyond its top band, which
        silently dropped 180 rows - all of them extreme pollution events,
        exactly the days a forecast matters most. Fix this in
        aqi_calculator.py too so the CSV itself is correct.
        """
        if pd.isna(v):
            return np.nan
        r = calculate_aqi_from_pm25(v)
        return 500.0 if r is None else float(r)

    if target_mode == "epa24h":
        pm_for_aqi = hourly["pm25"].rolling(24, min_periods=18).mean()
    else:
        pm_for_aqi = hourly["pm25"]
    hourly["aqi"] = pm_for_aqi.apply(_aqi)
    n_capped = int((hourly["pm25"] > 500.4).sum())
    if n_capped:
        print(f"  AQI capped at 500:           {n_capped} extreme rows")

    return hourly


def add_features(h: pd.DataFrame) -> pd.DataFrame:
    """
    All features use only data at or before time t. Nothing here peeks
    forward; rolling windows are taken on shift(1) so the current hour
    is never inside its own average.
    """
    a = h["aqi"]

    h["aqi_lag1"] = a.shift(1)
    h["aqi_lag2"] = a.shift(2)
    h["aqi_lag3"] = a.shift(3)
    h["aqi_diff1"] = a.diff(1)
    h["aqi_diff3"] = a.diff(3)

    h["aqi_lag24"] = a.shift(24)
    h["aqi_lag48"] = a.shift(48)
    h["aqi_lag168"] = a.shift(168)

    h["aqi_roll24"] = a.shift(1).rolling(24, min_periods=18).mean()
    h["aqi_roll72"] = a.shift(1).rolling(72, min_periods=48).mean()
    h["aqi_roll168"] = a.shift(1).rolling(168, min_periods=120).mean()

    h["aqi_minus_roll24"] = a - h["aqi_roll24"]
    h["aqi_minus_roll168"] = a - h["aqi_roll168"]

    # Same-hour-of-day norm over the past week: mean of this hour on the
    # previous 3 days. Distinguishes "high for 3pm" from "high overall".
    same_hour_norm = (a.shift(24) + a.shift(48) + a.shift(72)) / 3.0
    h["aqi_minus_hourly_norm"] = a - same_hour_norm

    hour = h["ts"].dt.hour
    h["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    h["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = h["ts"].dt.dayofyear
    h["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    h["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    h["day_of_week"] = h["ts"].dt.dayofweek

    for hz in HORIZONS:
        h[f"target_{hz}"] = a.shift(-hz)
        h[f"delta_{hz}"] = h[f"target_{hz}"] - a

    return h


def evaluate_horizon(h: pd.DataFrame, hz: int, n_folds: int):
    """Expanding-window CV for one horizon. Models predict the delta."""
    cols = FEATURE_COLUMNS + [f"delta_{hz}", f"target_{hz}"]
    clean = h.dropna(subset=cols).reset_index(drop=True)
    if len(clean) < MIN_FOLD_TRAIN * 2:
        raise ValueError(f"only {len(clean)} usable rows")

    per_model = {}
    n_used = 0
    for tr_i, te_i in TimeSeriesSplit(n_splits=n_folds).split(clean):
        if len(tr_i) < MIN_FOLD_TRAIN:
            continue
        n_used += 1
        tr, te = clean.iloc[tr_i], clean.iloc[te_i]
        y_tr, y_te = tr[f"delta_{hz}"], te[f"delta_{hz}"]
        base_te = te["aqi"].to_numpy(dtype=float)

        for name, model in make_candidates(len(tr)).items():
            model.fit(tr[FEATURE_COLUMNS], y_tr)
            pd_ = np.asarray(model.predict(te[FEATURE_COLUMNS]), dtype=float)
            pa = base_te + pd_
            per_model.setdefault(name, []).append({
                "rmse": float(np.sqrt(mean_squared_error(te[f"target_{hz}"], pa))),
                "mae": float(mean_absolute_error(te[f"target_{hz}"], pa)),
                "r2": float(r2_score(te[f"target_{hz}"], pa)),
                "delta_r2": float(r2_score(y_te, pd_)),
            })

    if not per_model:
        raise ValueError("no fold met the minimum training size")

    return {n: {k: float(np.mean([f[k] for f in folds]))
                for k in ("rmse", "mae", "r2", "delta_r2")}
            for n, folds in per_model.items()}, len(clean), n_used


def main():
    ap = argparse.ArgumentParser(description="Hourly AQI forecasting to 72h")
    ap.add_argument("--cv-folds", type=int, default=4)
    ap.add_argument("--since", type=str, default="2025-07-01",
                    help="Regime cutoff (YYYY-MM-DD). Data before this has "
                         "~9x higher hourly variability. Pass 'none' to use "
                         "all history.")
    ap.add_argument("--target-mode", choices=["instant", "epa24h"], default="epa24h",
                    help="instant = AQI from the current hourly reading; "
                         "epa24h = AQI from the 24h rolling mean concentration "
                         "(the EPA definition, and what station feeds publish)")
    args = ap.parse_args()
    since = None if str(args.since).lower() in ("none", "all", "") else args.since

    if not FEATURES_PATH.exists():
        print(f"features.csv not found at {FEATURES_PATH}.", file=sys.stderr)
        sys.exit(1)

    try:
        hourly = add_features(load_hourly(since, args.target_mode))
    except (ValueError, KeyError, ImportError) as e:
        print(f"Could not build hourly dataset: {e}", file=sys.stderr)
        sys.exit(1)

    usable = hourly.dropna(subset=FEATURE_COLUMNS)
    print(f"  usable rows (all features):  {len(usable)}")
    print(f"  range: {hourly['ts'].min()} to {hourly['ts'].max()} ({LOCAL_TZ})")
    print(f"  hourly AQI mean/std:         {hourly['aqi'].mean():.1f} / {hourly['aqi'].std():.1f}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows, curve = [], {}

    for hz in HORIZONS:
        try:
            agg, n_rows, n_folds = evaluate_horizon(hourly, hz, args.cv_folds)
        except ValueError as e:
            print(f"+{hz}h: skipped ({e})", file=sys.stderr)
            continue

        curve[hz] = agg
        winner = min(agg, key=lambda k: agg[k]["rmse"])
        gain = (agg["persistence"]["rmse"] - agg[winner]["rmse"]) / agg["persistence"]["rmse"] * 100

        marker = " *" if hz in REPORT_HORIZONS else "  "
        print(f"+{hz:>3}h{marker} n={n_rows:<5} folds={n_folds}  "
              f"winner={winner:<17} RMSE={agg[winner]['rmse']:6.2f}  "
              f"(persistence {agg['persistence']['rmse']:6.2f}, {gain:+.1f}%)")

        for name, m in agg.items():
            rows.append({"horizon_h": hz, "model": name, "n_rows": n_rows, **m})

    if not curve:
        print("No horizons evaluated.", file=sys.stderr)
        sys.exit(1)

    # Mean RMSE across the whole curve is the number that matters for a
    # 72h dashboard - a model that only wins at multiples of 24 is not
    # actually usable for plotting a continuous forecast.
    print("\nMean RMSE across the full 1-72h curve:")
    names = list(next(iter(curve.values())).keys())
    for name in sorted(names, key=lambda n: np.mean([c[n]["rmse"] for c in curve.values()])):
        vals = [c[name]["rmse"] for c in curve.values()]
        sub = [curve[h][name]["rmse"] for h in REPORT_HORIZONS if h in curve]
        print(f"  {name:20s} all={np.mean(vals):6.2f}   at 24/48/72h={np.mean(sub):6.2f}")

    # Deployment models: one per reporting horizon, fit on everything.
    for hz in REPORT_HORIZONS:
        if hz not in curve:
            continue
        best = min(curve[hz], key=lambda k: curve[hz][k]["rmse"])
        cols = FEATURE_COLUMNS + [f"delta_{hz}"]
        clean = hourly.dropna(subset=cols)
        model = make_candidates(len(clean))[best]
        model.fit(clean[FEATURE_COLUMNS], clean[f"delta_{hz}"])
        joblib.dump({"model": model, "kind": best, "target": "delta",
                     "horizon_hours": hz, "features": FEATURE_COLUMNS},
                    MODELS_DIR / f"aqi_hourly_h{hz}.pkl")

    summary = MODELS_DIR / "hourly_training_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    print("\nNOTE: models predict a DELTA:")
    print("      forecast = current_aqi + model.predict(current_features)")
    print(f"Summary saved to: {summary}")


if __name__ == "__main__":
    main()
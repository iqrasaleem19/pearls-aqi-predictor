"""
Trains and evaluates AQI forecasting models for 3 horizons.

v6 changes - the persistence baseline was the wrong baseline:
-------------------------------------------------------------
v5 revealed that persistence has NEGATIVE R2 at day+2 (-0.05) and
day+3 (-0.43). Negative R2 means predicting today's AQI for three days
ahead is worse than ignoring the data and predicting the average every
time. So "persistence wins, ship it" was not a defensible conclusion -
it only won because nothing better was in the comparison.

What negative R2 actually indicates is MEAN REVERSION: an unusually
high day tends to fall back toward its recent norm rather than persist.
Two extra baselines are added to exploit that directly:

  * climatology_7d - predict the 7-day rolling mean instead of today's
    value. Equivalent to delta = rolling_mean_7 - aqi.

  * damped_persistence - predict a fraction of the way back to the
    rolling mean: delta = k * (rolling_mean_7 - aqi), with a single
    parameter k fit by least squares on the training fold. k=0 reduces
    to persistence, k=1 reduces to climatology, and anything between is
    the optimal blend. One parameter estimated from 40+ rows is far
    more stable than 14 correlated features, which is why the RF and
    Ridge models could not find this structure despite having
    aqi_minus_rolling_7 available to them.

Other v6 fixes:
  * Ridge instability. At day+3 v5 produced RMSE 65 +/- 64: RidgeCV's
    inner TimeSeriesSplit got only 2 splits on the earliest folds,
    selected a near-zero alpha from tiny validation sets, and
    extrapolated wildly. Fixed with an alpha floor and a guard on the
    inner split count.
  * MIN_FOLD_TRAIN raised so folds with too little history are skipped
    rather than contributing garbage to the mean.
  * Per-fold RMSE is now printed. The +/- spread in v5 was not noise,
    it was seasonality - early folds test Karachi winter (high,
    volatile AQI), late folds test monsoon (low, calm). Showing folds
    individually makes that legible instead of hiding it in a std.

v5 (kept): EPA-correct daily AQI from mean PM2.5, cyclical day-of-year
encoding, expanding-window CV, gap reporting.
v4 (kept): models predict the DELTA, not the level.
v3 (kept): tz-safe parsing, local day boundaries, calendar reindex.

Usage:
    python -m src.training_pipeline.train_model
    python -m src.training_pipeline.train_model --cv-folds 5
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25
except ImportError:
    calculate_aqi_from_pm25 = None

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

HORIZONS = [1, 2, 3]
LOCAL_TZ = "Asia/Karachi"
DEFAULT_MIN_READINGS = 18
MIN_FOLD_TRAIN = 60   # folds with less history than this are skipped

POLLUTANT_COLUMNS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

FEATURE_COLUMNS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "aqi", "aqi_change_rate",
    "aqi_lag1", "aqi_lag2",
    "aqi_rolling_mean_3", "aqi_rolling_mean_7",
    "aqi_minus_rolling_3", "aqi_minus_rolling_7",
    "doy_sin", "doy_cos",
]


class ZeroDeltaBaseline(RegressorMixin, BaseEstimator):
    """Persistence: tomorrow equals today. delta = 0."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class ClimatologyBaseline(RegressorMixin, BaseEstimator):
    """
    Predict the recent norm instead of today: delta = rolling_7 - aqi.

    aqi_minus_rolling_7 is already (aqi - rolling_7), so the required
    delta is just its negation - no fitting needed.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return -np.asarray(X["aqi_minus_rolling_7"], dtype=float)


class DampedPersistence(RegressorMixin, BaseEstimator):
    """
    Blend of persistence and climatology with one fitted parameter:

        delta = k * (rolling_7 - aqi)

    k = 0 is pure persistence, k = 1 is pure climatology. k is fit by
    least squares through the origin on the training fold, so the model
    learns how fast this particular series reverts. Expected to rise
    with horizon: little reversion by tomorrow, much more by day+3.
    """

    def fit(self, X, y):
        x = -np.asarray(X["aqi_minus_rolling_7"], dtype=float)
        y = np.asarray(y, dtype=float)
        denom = float(np.dot(x, x))
        self.k_ = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        return self

    def predict(self, X):
        return self.k_ * -np.asarray(X["aqi_minus_rolling_7"], dtype=float)


def make_candidates(n_train: int) -> dict:
    """Fresh unfitted models, rebuilt per fold so nothing leaks across folds."""
    # Guard the inner CV: too few splits on a small fold makes RidgeCV
    # select a near-zero alpha off a tiny validation set and extrapolate.
    inner_splits = max(3, min(5, n_train // 40))
    return {
        "persistence_baseline": ZeroDeltaBaseline(),
        "climatology_7d": ClimatologyBaseline(),
        "damped_persistence": DampedPersistence(),
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=8, min_samples_leaf=3,
            max_features=0.5, random_state=42, n_jobs=-1,
        ),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            # Alpha floor of 1.0: unregularised Ridge on 14 collinear
            # features over ~60 rows is what produced the day+3 blowup.
            ("ridge", RidgeCV(alphas=np.logspace(0, 4, 30),
                              cv=TimeSeriesSplit(n_splits=inner_splits))),
        ]),
    }


def _parse_timestamps_utc(series: pd.Series) -> pd.Series:
    """Parses mixed tz-naive/tz-aware strings into one tz-aware UTC Series."""
    return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)


def load_and_aggregate_daily(min_readings: int) -> pd.DataFrame:
    """Loads hourly features.csv, aggregates to one row per LOCAL day."""
    df = pd.read_csv(FEATURES_PATH)

    missing = [c for c in POLLUTANT_COLUMNS + ["aqi"] if c not in df.columns]
    if missing:
        raise KeyError(f"features.csv is missing required columns: {missing}")

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
    df["date"] = local.dt.normalize().dt.tz_localize(None)

    daily = df.groupby("date").agg({c: "mean" for c in POLLUTANT_COLUMNS})
    daily["n_readings"] = df.groupby("date").size()
    daily = daily.reset_index()

    if calculate_aqi_from_pm25 is not None:
        daily["aqi"] = daily["pm25"].apply(
            lambda v: calculate_aqi_from_pm25(v) if pd.notna(v) else np.nan
        )
    else:
        print("  WARNING: aqi_calculator unavailable, using mean(hourly AQI)")
        daily = daily.merge(df.groupby("date")["aqi"].mean().reset_index(),
                            on="date", how="left")

    print("Data coverage")
    print(f"  raw hourly rows:             {len(df)}")
    print(f"  distinct local days:         {len(daily)}")

    n_partial = int((daily["n_readings"] < min_readings).sum())
    if n_partial:
        print(f"  partial days dropped:        {n_partial} (< {min_readings} readings)")
    daily = daily[daily["n_readings"] >= min_readings].copy()
    if daily.empty:
        raise ValueError(f"No days had >= {min_readings} readings.")

    daily = daily.sort_values("date").reset_index(drop=True)

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    gap_dates = full_range.difference(pd.DatetimeIndex(daily["date"]))
    if len(gap_dates):
        print(f"  calendar gaps:               {len(gap_dates)} "
              f"({', '.join(str(d.date()) for d in gap_dates[:5])}"
              f"{', ...' if len(gap_dates) > 5 else ''})")

    # Fill only short interior holes. limit=2 refuses runs longer than
    # two days; limit_area="inside" refuses to invent data at the edges.
    # Defensible on a series this autocorrelated, and it stops a single
    # missing day from cascading into ~8 unusable rows.
    daily["aqi"] = daily["aqi"].interpolate(limit=2, limit_area="inside")
    for c in POLLUTANT_COLUMNS:
        daily[c] = daily[c].interpolate(limit=2, limit_area="inside")

    doy = daily["date"].dt.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    daily["aqi_change_rate"] = daily["aqi"].diff()
    daily["aqi_lag1"] = daily["aqi"].shift(1)
    daily["aqi_lag2"] = daily["aqi"].shift(2)
    daily["aqi_rolling_mean_3"] = daily["aqi"].shift(1).rolling(3, min_periods=2).mean()
    daily["aqi_rolling_mean_7"] = daily["aqi"].shift(1).rolling(7, min_periods=5).mean()
    daily["aqi_minus_rolling_3"] = daily["aqi"] - daily["aqi_rolling_mean_3"]
    daily["aqi_minus_rolling_7"] = daily["aqi"] - daily["aqi_rolling_mean_7"]

    return daily


def build_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """Adds absolute and delta targets per horizon."""
    for h in HORIZONS:
        daily[f"aqi_target_{h}"] = daily["aqi"].shift(-h)
        daily[f"aqi_delta_{h}"] = daily[f"aqi_target_{h}"] - daily["aqi"]
    return daily


def _fit_score(model, tr: pd.DataFrame, te: pd.DataFrame, horizon: int) -> dict:
    """Fits on tr, scores on te. Returns absolute-scale and delta-scale metrics."""
    delta_col, abs_col = f"aqi_delta_{horizon}", f"aqi_target_{horizon}"

    model.fit(tr[FEATURE_COLUMNS], tr[delta_col])
    pred_delta = np.asarray(model.predict(te[FEATURE_COLUMNS]), dtype=float)
    pred_abs = te["aqi"].to_numpy(dtype=float) + pred_delta

    return {
        "rmse": float(np.sqrt(mean_squared_error(te[abs_col], pred_abs))),
        "mae": float(mean_absolute_error(te[abs_col], pred_abs)),
        "r2": float(r2_score(te[abs_col], pred_abs)),
        "delta_r2": float(r2_score(te[delta_col], pred_delta)),
        "k": getattr(model, "k_", np.nan),
    }


def cross_validate(daily: pd.DataFrame, horizon: int, n_folds: int):
    """Expanding-window CV: each fold trains only on data preceding it."""
    delta_col, abs_col = f"aqi_delta_{horizon}", f"aqi_target_{horizon}"
    clean = daily.dropna(subset=FEATURE_COLUMNS + [delta_col, abs_col]).reset_index(drop=True)

    if len(clean) < MIN_FOLD_TRAIN * 2:
        raise ValueError(f"Only {len(clean)} usable rows for day+{horizon}")

    per_model, fold_info = {}, []
    for i, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(clean), 1):
        if len(tr_idx) < MIN_FOLD_TRAIN:
            print(f"    fold {i}: skipped ({len(tr_idx)} train rows < {MIN_FOLD_TRAIN})")
            continue
        tr, te = clean.iloc[tr_idx], clean.iloc[te_idx]
        fold_info.append((i, te["date"].min().date(), te["date"].max().date()))
        for name, model in make_candidates(len(tr)).items():
            per_model.setdefault(name, []).append(_fit_score(model, tr, te, horizon))

    if not fold_info:
        raise ValueError(f"No fold had >= {MIN_FOLD_TRAIN} training rows")
    return per_model, len(clean), fold_info


def fit_final(daily: pd.DataFrame, horizon: int, name: str):
    """Refits the chosen model on ALL usable data, for deployment."""
    delta_col, abs_col = f"aqi_delta_{horizon}", f"aqi_target_{horizon}"
    clean = daily.dropna(subset=FEATURE_COLUMNS + [delta_col, abs_col])
    model = make_candidates(len(clean))[name]
    model.fit(clean[FEATURE_COLUMNS], clean[delta_col])
    return model, len(clean)


def main():
    parser = argparse.ArgumentParser(description="Train AQI forecasting models")
    parser.add_argument("--min-readings", type=int, default=DEFAULT_MIN_READINGS)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        print(f"features.csv not found at {FEATURES_PATH}.", file=sys.stderr)
        sys.exit(1)

    try:
        daily = load_and_aggregate_daily(args.min_readings)
    except (ValueError, KeyError) as e:
        print(f"Could not build daily dataset: {e}", file=sys.stderr)
        sys.exit(1)

    daily = build_targets(daily)
    usable = daily.dropna(subset=FEATURE_COLUMNS)

    print(f"  usable rows (all features):  {len(usable)}")
    print(f"  date range:                  {daily['date'].min().date()} to "
          f"{daily['date'].max().date()} ({LOCAL_TZ})")
    print(f"  daily AQI mean/std:          {daily['aqi'].mean():.1f} / {daily['aqi'].std():.1f}")
    print(f"  evaluation: {args.cv_folds}-fold expanding-window CV\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for horizon in HORIZONS:
        print(f"--- Horizon: day+{horizon} ---")
        try:
            per_model, n_rows, fold_info = cross_validate(daily, horizon, args.cv_folds)
        except ValueError as e:
            print(f"  Skipped: {e}\n", file=sys.stderr)
            continue

        print(f"  {n_rows} usable rows, {len(fold_info)} folds:")
        for i, start, end in fold_info:
            print(f"    fold {i}: test {start} -> {end}")

        n_f = len(fold_info)
        print(f"\n  {'model':22s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'dR2':>7s}   "
              + " ".join(f"{'f'+str(i):>6s}" for i, _, _ in fold_info))

        agg = {}
        for name, folds in per_model.items():
            rmses = [f["rmse"] for f in folds]
            agg[name] = {
                "rmse": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                "mae": float(np.mean([f["mae"] for f in folds])),
                "r2": float(np.mean([f["r2"] for f in folds])),
                "delta_r2": float(np.mean([f["delta_r2"] for f in folds])),
                "k": (lambda ks: float(np.mean(ks)) if ks else float("nan"))(
                    [f["k"] for f in folds if not np.isnan(f["k"])]
                ),
            }
            a = agg[name]
            per_fold = " ".join(f"{r:6.1f}" for r in rmses)
            print(f"  {name:22s} {a['rmse']:7.2f} {a['mae']:7.2f} {a['r2']:7.3f} "
                  f"{a['delta_r2']:7.3f}   {per_fold}")
            summary_rows.append({"horizon": horizon, "model": name,
                                 "n_rows": n_rows, "n_folds": n_f, **a})

        if not np.isnan(agg["damped_persistence"]["k"]):
            print(f"  (damped_persistence reversion k = {agg['damped_persistence']['k']:.2f}; "
                  f"0 = pure persistence, 1 = pure climatology)")

        base = agg["persistence_baseline"]
        winner_name = min(agg, key=lambda k: agg[k]["rmse"])
        win = agg[winner_name]

        # Compare on the delta scale: won-by-how-much relative to fold noise.
        if winner_name != "persistence_baseline":
            gain = (base["rmse"] - win["rmse"]) / base["rmse"] * 100
            print(f"  -> Winner: {winner_name} ({gain:.1f}% better RMSE than persistence)\n")
        else:
            print("  -> Winner: persistence_baseline\n")

        model, n_fit = fit_final(daily, horizon, winner_name)
        joblib.dump(
            {"model": model, "kind": winner_name, "target": "delta",
             "horizon": horizon, "features": FEATURE_COLUMNS, "n_train": n_fit},
            MODELS_DIR / f"aqi_model_day{horizon}.pkl",
        )
        summary_rows.append({"horizon": horizon, "model": f"WINNER:{winner_name}",
                             "n_rows": n_rows, "n_folds": n_f, **win})

    if not summary_rows:
        print("No horizons trained successfully.", file=sys.stderr)
        sys.exit(1)

    summary_path = MODELS_DIR / "training_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("NOTE: saved models predict a DELTA:")
    print("      forecast = today_aqi + model.predict(today_features)")
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
"""
Hopsworks feature store and model registry sync.

What goes where
---------------
Feature group `aqi_features_hourly` holds the ENGINEERED hourly frame -
lags, rolling means, deviations, weather and cyclical time already
computed - not the raw pollutant readings.

That choice matters. Right now train_hourly.py builds features in
add_features() and app.py rebuilds them in build_feature_row(). Nothing
enforces that the two agree. If they ever drift, the dashboard serves
predictions from a different feature space than the model was trained
on, and it fails silently - the numbers still look plausible. Pushing
engineered features means both read identical rows and that class of bug
becomes impossible.

The trade-off is that changing a feature definition now requires a new
feature group version and a backfill, rather than editing one function.
For a project this size that is the correct pattern.

Model registry holds each aqi_hourly_h*.pkl with its cross-validated
metrics attached, so the registry UI shows RMSE per horizon rather than
being an opaque blob store.

Windows note
------------
The Hopsworks client hardcodes a Unix /tmp path for certificate
materialisation. On Windows that resolves relative to the current
drive, so `mkdir D:\\tmp` (or whichever drive the project sits on) is
required once before any login succeeds. ensure_tmp() below does this
automatically.

Usage:
    python -m src.feature_store.hopsworks_sync --dry-run
    python -m src.feature_store.hopsworks_sync --features
    python -m src.feature_store.hopsworks_sync --models
    python -m src.feature_store.hopsworks_sync --all
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.training_pipeline.train_hourly import (  # noqa: F401
    MeanEnsemble, BoostedResidual, ZeroDeltaBaseline,
    DiurnalNaive, DampedReversion,
)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
SUMMARY_PATH = MODELS_DIR / "hourly_training_summary.csv"

FEATURE_GROUP_NAME = "aqi_features_hourly"
FEATURE_GROUP_VERSION = 4
MODEL_PREFIX = "aqi_hourly"

# Hopsworks requires lowercase alphanumeric + underscore column names,
# a non-null primary key, and no duplicate keys. Validated before upload
# so a malformed write fails loudly instead of half-succeeding.
PRIMARY_KEY = "ts"
EVENT_TIME = "ts"


def ensure_tmp():
    """
    Creates the /tmp directory the client expects. On Windows this
    resolves against the current drive; on Unix it already exists.
    """
    try:
        Path("/tmp").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def connect():
    """Logs in using credentials from .env."""
    from dotenv import load_dotenv
    import hopsworks

    load_dotenv(ROOT / ".env")
    key = os.environ.get("HOPSWORKS_API_KEY")
    proj = os.environ.get("HOPSWORKS_PROJECT")
    if not key or not proj:
        raise RuntimeError(
            "HOPSWORKS_API_KEY and HOPSWORKS_PROJECT must be set in .env"
        )

    ensure_tmp()
    return hopsworks.login(api_key_value=key, project=proj)


def build_frame(since: str = "2025-07-01") -> pd.DataFrame:
    """
    Builds the engineered frame by calling train_hourly's own functions,
    so there is exactly one definition of what a feature is.
    """
    from src.training_pipeline.train_hourly import (
        load_hourly, add_features, FEATURE_COLUMNS, HORIZONS,
    )

    hourly = add_features(load_hourly(since))
    keep = ["ts"] + FEATURE_COLUMNS
    keep = [c for c in dict.fromkeys(keep) if c in hourly.columns]

    df = hourly[keep].copy()
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> list:
    """
    Checks the frame against Hopsworks' constraints. Returns a list of
    problems; empty means safe to upload.
    """
    problems = []

    if PRIMARY_KEY not in df.columns:
        problems.append(f"missing primary key column '{PRIMARY_KEY}'")
        return problems

    if df[PRIMARY_KEY].isna().any():
        problems.append(f"primary key '{PRIMARY_KEY}' contains nulls")

    dupes = int(df[PRIMARY_KEY].duplicated().sum())
    if dupes:
        problems.append(f"primary key has {dupes} duplicate value(s)")

    for c in df.columns:
        if c != c.lower():
            problems.append(f"column '{c}' has uppercase characters")
        if not all(ch.isalnum() or ch == "_" for ch in c):
            problems.append(f"column '{c}' has non-alphanumeric characters")
        if c[0].isdigit():
            problems.append(f"column '{c}' starts with a digit")

    if df.empty:
        problems.append("frame is empty")

    # Object-dtype columns become strings in the offline store, which
    # silently breaks numeric features on read-back.
    for c, dt in df.dtypes.items():
        if dt == object and c != PRIMARY_KEY:
            problems.append(f"column '{c}' is object dtype, expected numeric")

    return problems


def describe(df: pd.DataFrame) -> None:
    """Prints what would be uploaded."""
    print(f"  rows:        {len(df)}")
    print(f"  columns:     {len(df.columns)}")
    print(f"  primary key: {PRIMARY_KEY}")
    if PRIMARY_KEY in df.columns and len(df):
        print(f"  range:       {df[PRIMARY_KEY].min()} -> {df[PRIMARY_KEY].max()}")
    nulls = df.isna().mean()
    worst = nulls[nulls > 0].sort_values(ascending=False).head(6)
    if len(worst):
        print("  columns with nulls:")
        for c, pct in worst.items():
            print(f"    {c:26s} {pct * 100:5.1f}%")
    else:
        print("  no nulls")


def sync_features(df: pd.DataFrame, project) -> None:
    """Creates or updates the feature group and inserts the frame."""
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=[PRIMARY_KEY],
        event_time=EVENT_TIME,
        time_travel_format="HUDI",
        description=(
            "Engineered hourly AQI features for Karachi. Pollutants from "
            "OpenWeather (CAMS), weather from Open-Meteo (ERA5). Lags, "
            "rolling means, mean-reversion deviations, cyclical time, plus "
            "delta and absolute targets at 1-72h. Post-2025-07 regime only."
        ),
        online_enabled=False,
    )

    print(f"  inserting {len(df)} rows into {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}...")
    fg.insert(df, write_options={"wait_for_job": True})
    print("  done")


def sync_models(project) -> None:
    """Registers each saved model with its cross-validated metrics."""
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    mr = project.get_model_registry()

    skill = {}
    if SUMMARY_PATH.exists():
        s = pd.read_csv(SUMMARY_PATH)
        for hz, grp in s.groupby("horizon_h"):
            win = grp[grp["model"].astype(str).str.startswith("WINNER:")]
            row = win.iloc[0] if len(win) else grp.loc[grp["rmse"].idxmin()]
            skill[int(hz)] = {
                "rmse": float(row["rmse"]),
                "mae": float(row.get("mae", np.nan)),
                "r2": float(row.get("r2", np.nan)),
            }

    paths = sorted(MODELS_DIR.glob(f"{MODEL_PREFIX}_h*.pkl"))
    if not paths:
        print(f"  no models found in {MODELS_DIR}")
        return

    for path in paths:
        bundle = joblib.load(path)
        hz = int(bundle["horizon_hours"])
        metrics = skill.get(hz, {})
        metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}

        model = mr.python.create_model(
            name=f"{MODEL_PREFIX}_h{hz}",
            metrics=metrics,
            description=(
                f"AQI forecast at +{hz}h. Winner: {bundle.get('kind')}. "
                f"Predicts a DELTA - inference is "
                f"current_aqi + model.predict(features)."
            ),
        )
        model.save(str(path), keep_original_files=True)
        print(f"  registered {MODEL_PREFIX}_h{hz} "
              f"({bundle.get('kind')}, RMSE={metrics.get('rmse', float('nan')):.2f})")


def load_features_from_store(project=None) -> pd.DataFrame:
    """
    Read path for training and the dashboard. Returns the same engineered
    frame that was uploaded, so both consume identical rows.
    """
    if project is None:
        project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    return fg.read().sort_values(PRIMARY_KEY).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Sync features and models to Hopsworks")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and validate the frame locally, upload nothing")
    ap.add_argument("--features", action="store_true", help="Upload the feature group")
    ap.add_argument("--models", action="store_true", help="Register saved models")
    ap.add_argument("--all", action="store_true", help="Both")
    ap.add_argument("--since", type=str, default="2025-07-01",
                    help="Regime cutoff, must match training")
    args = ap.parse_args()

    do_features = args.features or args.all
    do_models = args.models or args.all
    if not (do_features or do_models or args.dry_run):
        ap.error("pass --dry-run, --features, --models or --all")

    if do_features or args.dry_run:
        print("Building engineered frame...")
        df = build_frame(args.since)
        describe(df)

        problems = validate(df)
        if problems:
            print("\nVALIDATION FAILED:")
            for p in problems:
                print(f"  - {p}")
            sys.exit(1)
        print("  validation passed")

    if args.dry_run:
        print("\nDry run - nothing uploaded.")
        return

    project = connect()
    print(f"Connected to {project.name}\n")

    if do_features:
        print("Feature group:")
        sync_features(df, project)

    if do_models:
        print("\nModel registry:")
        sync_models(project)

    print("\nSync complete.")


if __name__ == "__main__":
    main() 
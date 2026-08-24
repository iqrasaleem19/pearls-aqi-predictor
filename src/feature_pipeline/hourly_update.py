"""
Hourly feature pipeline. Stateless - Hopsworks is the source of truth.

Runs on a fresh CI machine with no local data. The sequence is:

  1. Read the recent tail of the feature group. Enough history is needed
     to compute aqi_roll168 and aqi_lag168, so 14 days are pulled - a
     comfortable margin over the 7 days those features strictly need.
  2. Fetch the current reading from OpenWeather.
  3. Rebuild the hourly grid, recompute features over history + the new
     row, and keep only rows newer than what the store already holds.
  4. Insert. The feature group's primary key is `ts`, so a re-run over
     an hour that already exists updates in place rather than
     duplicating.

Why not just append the new row directly: lag and rolling features for
hour T depend on hours T-1 through T-168. They cannot be computed from
a single reading, so the history has to come back out of the store,
which is exactly what a feature store is for.

Idempotent by design. Running it twice for the same hour is harmless,
and a missed run self-heals on the next one as long as the gap is under
the interpolation limit.

Usage:
    python -m src.feature_pipeline.hourly_update
    python -m src.feature_pipeline.hourly_update --dry-run
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25  # noqa: E402
from src.feature_store.hopsworks_sync import (  # noqa: E402
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, connect, validate,
)
from src.training_pipeline.train_hourly import (  # noqa: E402
    FEATURE_COLUMNS, POLLUTANT_COLUMNS, LOCAL_TZ,
)

CURRENT_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

# History needed to compute the longest-window feature (roll168 / lag168)
# is 7 days. 14 gives margin for gaps without pulling the whole group.
HISTORY_DAYS = 14


def fetch_current(lat: float, lon: float, api_key: str) -> dict:
    """Fetches the current air pollution reading."""
    r = requests.get(
        CURRENT_URL,
        params={"lat": lat, "lon": lon, "appid": api_key},
        timeout=30,
    )
    r.raise_for_status()
    entries = r.json().get("list", [])
    if not entries:
        raise RuntimeError("OpenWeather returned no readings")

    entry = entries[0]
    comp = entry["components"]
    observed = pd.to_datetime(entry["dt"], unit="s", utc=True)

    row = {"ts_utc": observed}
    for name, key in [("pm25", "pm2_5"), ("pm10", "pm10"), ("o3", "o3"),
                      ("no2", "no2"), ("so2", "so2"), ("co", "co")]:
        v = comp.get(key)
        # -9999 is the missing-data sentinel, not a concentration.
        row[name] = np.nan if v is None or v < 0 else float(v)
    return row


def read_history(fs, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Reads the recent tail of the feature group."""
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    if df.empty:
        return df

    ts = pd.to_datetime(df["ts"], utc=True)
    df["ts"] = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    cutoff = df["ts"].max() - pd.Timedelta(days=days)
    return df[df["ts"] >= cutoff].sort_values("ts").reset_index(drop=True)


def rebuild_features(history: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    """
    Recomputes features over history plus the new reading.

    Deliberately mirrors add_features() in train_hourly.py. The duplication
    is a known weakness - if the two ever drift, serving silently uses a
    different feature space than training. The right fix is a shared
    feature-definition module that both import; flagged rather than
    hidden.
    """
    local_ts = new_row["ts_utc"].tz_convert(LOCAL_TZ).floor("h").tz_localize(None)

    incoming = pd.DataFrame([{
        "ts": local_ts,
        **{c: new_row[c] for c in POLLUTANT_COLUMNS},
    }])

    base_cols = ["ts"] + POLLUTANT_COLUMNS
    hist = history[[c for c in base_cols if c in history.columns]].copy()
    combined = pd.concat([hist, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")

    # Continuous hourly grid: shift(n) means "n rows back", so a gap
    # would silently make lag24 point at the wrong hour.
    grid = pd.date_range(combined["ts"].min(), combined["ts"].max(), freq="h")
    h = combined.set_index("ts").reindex(grid).rename_axis("ts")

    for c in POLLUTANT_COLUMNS:
        h[c] = h[c].interpolate(limit=3, limit_area="inside")

    # EPA 24h-average definition - must match train_hourly's target mode.
    pm24 = h["pm25"].rolling(24, min_periods=18).mean()
    h["aqi"] = pm24.apply(
        lambda v: np.nan if pd.isna(v) else float(calculate_aqi_from_pm25(v))
    )

    a = h["aqi"]
    for L in (1, 2, 3, 24, 48, 168):
        h[f"aqi_lag{L}"] = a.shift(L)
    h["aqi_diff1"] = a.diff(1)
    h["aqi_diff3"] = a.diff(3)
    h["aqi_roll24"] = a.shift(1).rolling(24, min_periods=18).mean()
    h["aqi_roll72"] = a.shift(1).rolling(72, min_periods=48).mean()
    h["aqi_roll168"] = a.shift(1).rolling(168, min_periods=120).mean()
    h["aqi_minus_roll24"] = a - h["aqi_roll24"]
    h["aqi_minus_roll168"] = a - h["aqi_roll168"]
    h["aqi_minus_hourly_norm"] = a - (a.shift(24) + a.shift(48) + a.shift(72)) / 3.0

    hour = h.index.hour
    h["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    h["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = h.index.dayofyear
    h["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    h["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    h["day_of_week"] = h.index.dayofweek

    return h.reset_index()


def main():
    ap = argparse.ArgumentParser(description="Hourly feature pipeline")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute everything, insert nothing")
    args = ap.parse_args()
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        print("OPENWEATHER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        from src.config import CITY_LAT, CITY_LON
    except ImportError:
        CITY_LAT, CITY_LON = 24.8607, 67.0011

    print(f"Run at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

    project = connect()
    fs = project.get_feature_store()

    history = read_history(fs)
    if history.empty:
        print("Feature group is empty - run the backfill sync first", file=sys.stderr)
        sys.exit(1)
    latest_stored = history["ts"].max()
    print(f"  store holds {len(history)} recent rows, latest {latest_stored}")

    reading = fetch_current(CITY_LAT, CITY_LON, api_key)
    print(f"  fetched reading for {reading['ts_utc']} UTC "
          f"(pm25={reading['pm25']})")
    reading = fetch_current(CITY_LAT, CITY_LON, api_key)
    print(f"  fetched reading for {reading['ts_utc']} UTC "
          f"(pm25={reading['pm25']})")

    gap_hours = (reading["ts_utc"].tz_convert(LOCAL_TZ).tz_localize(None)
                 - latest_stored).total_seconds() / 3600
    if gap_hours > 12:
        print(f"  store is {gap_hours:.0f}h behind - run the backfill to "
              f"close the gap before the hourly job can extend it",
              file=sys.stderr)
        sys.exit(1)

    
    rebuilt = rebuild_features(history, reading)

    # Only rows the store does not already have. Recomputing over history
    # would otherwise re-insert unchanged rows on every run.
    fresh = rebuilt[rebuilt["ts"] > latest_stored].copy()
    fresh = fresh.dropna(subset=FEATURE_COLUMNS)

    if fresh.empty:
        print("  no new complete rows - store is already current")
        return

    keep = ["ts"] + FEATURE_COLUMNS
    fresh = fresh[keep]

    problems = validate(fresh)
    if problems:
        print("  VALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"    - {p}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(fresh)} new row(s): "
          f"{fresh['ts'].min()} -> {fresh['ts'].max()}")

    if args.dry_run:
        print("  dry run - nothing inserted")
        return

    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    fg.insert(fresh, write_options={"wait_for_job": True})
    print("  inserted")


if __name__ == "__main__":
    main()
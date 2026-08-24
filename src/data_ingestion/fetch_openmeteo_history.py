"""
Backfills historical WEATHER from Open-Meteo and joins it onto
features.csv, matched on the local hour.

Why this exists
---------------
OpenWeather paywalls historical weather, so every backfilled row in
features.csv has NaN temperature/humidity/wind/pressure - 99.99% of the
dataset. The audit (check 7) showed a saturated linear model given every
available pollutant feature explains essentially none of the 24h-ahead
AQI CHANGE. That is the expected result when the causal variables are
absent: wind disperses particulates, temperature inversions and a low
boundary layer trap them, humidity drives secondary aerosol formation.
None of that is in the pollutant columns.

Open-Meteo's archive (ERA5 reanalysis) is free, needs no API key, and
covers 1940-present. This script pulls the hourly weather for the same
coordinates and window as the existing data.

Two things worth knowing about the data
---------------------------------------
1. ERA5 is reanalysis, not station observation - a physical model
   constrained by observations. For wind and temperature over a coastal
   city it is reliable; treat boundary_layer_height as indicative
   rather than exact.
2. ERA5 has roughly a 5-day lag before recent dates are finalised. The
   most recent few days may come back null. That is expected, and the
   live pipeline covers the present anyway.

Usage:
    python -m src.data_ingestion.fetch_openmeteo_history
    python -m src.data_ingestion.fetch_openmeteo_history --dry-run
    python -m src.data_ingestion.fetch_openmeteo_history --no-join
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from src.config import CITY_LAT, CITY_LON
except ImportError:
    CITY_LAT, CITY_LON = 24.8607, 67.0011  # Karachi

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
WEATHER_PATH = PROCESSED_DIR / "weather_history.csv"

LOCAL_TZ = "Asia/Karachi"

# Requested in priority order. Each is probed individually first, since
# the archive's variable list changes over time and one unsupported name
# would otherwise fail the whole request.
CANDIDATE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    # The dispersion variables - most likely to carry real signal about
    # AQI change, and the most likely to be unavailable.
    "boundary_layer_height",
    "temperature_100m",  # with temperature_2m, gives an inversion proxy
]

CHUNK_DAYS = 365
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def probe_variables(lat: float, lon: float) -> list:
    """
    Tests each candidate variable with a one-day request and keeps the
    ones the archive actually returns. Cheaper than discovering an
    unsupported name partway through a multi-year backfill.
    """
    print("Probing available variables...")
    available = []
    for var in CANDIDATE_VARIABLES:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": "2023-01-01", "end_date": "2023-01-01",
            "hourly": var, "timezone": "UTC",
        }
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=30)
            if r.status_code == 200 and var in r.json().get("hourly", {}):
                available.append(var)
                print(f"  ok        {var}")
            else:
                reason = ""
                try:
                    reason = r.json().get("reason", "")[:60]
                except Exception:
                    pass
                print(f"  skipped   {var}  {reason}")
        except Exception as e:
            print(f"  skipped   {var}  ({e})")
        time.sleep(0.3)

    if not available:
        raise RuntimeError("No variables available - check connectivity to Open-Meteo")
    return available


def fetch_chunk(lat: float, lon: float, start: str, end: str, variables: list) -> pd.DataFrame:
    """Fetches one date range. Returns an empty frame if all retries fail."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            hourly = r.json().get("hourly", {})
            if not hourly.get("time"):
                return pd.DataFrame()
            df = pd.DataFrame(hourly)
            # Archive returns naive UTC strings; make them explicit.
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    failed after {MAX_RETRIES} attempts: {e}", file=sys.stderr)
                return pd.DataFrame()
            wait = RETRY_BACKOFF ** attempt
            print(f"    attempt {attempt} failed ({e}); retrying in {wait:.0f}s")
            time.sleep(wait)
    return pd.DataFrame()


def target_date_range() -> tuple:
    """Reads features.csv to match the weather window to the AQI window."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} not found - run the AQI backfill first")

    df = pd.read_csv(FEATURES_PATH, usecols=["fetched_at"])
    ts = pd.to_datetime(df["fetched_at"], format="mixed", errors="coerce", utc=True)
    ts = ts.dropna()
    if ts.empty:
        raise ValueError("No parseable timestamps in features.csv")
    return ts.min().date(), ts.max().date()


def backfill_weather(lat: float, lon: float, start_date, end_date,
                     variables: list) -> pd.DataFrame:
    """Pulls the full range in yearly chunks."""
    chunks, cur = [], pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total = int((end - cur).days / CHUNK_DAYS) + 1
    i = 0

    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end)
        i += 1
        print(f"[{i}/{total}] {cur.date()} -> {chunk_end.date()}", end="")
        df = fetch_chunk(lat, lon, str(cur.date()), str(chunk_end.date()), variables)
        print(f"  ({len(df)} hours)" if len(df) else "  FAILED")
        if len(df):
            chunks.append(df)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)

    if not chunks:
        raise RuntimeError("No weather data retrieved")

    weather = pd.concat(chunks, ignore_index=True)
    weather = weather.drop_duplicates(subset=["time"]).sort_values("time")
    return weather.reset_index(drop=True)


def derive_features(w: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the physically-motivated derived variables. These matter more
    than the raw readings: dispersion depends on wind and mixing depth,
    not on temperature alone.
    """
    if "temperature_100m" in w.columns and "temperature_2m" in w.columns:
        # Positive = temperature rises with height = inversion = trapped
        # pollutants. This is the classic winter smog mechanism.
        w["inversion_strength"] = w["temperature_100m"] - w["temperature_2m"]

    if "wind_speed_10m" in w.columns:
        # Ventilation index proxy: wind speed x mixing depth. The
        # standard single-number measure of an airshed's ability to
        # clear itself.
        if "boundary_layer_height" in w.columns:
            w["ventilation_index"] = w["wind_speed_10m"] * w["boundary_layer_height"]
        w["wind_speed_24h_mean"] = w["wind_speed_10m"].rolling(24, min_periods=12).mean()

    if "wind_direction_10m" in w.columns:
        # Direction is circular - 359 and 1 degrees are adjacent. Encode
        # as components so models can use it at all.
        rad = w["wind_direction_10m"] * 3.141592653589793 / 180.0
        w["wind_u"] = -w["wind_speed_10m"] * rad.apply(lambda x: __import__("math").sin(x)) \
            if "wind_speed_10m" in w.columns else None
        w["wind_v"] = -w["wind_speed_10m"] * rad.apply(lambda x: __import__("math").cos(x)) \
            if "wind_speed_10m" in w.columns else None

    return w


def join_to_features(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Joins weather onto features.csv on the floored UTC hour.

    Existing weather columns from the live pipeline (populated in 2 rows)
    are dropped in favour of the complete ERA5 series, so the column is
    consistent rather than a mix of two sources.
    """
    df = pd.read_csv(FEATURES_PATH)
    ts = pd.to_datetime(df["fetched_at"], format="mixed", errors="coerce", utc=True)
    df["_hour"] = ts.dt.floor("h")

    stale = [c for c in ["temperature", "humidity", "wind", "pressure"] if c in df.columns]
    if stale:
        print(f"  dropping sparse legacy weather columns: {', '.join(stale)}")
        df = df.drop(columns=stale)

    weather = weather.rename(columns={"time": "_hour"})
    merged = df.merge(weather, on="_hour", how="left")

    wcols = [c for c in weather.columns if c != "_hour"]
    coverage = merged[wcols].notna().all(axis=1).mean() * 100
    print(f"  weather coverage after join: {coverage:.1f}% of rows")
    if coverage < 90:
        print("  NOTE: rows near the present may be unfilled - ERA5 lags"
              " ~5 days behind real time.")

    return merged.drop(columns=["_hour"])


def main():
    ap = argparse.ArgumentParser(description="Backfill historical weather from Open-Meteo")
    ap.add_argument("--lat", type=float, default=CITY_LAT)
    ap.add_argument("--lon", type=float, default=CITY_LON)
    ap.add_argument("--dry-run", action="store_true",
                    help="Probe variables and print the plan, fetch nothing")
    ap.add_argument("--no-join", action="store_true",
                    help="Save weather_history.csv but leave features.csv untouched")
    args = ap.parse_args()

    try:
        start_date, end_date = target_date_range()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"Target window (from features.csv): {start_date} -> {end_date}")
    print(f"Location: {args.lat}, {args.lon}\n")

    try:
        variables = probe_variables(args.lat, args.lon)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(variables)} variables available")
    if args.dry_run:
        days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        print(f"Dry run: would fetch ~{days} days in "
              f"{days // CHUNK_DAYS + 1} chunk(s). Nothing written.")
        return

    print()
    weather = backfill_weather(args.lat, args.lon, start_date, end_date, variables)
    weather = derive_features(weather)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_PATH, index=False)
    print(f"\nWeather rows: {len(weather)}")
    print(f"Columns: {', '.join(c for c in weather.columns if c != 'time')}")
    print(f"Saved to: {WEATHER_PATH}")

    nulls = weather.drop(columns=["time"]).isna().mean() * 100
    worst = nulls[nulls > 1].sort_values(ascending=False)
    if len(worst):
        print("\nColumns with >1% missing:")
        for c, pct in worst.items():
            print(f"  {c:26s} {pct:5.1f}%")

    if args.no_join:
        print("\n--no-join set; features.csv unchanged.")
        return

    print("\nJoining onto features.csv...")
    backup = FEATURES_PATH.with_suffix(".pre_weather.csv")
    pd.read_csv(FEATURES_PATH).to_csv(backup, index=False)
    print(f"  backup written: {backup.name}")

    merged = join_to_features(weather)
    merged.to_csv(FEATURES_PATH, index=False)
    print(f"  features.csv: {len(merged)} rows, {len(merged.columns)} columns")
    print("\nNext: add the new weather columns to FEATURE_COLUMNS in")
    print("train_hourly.py, then re-run training to test whether they help.")


if __name__ == "__main__":
    main()
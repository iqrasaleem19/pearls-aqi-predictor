"""
Pulls HISTORICAL air pollution data from OpenWeather's history endpoint.

Hardened for long runs. A 300-day backfill is ~43 API calls and survives
sloppiness; a multi-year backfill is ~290 calls over several minutes,
where a single unhandled failure costs the whole run. Changes:

1. Timezone consistency. The original wrote tz-NAIVE timestamps
   (pd.to_datetime(dt, unit="s")) while the live hourly pipeline writes
   tz-AWARE ones, so features.csv accumulated both. Reading that back
   with parse_dates yields object dtype, and sort_values then raises
   TypeError comparing aware to naive Timestamps - a crash that would
   land AFTER every fetch completed, discarding the entire run. All
   timestamps are now tz-aware UTC on both read and write, which also
   permanently fixes the mixed-format parsing problem downstream.

2. Retries with backoff. Chunks previously failed silently via
   `except: continue`, each loss creating a 7-day hole that invalidates
   ~10 rows apiece through the lag/rolling windows. Now 3 attempts with
   exponential backoff, and any range that still fails is reported at
   the end so it can be re-run.

3. Checkpointing. Partial results are written every CHECKPOINT_EVERY
   chunks, so a crash or Ctrl-C at chunk 280 doesn't discard 279 chunks
   of work.

4. Hour-floored deduplication. Live rows land at 14:37:22 and
   backfilled rows at 14:00:00, so exact-match dedup let duplicate
   hours through. Dedup now keys on the floored hour, preferring live
   rows (which carry weather columns) over backfilled ones.

Note: OpenWeather's historical WEATHER data (temp/humidity/wind)
requires a paid plan, so backfilled rows carry pollutants only. Those
columns stay NaN and are populated going forward by the live hourly
pipeline. Documented limitation of the free tier - keep it in the
report. Open-Meteo offers free historical weather if you want to fill
this gap.

Usage:
    python -m src.feature_engineering.backfill --days 300
    python -m src.feature_engineering.backfill --days 2000
    python -m src.feature_engineering.backfill --days 2000 --chunk-days 14
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import OPENWEATHER_API_KEY, CITY_LAT, CITY_LON
from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25

HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
CHECKPOINT_PATH = PROCESSED_DIR / ".backfill_checkpoint.csv"

EARLIEST_AVAILABLE = datetime(2020, 11, 27, tzinfo=timezone.utc)

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
CHECKPOINT_EVERY = 20

WEATHER_COLUMNS = ["humidity", "pressure", "temperature", "wind"]


def _parse_timestamps_utc(series: pd.Series) -> pd.Series:
    """
    Parses a column mixing tz-naive and tz-aware strings, with and
    without microseconds, into one tz-aware UTC datetime64 Series.
    utc=True is load-bearing: without it, mixed offsets fall back to
    object dtype and every later .dt or sort call breaks.
    """
    return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)


def fetch_history(start: datetime, end: datetime,
                  lat: float = CITY_LAT, lon: float = CITY_LON) -> dict:
    """Calls the historical air pollution endpoint for a UTC datetime range."""
    params = {
        "lat": lat, "lon": lon,
        "start": int(start.timestamp()), "end": int(end.timestamp()),
        "appid": OPENWEATHER_API_KEY,
    }
    response = requests.get(HISTORY_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_history_with_retry(start: datetime, end: datetime) -> list:
    """
    Fetches one chunk, retrying transient failures. Returns [] only when
    every attempt failed, so the caller can record the lost range.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_history(start, end).get("list", [])
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    failed after {MAX_RETRIES} attempts: {e}", file=sys.stderr)
                return []
            wait = RETRY_BACKOFF ** attempt
            print(f"    attempt {attempt} failed ({e}); retrying in {wait:.0f}s")
            time.sleep(wait)
    return []


def history_entry_to_row(entry: dict, city_name: str) -> dict:
    """Converts one entry from the history 'list' array into a feature row."""
    components = entry["components"]
    # utc=True keeps backfilled rows tz-aware, matching the live pipeline.
    observed = pd.to_datetime(entry["dt"], unit="s", utc=True)
    pm25 = components.get("pm2_5")

    row = {
        "city": city_name,
        "fetched_at": observed,
        "station_timestamp": observed,  # history has no separate station time
        "aqi": calculate_aqi_from_pm25(pm25) if pm25 is not None else None,
        "openweather_aqi_scale": entry["main"]["aqi"],
        "pm25": pm25,
        "pm10": components.get("pm10"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "hour": observed.hour,
        "day": observed.day,
        "month": observed.month,
        "day_of_week": observed.dayofweek,
        "is_backfilled": True,
    }
    # Paid-tier only; populated going forward by the live pipeline.
    for c in WEATHER_COLUMNS:
        row[c] = None
    return row


def add_aqi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Row-over-row AQI change per city, computed after the full merge."""
    df = df.sort_values(["city", "fetched_at"]).reset_index(drop=True)
    df["aqi_change_rate"] = df.groupby("city")["aqi"].diff().fillna(0)
    return df


def merge_and_dedupe(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges on the floored hour rather than the exact timestamp.

    Live rows land at 14:37:22 and backfilled rows at 14:00:00, so exact
    matching treated them as different observations and let duplicate
    hours through. Live rows win ties because they carry weather data.
    """
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["fetched_at"] = _parse_timestamps_utc(combined["fetched_at"])
    if "station_timestamp" in combined.columns:
        combined["station_timestamp"] = _parse_timestamps_utc(combined["station_timestamp"])

    bad = int(combined["fetched_at"].isna().sum())
    if bad:
        print(f"  dropping {bad} rows with unparseable timestamps")
        combined = combined.dropna(subset=["fetched_at"])

    if "is_backfilled" not in combined.columns:
        combined["is_backfilled"] = False
    combined["is_backfilled"] = combined["is_backfilled"].fillna(False).astype(bool)

    combined["_hour_key"] = combined["fetched_at"].dt.floor("h")
    # Live rows (is_backfilled False) sort last, so keep="last" prefers them.
    combined = combined.sort_values(["city", "_hour_key", "is_backfilled"],
                                    ascending=[True, True, False])
    combined = combined.drop_duplicates(subset=["city", "_hour_key"], keep="last")
    return combined.drop(columns=["_hour_key"])


def load_existing() -> pd.DataFrame:
    """Reads features.csv as raw strings; parsing happens in merge_and_dedupe."""
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_PATH)


def backfill(days: int, city_name: str = "Karachi", chunk_days: int = 7,
             pause: float = 0.5) -> pd.DataFrame:
    """Pulls `days` of history in chunks and merges into features.csv."""
    end = datetime.now(timezone.utc)
    start = max(end - timedelta(days=days), EARLIEST_AVAILABLE)

    if start > end - timedelta(days=days):
        print(f"Note: clamped to earliest available data ({EARLIEST_AVAILABLE.date()})")

    total_chunks = int(((end - start).days / chunk_days) + 1)
    print(f"Backfilling {start.date()} -> {end.date()} "
          f"(~{total_chunks} chunks of {chunk_days}d)\n")

    all_rows, failed_ranges = [], []
    chunk_start, i = start, 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        i += 1
        print(f"[{i}/{total_chunks}] {chunk_start.date()} -> {chunk_end.date()}", end="")

        entries = fetch_history_with_retry(chunk_start, chunk_end)
        if entries:
            print(f"  ({len(entries)} readings)")
            all_rows.extend(history_entry_to_row(e, city_name) for e in entries)
        else:
            print("  FAILED")
            failed_ranges.append((chunk_start.date(), chunk_end.date()))

        if i % CHECKPOINT_EVERY == 0 and all_rows:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_rows).to_csv(CHECKPOINT_PATH, index=False)
            print(f"    checkpoint: {len(all_rows)} rows saved")

        chunk_start = chunk_end
        time.sleep(pause)

    if not all_rows:
        print("\nNo data retrieved.", file=sys.stderr)
        return load_existing()

    new_df = pd.DataFrame(all_rows)
    print(f"\nFetched {len(new_df)} new readings")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing()
    before = len(existing)

    combined = merge_and_dedupe(existing, new_df) if before else new_df
    combined = add_aqi_change_rate(combined)
    combined.to_csv(FEATURES_PATH, index=False)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"features.csv: {before} -> {len(combined)} rows")

    if failed_ranges:
        print(f"\n{len(failed_ranges)} chunk(s) failed - re-run to fill these gaps:")
        for s, e in failed_ranges:
            print(f"  {s} -> {e}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Backfill historical AQI features")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--city", type=str, default="Karachi")
    parser.add_argument("--chunk-days", type=int, default=7,
                        help="Days per API call. Raise to reduce call count.")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Seconds between calls")
    args = parser.parse_args()

    result = backfill(args.days, args.city, args.chunk_days, args.pause)
    if result.empty:
        sys.exit(1)

    ts = _parse_timestamps_utc(result["fetched_at"])
    span_days = (ts.max() - ts.min()).days
    print(f"\nTotal rows:  {len(result)}")
    print(f"Range:       {ts.min()} -> {ts.max()} ({span_days} days)")
    print(f"Coverage:    {len(result) / max(span_days * 24, 1) * 100:.1f}% of hourly slots")
    print(f"Saved to:    {FEATURES_PATH}")


if __name__ == "__main__":
    main()
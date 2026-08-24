"""
Turns a raw OpenWeather API response into a flat, structured feature row
and appends it to data/processed/features.csv.

Every hourly pull adds one row. Over time this file IS your training
dataset for the forecasting model.

Usage:
    python -m src.feature_engineering.build_features
"""

import sys
from pathlib import Path

import pandas as pd

from src.data_ingestion.fetch_openweather import fetch_combined, save_raw
from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.csv"


def extract_features(payload: dict) -> dict:
    """
    Flattens the merged OpenWeather (pollution + weather) response into
    a single dict of model-ready features, computing the EPA AQI from
    PM2.5 since OpenWeather doesn't provide it directly.
    """
    components = payload["pollution"]["list"][0]["components"]
    owm_aqi = payload["pollution"]["list"][0]["main"]["aqi"]  # OpenWeather's own 1-5 scale

    weather_main = payload["weather"]["main"]
    wind = payload["weather"]["wind"]
    city_name = payload["weather"].get("name", "Karachi")

    fetched_at = pd.to_datetime(payload["fetched_at_utc"]).tz_localize(None)
    station_timestamp = pd.to_datetime(payload["weather"]["dt"], unit="s")

    pm25 = components.get("pm2_5")
    epa_aqi = calculate_aqi_from_pm25(pm25)

    features = {
        "city": city_name,
        "fetched_at": fetched_at,
        "station_timestamp": station_timestamp,

        # Target / core signal
        "aqi": epa_aqi,
        "openweather_aqi_scale": owm_aqi,  # kept for reference, not the target

        # Pollutant readings
        "pm25": pm25,
        "pm10": components.get("pm10"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),

        # Weather readings
        "humidity": weather_main.get("humidity"),
        "pressure": weather_main.get("pressure"),
        "temperature": weather_main.get("temp"),
        "wind": wind.get("speed"),

        # Time-based features
        "hour": fetched_at.hour,
        "day": fetched_at.day,
        "month": fetched_at.month,
        "day_of_week": fetched_at.dayofweek,
    }

    return features


def add_derived_features(new_row: dict, history: pd.DataFrame) -> dict:
    """
    Adds features that depend on previous readings, e.g. AQI change rate.
    If there's no prior history for this city yet, defaults to 0.
    """
    city_history = history[history["city"] == new_row["city"]] if not history.empty else history

    if not city_history.empty:
        last_row = city_history.sort_values("fetched_at").iloc[-1]
        if pd.notna(last_row["aqi"]) and pd.notna(new_row["aqi"]):
            new_row["aqi_change_rate"] = new_row["aqi"] - last_row["aqi"]
        else:
            new_row["aqi_change_rate"] = 0
    else:
        new_row["aqi_change_rate"] = 0

    return new_row


def append_features(new_row: dict) -> pd.DataFrame:
    """
    Appends the new feature row to features.csv, creating the file if it
    doesn't exist yet. Returns the full updated dataframe.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURES_PATH.exists():
        history = pd.read_csv(FEATURES_PATH, parse_dates=["fetched_at", "station_timestamp"])
    else:
        history = pd.DataFrame()

    new_row = add_derived_features(new_row, history)

    updated = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    updated = updated.drop_duplicates(subset=["city", "fetched_at"], keep="last")
    updated = updated.sort_values("fetched_at")

    updated.to_csv(FEATURES_PATH, index=False)
    return updated


def main():
    try:
        payload = fetch_combined()
    except Exception as e:
        print(f"Failed to fetch OpenWeather data: {e}", file=sys.stderr)
        sys.exit(1)

    save_raw(payload)

    features = extract_features(payload)
    updated_history = append_features(features)

    print("New feature row:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    print(f"\nTotal rows in features.csv: {len(updated_history)}")
    print(f"Saved to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
"""
Fetches current air pollution + weather data for a given lat/lon from
OpenWeather's free API. Used as the primary data source since AQICN's
Karachi station has stopped reporting.

Docs:
  Air Pollution: https://openweathermap.org/api/air-pollution
  Weather:       https://openweathermap.org/current

Usage:
    python -m src.data_ingestion.fetch_openweather
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import OPENWEATHER_API_KEY, CITY_LAT, CITY_LON

AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_air_pollution(lat: float = CITY_LAT, lon: float = CITY_LON) -> dict:
    """Calls OpenWeather's Air Pollution endpoint and returns pollutant concentrations."""
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    response = requests.get(AIR_POLLUTION_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_weather(lat: float = CITY_LAT, lon: float = CITY_LON) -> dict:
    """Calls OpenWeather's current weather endpoint and returns temp/humidity/wind."""
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(WEATHER_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_combined(lat: float = CITY_LAT, lon: float = CITY_LON) -> dict:
    """
    Fetches both endpoints and merges them into one payload, so the rest
    of the pipeline only has to deal with a single dict.
    """
    pollution = fetch_air_pollution(lat, lon)
    weather = fetch_weather(lat, lon)

    return {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lon": lon,
        "pollution": pollution,
        "weather": weather,
    }


def save_raw(payload: dict) -> Path:
    """Saves the merged raw response to data/raw/ with a timestamped filename."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = RAW_DATA_DIR / f"openweather_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    return filepath


def summarize(payload: dict) -> None:
    """Prints a quick human-readable summary of the key fields."""
    components = payload["pollution"]["list"][0]["components"]
    weather_main = payload["weather"]["main"]
    wind = payload["weather"]["wind"]

    print(f"Location:    lat={payload['lat']}, lon={payload['lon']}")
    print(f"PM2.5:       {components.get('pm2_5')}")
    print(f"PM10:        {components.get('pm10')}")
    print(f"O3:          {components.get('o3')}")
    print(f"NO2:         {components.get('no2')}")
    print(f"SO2:         {components.get('so2')}")
    print(f"CO:          {components.get('co')}")
    print(f"Temp:        {weather_main.get('temp')}°C")
    print(f"Humidity:    {weather_main.get('humidity')}%")
    print(f"Pressure:    {weather_main.get('pressure')} hPa")
    print(f"Wind speed:  {wind.get('speed')} m/s")


def main():
    try:
        payload = fetch_combined()
    except Exception as e:
        print(f"Failed to fetch OpenWeather data: {e}", file=sys.stderr)
        sys.exit(1)

    summarize(payload)
    saved_path = save_raw(payload)
    print(f"\nRaw response saved to: {saved_path}")


if __name__ == "__main__":
    main()
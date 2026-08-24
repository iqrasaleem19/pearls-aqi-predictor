"""
Fetches current AQI + pollutant data for a given city from the AQICN API.

AQICN feed docs: https://aqicn.org/json-api/doc/
Endpoint used: https://api.waqi.info/feed/{city}/?token={token}

Usage:
    python -m src.data_ingestion.fetch_aqicn
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import AQICN_TOKEN, AQICN_CITY

BASE_URL = "https://api.waqi.info/feed"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_aqi(city: str = AQICN_CITY) -> dict:
    """
    Calls the AQICN API for the given city and returns the parsed JSON response.
    Raises an exception if the request fails or the API reports an error.
    """
    url = f"{BASE_URL}/{city}/"
    params = {"token": AQICN_TOKEN}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN API returned an error: {payload}")

    return payload


def save_raw(payload: dict, city: str) -> Path:
    """
    Saves the raw API response to data/raw/ with a timestamped filename,
    so we keep a full history of pulls for later feature engineering.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = RAW_DATA_DIR / f"{city}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    return filepath


def summarize(payload: dict) -> None:
    """Prints a quick human-readable summary of the key fields."""
    data = payload["data"]
    iaqi = data.get("iaqi", {})

    print(f"City:        {data.get('city', {}).get('name')}")
    print(f"AQI:         {data.get('aqi')}")
    print(f"Dominant pol: {data.get('dominentpol')}")
    print(f"Timestamp:   {data.get('time', {}).get('s')} ({data.get('time', {}).get('tz')})")
    print("Pollutant/weather readings:")
    for key, value in iaqi.items():
        print(f"  {key:6s}: {value.get('v')}")


def main():
    try:
        payload = fetch_aqi()
    except Exception as e:
        print(f"Failed to fetch AQI data: {e}", file=sys.stderr)
        sys.exit(1)

    summarize(payload)
    saved_path = save_raw(payload, AQICN_CITY)
    print(f"\nRaw response saved to: {saved_path}")


if __name__ == "__main__":
    main()
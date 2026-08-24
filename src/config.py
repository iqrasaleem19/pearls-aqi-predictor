"""
Central place for loading configuration/environment variables.
Every other module should import settings from here instead of
calling os.getenv() directly, so we have one source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly point at the .env file in the project root, regardless of
# how or from where this module is invoked (fixes -m module execution
# not reliably auto-finding .env). In CI there is no .env and the
# variables come from the environment directly, so a missing file is
# not an error.
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# AQICN is no longer the data source: Karachi's only station has been
# offline since March 2025, which is why the project moved to
# OpenWeather. Kept so fetch_aqicn.py still imports, but not required -
# raising here would break every module that imports config.
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
AQICN_CITY = os.getenv("AQICN_CITY", "karachi")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY_LAT = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON = float(os.getenv("CITY_LON", "67.0011"))

if not OPENWEATHER_API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY is not set. Get a free key at "
        "https://home.openweathermap.org/api_keys and add it to your "
        ".env file, or set it as a repository secret in CI."
    )
"""
Quick data quality check on features.csv before deciding how to handle
missing weather columns.

Usage:
    python -m src.feature_engineering.inspect_data
"""

import pandas as pd
from pathlib import Path

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"

df = pd.read_csv(FEATURES_PATH, parse_dates=["fetched_at", "station_timestamp"])

print(f"Total rows: {len(df)}")
print(f"Date range: {df['fetched_at'].min()} to {df['fetched_at'].max()}")
print()

print("Missing values per column:")
print(df.isna().sum())
print()

print("Sample of rows WITH weather data (should be your live pulls):")
print(df[df["temperature"].notna()][["fetched_at", "aqi", "pm25", "temperature", "humidity"]].head(10))
print()

print("Sample of rows WITHOUT weather data (backfilled):")
print(df[df["temperature"].isna()][["fetched_at", "aqi", "pm25", "temperature", "humidity"]].head(5))
print()

print("AQI distribution:")
print(df["aqi"].describe())
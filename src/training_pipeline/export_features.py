"""
Exports the feature group from Hopsworks to data/processed/features.csv.

Why this exists
---------------
train_hourly.py reads a local CSV. In CI there is no CSV - the repo
ignores it, and the feature store is the source of truth. This bridges
the two: pull the feature group, write it where the trainer expects it.

The alternative would be teaching train_hourly.py to read from Hopsworks
directly, but that would couple every local run to a network call and an
API key. Keeping the trainer file-based means it behaves identically on
a laptop and on a runner; only the source of the file differs.

Shape note: the feature group stores ENGINEERED features (lags, rolling
means, deviations) but not raw pollutant readings in their original
form, and not the targets - those are derived at training time. The
trainer's load_hourly() expects a raw-ish CSV with fetched_at and
pollutant columns, so this writes the columns it needs in the format it
expects, reconstructing fetched_at from the ts primary key.

Usage:
    python -m src.training_pipeline.export_features
    python -m src.training_pipeline.export_features --out other/path.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feature_store.hopsworks_sync import (  # noqa: E402
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, connect,
)
from src.training_pipeline.train_hourly import LOCAL_TZ  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "features.csv"

# The trainer recomputes everything downstream of these, so only the
# raw-ish inputs need exporting. Anything else it derives itself.
PASSTHROUGH = ["pm25", "pm10", "o3", "no2", "so2", "co"]


def export(out_path: Path) -> pd.DataFrame:
    """Reads the feature group and writes a CSV the trainer can load."""
    project = connect()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    if df.empty:
        raise RuntimeError(
            f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} is empty - "
            f"run the backfill and sync first"
        )

    missing = [c for c in PASSTHROUGH if c not in df.columns]
    if missing:
        raise KeyError(f"feature group is missing expected columns: {missing}")

    # ts is stored as tz-aware UTC. The trainer parses fetched_at with
    # utc=True and converts to local itself, so write it back as UTC to
    # keep one convention end to end. Three timezone bugs in this project
    # have come from two components disagreeing about this.
    ts = pd.to_datetime(df["ts"], utc=True)

    out = pd.DataFrame({"fetched_at": ts, "station_timestamp": ts})
    for c in PASSTHROUGH:
        out[c] = df[c]

    out = out.sort_values("fetched_at").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    local = ts.dt.tz_convert(LOCAL_TZ)
    print(f"  rows:  {len(out)}")
    print(f"  range: {local.min()} -> {local.max()} ({LOCAL_TZ})")
    print(f"  saved: {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Export the Hopsworks feature group to a local CSV")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"Exporting {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}...")
    try:
        export(args.out)
    except (RuntimeError, KeyError) as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
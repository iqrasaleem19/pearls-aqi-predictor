"""
US EPA AQI calculation from PM2.5 concentration.

Uses the pre-2024 (legacy) PM2.5 breakpoint table, matching what this
project has used throughout. Note in the report which standard was
used: EPA's 2024 revision moved the 50-AQI boundary from 12.0 to 9.0
ug/m3, so values computed under the two tables are not comparable in
the 0-100 range.

Fixed: boundary gaps
--------------------
The breakpoint table is defined on truncated concentrations - EPA
requires rounding PM2.5 DOWN to one decimal place before lookup. The
previous implementation compared raw values against the bands, so any
reading landing in the gap between two bands matched neither and fell
through to None:

    band 1 covers 0.0 - 12.0
    band 2 covers 12.1 - 35.4
    -> 12.03, 12.06, 12.09 match nothing

This silently produced 33 null AQI values in the post-2025-07 window
alone (and 180 across the full history), clustered at 12.0x, 35.4x,
55.4x and 150.4x. Truncating first puts 12.03 -> 12.0, which lands
cleanly in band 1 and returns AQI 50 as intended.
"""

import math

# (C_low, C_high, I_low, I_high) - concentrations in ug/m3.
# Bands are contiguous on TRUNCATED concentrations: a value truncated to
# one decimal always lands in exactly one band.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),        # Good
    (12.1, 35.4, 51, 100),     # Moderate
    (35.5, 55.4, 101, 150),    # Unhealthy for Sensitive Groups
    (55.5, 150.4, 151, 200),   # Unhealthy
    (150.5, 250.4, 201, 300),  # Very Unhealthy
    (250.5, 350.4, 301, 400),  # Hazardous
    (350.5, 500.4, 401, 500),  # Hazardous
]

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def calculate_aqi_from_pm25(pm25):
    """
    Converts a PM2.5 concentration (ug/m3) to a US EPA AQI value (0-500).

    Returns None only for missing or invalid input. Concentrations above
    the table's top band (500.4) are capped at 500, which is the maximum
    the AQI scale defines - returning None there would discard the most
    severe pollution events in the dataset.
    """
    if pm25 is None:
        return None
    try:
        value = float(pm25)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value < 0:
        return None

    # EPA: truncate to one decimal place before lookup. This is what
    # makes the bands contiguous - without it, values between bands
    # match nothing.
    c = math.floor(value * 10) / 10

    if c > PM25_BREAKPOINTS[-1][1]:
        return 500

    for c_low, c_high, i_low, i_high in PM25_BREAKPOINTS:
        if c_low <= c <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (c - c_low) + i_low
            return int(round(aqi))

    # Unreachable given contiguous bands plus the cap above, but kept so
    # a future table edit that reintroduces a gap fails loudly instead of
    # silently returning None.
    raise ValueError(f"PM2.5 value {value} matched no breakpoint band")


def get_aqi_category(aqi):
    """Returns the EPA health category label for an AQI value."""
    if aqi is None:
        return None
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Hazardous" if aqi > 500 else None


if __name__ == "__main__":
    # Self-test. The first group is the boundary-gap regression: every
    # one of these returned None before the truncation fix.
    gaps = [12.01, 12.03, 12.06, 12.09, 35.43, 35.45, 35.48,
            55.41, 55.42, 150.43, 150.48]
    print("boundary-gap values (previously None):")
    for v in gaps:
        print(f"  pm25={v:8.2f} -> AQI {calculate_aqi_from_pm25(v)}")

    # Second group pins the values this project has been producing all
    # along, so the fix is verifiably a drop-in.
    known = [(0.0, 0), (5.0, 21), (12.0, 50), (12.1, 51), (25.0, 78),
             (35.4, 100), (35.5, 101), (45.0, 124), (55.4, 150),
             (55.5, 151), (90.0, 169), (150.4, 200), (150.5, 201),
             (250.5, 301), (325.4, 375), (400.0, 434)]
    print("\nregression against known values:")
    bad = 0
    for v, expected in known:
        got = calculate_aqi_from_pm25(v)
        ok = got == expected
        bad += not ok
        print(f"  pm25={v:8.2f} -> {got:4d}  expected {expected:4d}"
              f"{'' if ok else '   <-- MISMATCH'}")

    print(f"\nedge cases: None->{calculate_aqi_from_pm25(None)}, "
          f"neg->{calculate_aqi_from_pm25(-5)}, "
          f"1965->{calculate_aqi_from_pm25(1965.05)}")
    print("SELF-TEST PASSED" if bad == 0 else f"{bad} MISMATCHES")
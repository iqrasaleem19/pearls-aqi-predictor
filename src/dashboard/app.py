"""
Karachi AQI forecast dashboard.

Loads the models trained by train_hourly.py, rebuilds the feature vector
from the current row of features.csv, and plots the forecast curve.

Design notes worth knowing before reading the code:

* The models predict a DELTA, not a level. Every forecast here is
  current_aqi + model.predict(features). The dashboard never displays a
  raw model output.

* Skill is shown honestly. Each horizon is annotated with its
  cross-validated RMSE and how it compares to the persistence baseline.
  At 12h the baseline wins outright, and the dashboard says so rather
  than presenting a model forecast as if it were better.

* Uncertainty bands come from the measured CV RMSE at each horizon, not
  from a model-internal estimate. A +-1 RMSE band is roughly a 68%
  interval if errors are near-normal. This is the honest width: at 72h
  it is wide enough to span two AQI categories, which is the real
  state of knowledge.

* Only the post-2025-07 regime is used, matching training. Hourly
  variability dropped ~9x around July 2025; pooling across that break
  would make the displayed history misleading.

Usage:
    streamlit run src/dashboard/app.py
"""

import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"
SUMMARY_PATH = MODELS_DIR / "hourly_training_summary.csv"

LOCAL_TZ = "Asia/Karachi"
REGIME_START = "2025-07-01"

# EPA categories: (upper_bound, label, colour, advice)
CATEGORIES = [
    (50, "Good", "#4a9d5f", "Air quality is satisfactory."),
    (100, "Moderate", "#c9a227", "Unusually sensitive people should consider limiting prolonged outdoor exertion."),
    (150, "Unhealthy for Sensitive Groups", "#d97706", "Children, older adults, and people with heart or lung disease should limit prolonged outdoor exertion."),
    (200, "Unhealthy", "#c2410c", "Everyone should limit prolonged outdoor exertion."),
    (300, "Very Unhealthy", "#7e22ce", "Everyone should avoid prolonged outdoor exertion."),
    (500, "Hazardous", "#7f1d1d", "Everyone should avoid all outdoor exertion."),
]


def categorise(aqi):
    """Returns (label, colour, advice) for an AQI value."""
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown", "#6b7280", ""
    for bound, label, colour, advice in CATEGORIES:
        if aqi <= bound:
            return label, colour, advice
    return CATEGORIES[-1][1], CATEGORIES[-1][2], CATEGORIES[-1][3]


def tomorrow_summary(fc, now_ts):
    """
    Averages the forecast points falling on the next calendar day.

    A single +24h reading is one hour's snapshot; "what is tomorrow like"
    is better answered by the mean across every hour we forecast on that
    date. Returns None when no forecast point lands on tomorrow.
    """
    if fc is None or len(fc) == 0:
        return None
    tmr = (now_ts + timedelta(days=1)).date()
    day = fc[fc["ts"].apply(lambda t: t.date() == tmr)]
    if day.empty:
        return None
    return {
        "date": tmr,
        "aqi": float(day["aqi"].mean()),
        "low": float(day["aqi"].min()),
        "high": float(day["aqi"].max()),
        "rmse": float(day["rmse"].mean()) if day["rmse"].notna().any() else float("nan"),
        "n": len(day),
    }


@st.cache_data(ttl=900)
def load_history():
    """Loads features.csv onto an hourly local-time grid, post-regime only."""
    if not FEATURES_PATH.exists():
        return None

    df = pd.read_csv(FEATURES_PATH)
    fetched = pd.to_datetime(df["fetched_at"], format="mixed", errors="coerce", utc=True)
    if "station_timestamp" in df.columns:
        observed = pd.to_datetime(df["station_timestamp"], format="mixed",
                                  errors="coerce", utc=True).fillna(fetched)
    else:
        observed = fetched

    df = df.loc[observed.notna()].copy()
    df["ts"] = observed[observed.notna()].dt.tz_convert(LOCAL_TZ).dt.floor("h").dt.tz_localize(None)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    h = df.groupby("ts")[numeric].mean().sort_index()

    # -9999 is OpenWeather's missing-data sentinel, not a concentration.
    for c in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        if c in h.columns:
            h.loc[h[c] < 0, c] = np.nan
    from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25
    pm24 = h["pm25"].rolling(24, min_periods=18).mean()
    h["aqi"] = pm24.apply(lambda v: np.nan if pd.isna(v) else calculate_aqi_from_pm25(v))
    from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25
    pm24 = h["pm25"].rolling(24, min_periods=18).mean()
    h["aqi"] = pm24.apply(lambda v: np.nan if pd.isna(v) else float(calculate_aqi_from_pm25(v)))
    h = h[h.index >= pd.Timestamp(REGIME_START)]
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    return h.reindex(full).rename_axis("ts")


@st.cache_resource
def load_models():
    """Loads every aqi_hourly_h*.pkl, keyed by horizon in hours."""
    models = {}
    for path in sorted(MODELS_DIR.glob("aqi_hourly_h*.pkl")):
        try:
            bundle = joblib.load(path)
            models[int(bundle["horizon_hours"])] = bundle
        except Exception as e:
            st.warning(f"Could not load {path.name}: {e}")
    return models


@st.cache_data(ttl=900)
def load_skill():
    """
    Reads the CV metrics so the dashboard can show measured error and
    say plainly where the naive baseline still wins.
    """
    if not SUMMARY_PATH.exists():
        return {}
    df = pd.read_csv(SUMMARY_PATH)
    skill = {}
    for hz, grp in df.groupby("horizon_h"):
        grp = grp[~grp["model"].astype(str).str.startswith("WINNER:")]
        best = grp.loc[grp["rmse"].idxmin()]
        base = grp[grp["model"] == "persistence"]
        base_rmse = float(base["rmse"].iloc[0]) if len(base) else float("nan")
        skill[int(hz)] = {
            "model": str(best["model"]),
            "rmse": float(best["rmse"]),
            "baseline_rmse": base_rmse,
            "beats_baseline": bool(best["rmse"] < base_rmse - 1e-9),
        }
    return skill


def build_feature_row(h: pd.DataFrame, features: list):
    """
    Rebuilds the engineered feature vector for the most recent complete
    hour. Mirrors add_features() in train_hourly.py exactly - if the two
    ever drift apart, forecasts silently degrade, which is the argument
    for moving this into a feature store.
    """
    a = h["aqi"]
    f = pd.DataFrame(index=h.index)

    for c in h.columns:
        f[c] = h[c]

    for L in (1, 2, 3, 24, 48, 168):
        f[f"aqi_lag{L}"] = a.shift(L)
    f["aqi_diff1"] = a.diff(1)
    f["aqi_diff3"] = a.diff(3)
    f["aqi_roll24"] = a.shift(1).rolling(24, min_periods=18).mean()
    f["aqi_roll72"] = a.shift(1).rolling(72, min_periods=48).mean()
    f["aqi_roll168"] = a.shift(1).rolling(168, min_periods=120).mean()
    f["aqi_minus_roll24"] = a - f["aqi_roll24"]
    f["aqi_minus_roll168"] = a - f["aqi_roll168"]
    same_hour = (a.shift(24) + a.shift(48) + a.shift(72)) / 3.0
    f["aqi_minus_hourly_norm"] = a - same_hour

    hour = f.index.hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["day_of_week"] = f.index.dayofweek

    missing = [c for c in features if c not in f.columns]
    if missing:
        return None, missing

    usable = f.dropna(subset=features)
    if usable.empty:
        return None, ["no row has a complete feature set"]
    return usable.iloc[[-1]], []


def forecast(models, row, current_aqi, skill):
    """Runs each horizon's model and returns a tidy forecast frame."""
    out = []
    for hz in sorted(models):
        bundle = models[hz]
        try:
            delta = float(bundle["model"].predict(row[bundle["features"]])[0])
        except Exception as e:
            st.warning(f"+{hz}h model failed: {e}")
            continue
        s = skill.get(hz, {})
        out.append({
            "horizon": hz,
            "ts": row.index[-1] + timedelta(hours=hz),
            "aqi": float(np.clip(current_aqi + delta, 0, 500)),
            "rmse": s.get("rmse", np.nan),
            "model": bundle.get("kind", "unknown"),
            "beats_baseline": s.get("beats_baseline", True),
        })
    return pd.DataFrame(out)


def main():
    st.set_page_config(page_title="Karachi AQI Forecast", page_icon="◐", layout="wide")

    st.markdown("""
        <style>
        .stApp { background: #0f1418; }
        h1, h2, h3, p, span, label { color: #e8eaed; }
        .metric-big { font-size: 3.4rem; font-weight: 700; line-height: 1; }
        .metric-label { font-size: 0.8rem; letter-spacing: 0.08em;
                        text-transform: uppercase; color: #8b949e; }
        .card { background: #161b22; border: 1px solid #21262d;
                border-radius: 10px; padding: 1.1rem 1.3rem; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Karachi air quality forecast")

    h = load_history()
    if h is None:
        st.error("features.csv not found. Run the feature pipeline first.")
        return

    models = load_models()
    if not models:
        st.error("No models in models/. Run train_hourly.py first.")
        return

    skill = load_skill()

    any_features = next(iter(models.values()))["features"]
    row, problems = build_feature_row(h, any_features)
    if row is None:
        st.error("Could not build a complete feature vector.")
        st.write(problems)
        return

    now_ts = row.index[-1]
    current = float(row["aqi"].iloc[0])
    label, colour, advice = categorise(current)

    fc = forecast(models, row, current, skill)

    if len(fc):
        st.markdown(
            f"<div style='color:#8b949e; font-size:.92rem; margin:-.5rem 0 1.1rem'>"
            f"Forecasting <b style='color:#e8eaed'>{now_ts:%a %d %b, %H:%M}</b> &rarr; "
            f"<b style='color:#e8eaed'>{fc['ts'].max():%a %d %b, %H:%M}</b> "
            f"&nbsp;&middot;&nbsp; {int(fc['horizon'].max())} hours ahead "
            f"&nbsp;&middot;&nbsp; {len(fc)} forecast point{'s' if len(fc) != 1 else ''} "
            f"&nbsp;&middot;&nbsp; times are {LOCAL_TZ.split('/')[-1]} local</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "No forecast produced. The models loaded but none returned a "
            "prediction - check the warnings above for a feature mismatch."
        )

    left, right = st.columns([1, 2.4])

    with left:
        st.markdown(
            f"""<div class="card">
                <div class="metric-label">Now &middot; {now_ts:%a %d %b, %H:%M}</div>
                <div class="metric-big" style="color:{colour}">{current:.0f}</div>
                <div style="color:{colour}; font-weight:600; margin-top:.3rem">{label}</div>
                <div style="color:#8b949e; font-size:.85rem; margin-top:.7rem">{advice}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        tmr = tomorrow_summary(fc, now_ts)
        if tmr:
            lab, col, adv = categorise(tmr["aqi"])
            spread = ("" if tmr["n"] < 2 else
                      f"Range {tmr['low']:.0f}&ndash;{tmr['high']:.0f} across "
                      f"{tmr['n']} forecast hours &middot; ")
            err = "" if np.isnan(tmr["rmse"]) else f"typical error &plusmn;{tmr['rmse']:.0f}"
            st.markdown(
                f"""<div class="card" style="margin-top:.9rem; border-color:{col}55">
                    <div class="metric-label">Tomorrow &middot; {tmr['date']:%a %d %b}</div>
                    <div style="font-size:2.5rem; font-weight:700; color:{col}; line-height:1.15">
                        {tmr['aqi']:.0f}</div>
                    <div style="color:{col}; font-weight:600">{lab}</div>
                    <div style="color:#8b949e; font-size:.8rem; margin-top:.45rem">
                        {spread}{err}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        if len(fc):
            st.markdown(
                "<div class='metric-label' style='margin:1.1rem 0 .5rem'>"
                "Forecast points</div>", unsafe_allow_html=True)
            for _, r in fc[fc["horizon"].isin([24, 48, 72])].iterrows():
                lab, col, _ = categorise(r["aqi"])
                band = "" if np.isnan(r["rmse"]) else f" &plusmn;{r['rmse']:.0f}"
                flag = ("" if r["beats_baseline"] else
                        "<div style='color:#6b7280; font-size:.72rem; margin-top:.35rem'>"
                        "No better than assuming no change</div>")
                st.markdown(
                    f"""<div class="card" style="margin-bottom:.5rem">
                        <span class="metric-label">{r['ts']:%a %d %b, %H:%M}
                            &nbsp;&middot;&nbsp; +{r['horizon']:.0f}h</span><br>
                        <span style="font-size:1.6rem; font-weight:700; color:{col}">
                            {r['aqi']:.0f}</span>
                        <span style="color:#6b7280">{band}</span>
                        <span style="color:{col}; font-size:.85rem; margin-left:.4rem">{lab}</span>
                        {flag}
                    </div>""",
                    unsafe_allow_html=True,
                )

    with right:
        hist = h["aqi"].loc[h.index >= h.index.max() - pd.Timedelta(days=7)].dropna()
        fig = go.Figure()

        for i, (bound, lab, col, _) in enumerate(CATEGORIES[:4]):
            low = 0 if i == 0 else CATEGORIES[i - 1][0]
            fig.add_hrect(y0=low, y1=bound, fillcolor=col, opacity=0.07,
                          line_width=0, layer="below")

        fig.add_trace(go.Scatter(
            x=hist.index, y=hist.values, name="Observed",
            line=dict(color="#58a6ff", width=1.8),
        ))

        if len(fc):
            fx = [now_ts] + list(fc["ts"])
            fy = [current] + list(fc["aqi"])
            err = [0] + list(fc["rmse"].fillna(0))

            fig.add_trace(go.Scatter(
                x=fx + fx[::-1],
                y=list(np.add(fy, err)) + list(np.subtract(fy, err))[::-1],
                fill="toself", fillcolor="rgba(210,168,80,0.13)",
                line=dict(width=0), hoverinfo="skip",
                name="Uncertainty (±1 RMSE)",
            ))
            fig.add_trace(go.Scatter(
                x=fx, y=fy, name="Forecast",
                line=dict(color="#d2a850", width=2.4, dash="dot"),
                mode="lines+markers", marker=dict(size=6),
            ))

        fig.update_layout(
            height=430, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"),
            xaxis=dict(gridcolor="#21262d", title=None),
            yaxis=dict(gridcolor="#21262d", title="AQI"),
            legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### How much to trust this")

    if len(fc):
        tbl = fc[["horizon", "aqi", "model", "rmse", "beats_baseline"]].copy()
        tbl.columns = ["Hours ahead", "Forecast AQI", "Model", "Typical error (RMSE)", "Beats naive baseline"]
        tbl["Forecast AQI"] = tbl["Forecast AQI"].round(0)
        tbl["Typical error (RMSE)"] = tbl["Typical error (RMSE)"].round(1)
        tbl["Beats naive baseline"] = tbl["Beats naive baseline"].map({True: "yes", False: "no"})
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.caption(
        "Errors are cross-validated on held-out future data, not on the training set. "
        "Where 'beats naive baseline' reads no, assuming air quality stays as it is now "
        "is as accurate as the model — the forecast is shown for continuity, not because "
        "it knows better. Bands are ±1 RMSE, roughly a 68% interval. "
        "Pollutants: OpenWeather (CAMS). Weather: Open-Meteo (ERA5). "
        f"History shown from {REGIME_START}, when the pollutant feed's hour-to-hour "
        "variability changed sharply."
    )


if __name__ == "__main__":
    main()
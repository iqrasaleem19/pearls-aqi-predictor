"""
Karachi air quality forecast - dashboard.

Design brief
------------
Subject: a 72-hour AQI forecast for one coastal megacity, read by people
deciding whether to run errands, mask up, or keep a child indoors.

The honest centre of this project is that forecast skill DECAYS - around
+6h the models beat a naive baseline by roughly 40%, by +72h only a few
percent. Most dashboards hide that behind three equally confident
numbers. Here it is the signature: the forecast line dissolves into haze
as it extends, the uncertainty ribbon widens with measured error, and
each horizon carries its real skill against "assume nothing changes".

Palette is Karachi coastal dusk - deep marine ink, sea-haze grey, a warm
sodium-lamp accent for the forecast. Category colours are desaturated
from the usual traffic-light set so they read as information, not alarm.
Fraunces carries the headline and the single big reading; IBM Plex Mono
carries every number, because this is an instrument readout and numerals
should align in columns.

Data note: AQI is computed from the 24-HOUR ROLLING MEAN PM2.5
concentration, matching the EPA definition and the target the models
were trained on. Feeding instantaneous AQI to these models would
silently corrupt every prediction.

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

from src.feature_engineering.aqi_calculator import calculate_aqi_from_pm25  # noqa: E402
# Custom estimators must be importable before joblib can unpickle the
# saved bundles - pickle stores the class path, not the class itself.
from src.training_pipeline.train_hourly import (  # noqa: E402,F401
    MeanEnsemble, BoostedResidual, ZeroDeltaBaseline,
    DiurnalNaive, DampedReversion,
)

FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"
SUMMARY_PATH = MODELS_DIR / "hourly_training_summary.csv"
EXPLAIN_DIR = ROOT / "reports" / "explainability"

LOCAL_TZ = "Asia/Karachi"
REGIME_START = "2025-07-01"

INK = "#0a1420"
SURFACE = "#111e2c"
EDGE = "#1e3346"
HAZE = "#7c93a8"
BONE = "#e6ecf1"
SODIUM = "#e0a355"
# HAZE is for decorative chrome only. Anything a reader has to actually
# read - labels, units, captions - uses LABEL, which clears WCAG AA
# against the ink background where HAZE does not.
LABEL = "#a6bccf"

CATEGORIES = [
    (50, "Good", "#5b9c78", "Fine for outdoor activity."),
    (100, "Moderate", "#c9a94e",
     "Unusually sensitive people may want to limit long spells outdoors."),
    (150, "Unhealthy for sensitive groups", "#d18448",
     "Children, older adults and people with heart or lung conditions should "
     "ease off prolonged exertion outdoors."),
    (200, "Unhealthy", "#c25f4e",
     "Everyone should cut back on prolonged exertion outdoors."),
    (300, "Very unhealthy", "#9a5b8c", "Avoid prolonged exertion outdoors."),
    (500, "Hazardous", "#8d4a52", "Stay indoors where you can."),
]

POLLUTANTS = [
    ("pm25", "PM2.5", "µg/m³"), ("pm10", "PM10", "µg/m³"),
    ("o3", "Ozone", "µg/m³"), ("no2", "NO₂", "µg/m³"),
    ("so2", "SO₂", "µg/m³"), ("co", "CO", "µg/m³"),
]

WEATHER = [
    ("temperature_2m", "Temperature", "°C"),
    ("relative_humidity_2m", "Humidity", "%"),
    ("wind_speed_10m", "Wind", "km/h"),
    ("surface_pressure", "Pressure", "hPa"),
]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,600&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

.stApp {{
    background:
        radial-gradient(1100px 520px at 12% -12%, #16283a 0%, transparent 62%),
        {INK};
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2.4rem; max-width: 1200px; }}
/* Streamlit applies its own colour and opacity to markdown wrappers,
   which washes out mono numerals set at light weights. */
.stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"] {{
    color: {BONE}; opacity: 1;
}}
.cell .v, .metric .v, .day .avg, .tomorrow .val, .hrow .num {{
    -webkit-font-smoothing: antialiased;
}}
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', system-ui, sans-serif; color: {BONE};
}}

.masthead {{
    display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap;
    border-bottom: 1px solid {EDGE}; padding-bottom: .85rem;
}}
.masthead h1 {{
    font-family: 'Fraunces', Georgia, serif; font-weight: 300;
    font-size: 2.05rem; letter-spacing: -.015em; margin: 0; color: {BONE};
}}
.masthead .place {{
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
    letter-spacing: .22em; text-transform: uppercase; color: {LABEL};
}}
.window {{
    font-family: 'IBM Plex Mono', monospace; font-size: .74rem;
    color: {LABEL}; letter-spacing: .04em; margin: .6rem 0 1.5rem;
}}
.window b {{ color: {BONE}; font-weight: 400; }}

.now {{
    background: linear-gradient(160deg, {SURFACE} 0%, #0d1a27 100%);
    border: 1px solid {EDGE}; border-radius: 3px; padding: 1.4rem 1.5rem 1.3rem;
}}
.eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: .66rem;
    letter-spacing: .2em; text-transform: uppercase; color: {LABEL};
}}
.reading {{
    font-family: 'Fraunces', Georgia, serif; font-weight: 300;
    font-size: 4.9rem; line-height: .92; letter-spacing: -.03em;
    margin: .45rem 0 .1rem;
}}
.band {{ font-size: .94rem; font-weight: 500; margin-bottom: .45rem; }}
.guide {{ font-size: .85rem; color: {LABEL}; line-height: 1.55; }}

.scale {{ display: flex; gap: 2px; margin: .95rem 0 .2rem; }}
.scale span {{ flex: 1; height: 3px; border-radius: 1px; }}
.scale-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: .62rem;
    color: {LABEL}; letter-spacing: .1em; display: flex;
    justify-content: space-between;
}}

.tomorrow {{
    background: {SURFACE}; border: 1px solid {EDGE};
    border-left: 2px solid {SODIUM}; border-radius: 3px;
    padding: 1rem 1.15rem; margin-top: .8rem;
}}
.tomorrow .val {{
    font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem;
    font-weight: 400; line-height: 1.15; margin-top: .25rem;
}}
.tomorrow .spread {{
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
    color: {LABEL}; margin-top: .28rem;
}}

.section {{
    font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
    letter-spacing: .2em; text-transform: uppercase; color: {LABEL};
    margin: 2rem 0 .75rem; padding-bottom: .4rem;
    border-bottom: 1px solid {EDGE};
}}

.grid {{ display: flex; flex-wrap: wrap; gap: .6rem; }}
.cell {{
    flex: 1 1 150px; background: {SURFACE}; border: 1px solid {EDGE};
    border-radius: 3px; padding: .8rem .9rem;
}}
.cell .k {{
    font-family: 'IBM Plex Mono', monospace; font-size: .63rem;
    letter-spacing: .14em; text-transform: uppercase; color: {LABEL};
}}
.cell .v {{
    font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem;
    font-weight: 400; color: {BONE}; margin-top: .3rem;
}}
.cell .u {{ font-size: .68rem; color: {LABEL}; margin-left: .3rem; }}

.day {{
    flex: 1 1 180px; background: {SURFACE}; border: 1px solid {EDGE};
    border-top: 2px solid; border-radius: 3px; padding: .95rem 1rem;
}}
.day .d {{
    font-family: 'IBM Plex Mono', monospace; font-size: .66rem;
    letter-spacing: .16em; text-transform: uppercase; color: {LABEL};
}}
.day .avg {{
    font-family: 'IBM Plex Mono', monospace; font-size: 2rem;
    font-weight: 400; line-height: 1.2; margin: .3rem 0 .1rem;
}}
.day .lab {{ font-size: .78rem; font-weight: 500; margin-bottom: .5rem; }}
.day .rng {{
    font-family: 'IBM Plex Mono', monospace; font-size: .69rem; color: {LABEL};
}}

.rows {{ margin-top: .2rem; }}
.hrow {{
    display: grid; grid-template-columns: 168px 1fr 58px;
    align-items: center; gap: .9rem; padding: .58rem 0;
    border-bottom: 1px solid {EDGE};
}}
.hrow:last-child {{ border-bottom: none; }}
.hrow .when {{
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: {LABEL};
}}
.hrow .bar {{ height: 3px; background: {EDGE}; border-radius: 2px; }}
.hrow .fill {{
    display: block; height: 3px; border-radius: 2px; background: {SODIUM};
}}
.hrow .num {{
    font-family: 'IBM Plex Mono', monospace; font-size: 1.02rem;
    font-weight: 400; text-align: right;
}}

.model {{
    background: {SURFACE}; border: 1px solid {EDGE}; border-radius: 3px;
    padding: 1.05rem 1.2rem; display: flex; flex-wrap: wrap; gap: 2.2rem;
    align-items: flex-start;
}}
.model .name {{
    font-family: 'Fraunces', Georgia, serif; font-size: 1.3rem;
    font-weight: 300; color: {BONE};
}}
.model .sub {{
    font-family: 'IBM Plex Mono', monospace; font-size: .66rem;
    color: {LABEL}; letter-spacing: .1em; margin-top: .2rem;
}}
.metric .k {{
    font-family: 'IBM Plex Mono', monospace; font-size: .62rem;
    letter-spacing: .14em; text-transform: uppercase; color: {LABEL};
}}
.metric .v {{
    font-family: 'IBM Plex Mono', monospace; font-size: 1.38rem;
    font-weight: 400; color: {BONE}; margin-top: .25rem;
}}

.feat {{
    display: grid; grid-template-columns: 190px 1fr; align-items: center;
    gap: .8rem; padding: .3rem 0;
}}
.feat .fn {{
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: {LABEL};
}}
.feat .fb {{ height: 9px; border-radius: 1px; background: {SODIUM}; opacity: .8; }}

.note {{
    font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
    color: {LABEL}; letter-spacing: .03em; margin-top: .55rem;
}}
.foot {{
    font-size: .78rem; color: {LABEL}; line-height: 1.7;
    margin-top: 1.1rem; max-width: 78ch;
}}
</style>
"""


def categorise(aqi):
    """Returns (label, colour, guidance) for an AQI value."""
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown", HAZE, ""
    for bound, label, colour, guide in CATEGORIES:
        if aqi <= bound:
            return label, colour, guide
    return CATEGORIES[-1][1], CATEGORIES[-1][2], CATEGORIES[-1][3]


@st.cache_data(ttl=900)
def load_history():
    """Loads features.csv onto an hourly local-time grid, post-regime only."""
    if not FEATURES_PATH.exists():
        return None

    df = pd.read_csv(FEATURES_PATH)
    fetched = pd.to_datetime(df["fetched_at"], format="mixed",
                             errors="coerce", utc=True)
    if "station_timestamp" in df.columns:
        observed = pd.to_datetime(df["station_timestamp"], format="mixed",
                                  errors="coerce", utc=True).fillna(fetched)
    else:
        observed = fetched

    df = df.loc[observed.notna()].copy()
    df["ts"] = (observed[observed.notna()].dt.tz_convert(LOCAL_TZ)
                .dt.floor("h").dt.tz_localize(None))

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    h = df.groupby("ts")[numeric].mean().sort_index()

    # -9999 is OpenWeather's missing-data sentinel, not a concentration.
    for c, _, _ in POLLUTANTS:
        if c in h.columns:
            h.loc[h[c] < 0, c] = np.nan

    h = h[h.index >= pd.Timestamp(REGIME_START)]
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(full).rename_axis("ts")
    for c, _, _ in POLLUTANTS:
        if c in h.columns:
            h[c] = h[c].interpolate(limit=3, limit_area="inside")

    # Must match train_hourly's target mode: AQI from the 24h rolling mean
    # concentration, the EPA definition. Instantaneous AQI would put the
    # models on a different quantity than they trained on.
    pm24 = h["pm25"].rolling(24, min_periods=18).mean()
    h["aqi"] = pm24.apply(
        lambda v: np.nan if pd.isna(v) else float(calculate_aqi_from_pm25(v)))
    return h


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
    """Cross-validated error per horizon, and the gain over the baseline."""
    if not SUMMARY_PATH.exists():
        return {}
    df = pd.read_csv(SUMMARY_PATH)
    out = {}
    for hz, grp in df.groupby("horizon_h"):
        grp = grp[~grp["model"].astype(str).str.startswith("WINNER:")]
        best = grp.loc[grp["rmse"].idxmin()]
        base = grp[grp["model"] == "persistence"]
        base_rmse = float(base["rmse"].iloc[0]) if len(base) else float("nan")
        gain = ((base_rmse - float(best["rmse"])) / base_rmse * 100
                if base_rmse == base_rmse and base_rmse > 0 else float("nan"))
        out[int(hz)] = {
            "model": str(best["model"]), "rmse": float(best["rmse"]),
            "mae": float(best["mae"]) if "mae" in best else float("nan"),
            "r2": float(best["r2"]) if "r2" in best else float("nan"),
            "baseline_rmse": base_rmse, "gain": gain,
        }
    return out


@st.cache_data(ttl=3600)
def load_importance():
    """Feature importance from the explainability run, if it has been done."""
    for name in ("shap_importance_h24.csv", "ridge_coefficients.csv"):
        p = EXPLAIN_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        s = df.iloc[:, 0].abs() if len(df.columns) else None
        if s is None or s.empty:
            continue
        return s.sort_values(ascending=False).head(10), name
    return None, None


def build_feature_row(h: pd.DataFrame, features: list):
    """Rebuilds the engineered feature vector for the latest usable hour."""
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
    f["aqi_minus_hourly_norm"] = a - (a.shift(24) + a.shift(48) + a.shift(72)) / 3.0

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
        return None, ["no hour has a complete feature set"]
    return usable.iloc[[-1]], []


def forecast(models, row, current_aqi, skill):
    """Runs each horizon's model. Models predict a delta, not a level."""
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
            "horizon": hz, "ts": row.index[-1] + timedelta(hours=hz),
            "aqi": float(np.clip(current_aqi + delta, 0, 500)),
            "rmse": s.get("rmse", np.nan), "gain": s.get("gain", np.nan),
            "model": s.get("model", bundle.get("kind", "")),
        })
    return pd.DataFrame(out)


def daily_outlook(fc, now_ts, current):
    """Groups forecast points into calendar days for the outlook cards."""
    if fc is None or len(fc) == 0:
        return []
    rows = pd.concat([
        pd.DataFrame([{"ts": now_ts, "aqi": current, "rmse": 0.0}]),
        fc[["ts", "aqi", "rmse"]],
    ], ignore_index=True)
    rows["date"] = rows["ts"].apply(lambda t: t.date())

    out = []
    for d, g in rows.groupby("date"):
        out.append({
            "date": d, "avg": float(g["aqi"].mean()),
            "low": float(g["aqi"].min()), "high": float(g["aqi"].max()),
            "err": float(g["rmse"].mean()), "n": len(g),
            "today": d == now_ts.date(),
        })
    return sorted(out, key=lambda r: r["date"])


def curve_figure(hist, fc, now_ts, current):
    """
    The signature element: the forecast fades as skill decays.

    Drawn as consecutive segments whose opacity falls with horizon, with
    a ribbon that widens by the measured error. By +72h the trace has
    visibly dissolved - the honest picture, since at that range the model
    is only a few percent better than assuming nothing changes.
    """
    fig = go.Figure()

    for i, (bound, _, colour, _) in enumerate(CATEGORIES[:4]):
        low = 0 if i == 0 else CATEGORIES[i - 1][0]
        fig.add_hrect(y0=low, y1=bound, fillcolor=colour, opacity=0.055,
                      line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Observed",
        line=dict(color=BONE, width=1.6),
        hovertemplate="%{x|%a %H:%M} · %{y:.0f}<extra></extra>"))

    if len(fc):
        fx = [now_ts] + list(fc["ts"])
        fy = [current] + list(fc["aqi"])
        err = [0.0] + list(fc["rmse"].fillna(0))

        fig.add_trace(go.Scatter(
            x=fx + fx[::-1],
            y=list(np.add(fy, err)) + list(np.subtract(fy, err))[::-1],
            fill="toself", fillcolor="rgba(224,163,85,0.10)",
            line=dict(width=0), hoverinfo="skip", showlegend=False))

        n = len(fx) - 1
        for i in range(n):
            fig.add_trace(go.Scatter(
                x=fx[i:i + 2], y=fy[i:i + 2], mode="lines",
                line=dict(color=SODIUM, width=2.2, dash="dot"),
                opacity=max(0.18, 1.0 - 0.85 * (i / max(n - 1, 1))),
                showlegend=(i == 0), name="Forecast", hoverinfo="skip"))

        fig.add_trace(go.Scatter(
            x=fx[1:], y=fy[1:], mode="markers",
            marker=dict(size=6, color=SODIUM,
                        opacity=[max(0.25, 1.0 - 0.8 * (i / max(n - 1, 1)))
                                 for i in range(n)]),
            showlegend=False,
            hovertemplate="%{x|%a %H:%M} · %{y:.0f}<extra></extra>"))

    fig.update_layout(
        height=400, margin=dict(l=8, r=8, t=26, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=HAZE, family="IBM Plex Mono", size=11),
        xaxis=dict(gridcolor=EDGE, zeroline=False, title=None),
        yaxis=dict(gridcolor=EDGE, zeroline=False, title=None, ticksuffix="  "),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified")
    return fig


def cells(items, latest):
    """Renders a row of reading cells, skipping anything unmeasured."""
    parts = ['<div class="grid">']
    any_shown = False
    for key, label, unit in items:
        if key not in latest.index or pd.isna(latest[key]):
            continue
        any_shown = True
        val = float(latest[key])
        fmt = f"{val:,.0f}" if abs(val) >= 100 else f"{val:.1f}"
        parts.append(
            f'<div class="cell"><div class="k">{label}</div>'
            f'<div class="v">{fmt}<span class="u">{unit}</span></div></div>')
    parts.append("</div>")
    return "".join(parts) if any_shown else None


def main():
    st.set_page_config(page_title="Karachi air quality forecast",
                       page_icon="◐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    h = load_history()
    if h is None:
        st.error("No features.csv found. Run the feature pipeline first.")
        return
    models = load_models()
    if not models:
        st.error("No models in models/. Run the training pipeline first.")
        return

    skill = load_skill()
    feats = next(iter(models.values()))["features"]
    row, problems = build_feature_row(h, feats)
    if row is None:
        st.error("Could not assemble a complete feature vector.")
        st.write(problems)
        return

    now_ts = row.index[-1]
    latest = row.iloc[0]
    current = float(latest["aqi"])
    label, colour, guide = categorise(current)
    fc = forecast(models, row, current, skill)

    st.markdown(
        '<div class="masthead"><h1>Air quality forecast</h1>'
        '<span class="place">Karachi · 24.86 N 67.00 E</span></div>',
        unsafe_allow_html=True)

    if len(fc):
        st.markdown(
            f'<div class="window">Reading at <b>{now_ts:%a %d %b, %H:%M}</b>'
            f'&nbsp; → &nbsp;forecast to '
            f'<b>{fc["ts"].max():%a %d %b, %H:%M}</b>'
            f'&nbsp; · &nbsp;{int(fc["horizon"].max())} hours ahead'
            f'&nbsp; · &nbsp;{len(fc)} points&nbsp; · &nbsp;local time</div>',
            unsafe_allow_html=True)

    left, right = st.columns([1, 2.1], gap="large")

    with left:
        scale = "".join(
            f'<span style="background:{c};opacity:'
            f'{1.0 if lab == label else 0.22}"></span>'
            for _, lab, c, _ in CATEGORIES)
        st.markdown(
            f'<div class="now"><div class="eyebrow">Now</div>'
            f'<div class="reading" style="color:{colour}">{current:.0f}</div>'
            f'<div class="band" style="color:{colour}">{label}</div>'
            f'<div class="guide">{guide}</div>'
            f'<div class="scale">{scale}</div>'
            f'<div class="scale-label"><span>0</span><span>500</span></div>'
            f'</div>', unsafe_allow_html=True)

        tmr = [d for d in daily_outlook(fc, now_ts, current) if not d["today"]]
        if tmr:
            t = tmr[0]
            tl, tc, _ = categorise(t["avg"])
            st.markdown(
                f'<div class="tomorrow">'
                f'<div class="eyebrow">Tomorrow · {t["date"]:%a %d %b}</div>'
                f'<div class="val" style="color:{tc}">{t["avg"]:.0f}</div>'
                f'<div class="band" style="color:{tc}">{tl}</div>'
                f'<div class="spread">{t["low"]:.0f}–{t["high"]:.0f} across '
                f'{t["n"]} points · typical error ±{t["err"]:.0f}</div></div>',
                unsafe_allow_html=True)

    with right:
        cutoff = h.index.max() - pd.Timedelta(days=5)
        hist = h["aqi"].loc[h.index >= cutoff].dropna()
        st.plotly_chart(curve_figure(hist, fc, now_ts, current),
                        use_container_width=True,
                        config={"displayModeBar": False})

    days = daily_outlook(fc, now_ts, current)
    if len(days) > 1:
        st.markdown('<div class="section">Day by day</div>',
                    unsafe_allow_html=True)
        parts = ['<div class="grid">']
        for d in days:
            lab, col, _ = categorise(d["avg"])
            when = "Today" if d["today"] else f'{d["date"]:%a %d %b}'
            parts.append(
                f'<div class="day" style="border-top-color:{col}">'
                f'<div class="d">{when}</div>'
                f'<div class="avg" style="color:{col}">{d["avg"]:.0f}</div>'
                f'<div class="lab" style="color:{col}">{lab}</div>'
                f'<div class="rng">{d["low"]:.0f}–{d["high"]:.0f}'
                f'&nbsp; · &nbsp;±{d["err"]:.0f}</div></div>')
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

    if len(fc):
        st.markdown('<div class="section">How far ahead, and how certain</div>',
                    unsafe_allow_html=True)
        worst = max(float(fc["rmse"].max()), 1.0)
        parts = ['<div class="rows">']
        for _, r in fc.iterrows():
            _, c, _ = categorise(r["aqi"])
            width = min(100.0, float(r["rmse"]) / worst * 100.0)
            parts.append(
                f'<div class="hrow"><span class="when">'
                f'+{int(r["horizon"])}h&nbsp; · &nbsp;{r["ts"]:%a %d %b %H:%M}'
                f'</span><span class="bar"><span class="fill" '
                f'style="width:{width:.0f}%"></span></span>'
                f'<span class="num" style="color:{c}">{r["aqi"]:.0f}</span>'
                f'</div>')
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)
        st.markdown('<div class="note">Bar length is the typical error at that '
                    'range — longer means less certain.</div>',
                    unsafe_allow_html=True)

    pol = cells(POLLUTANTS, latest)
    if pol:
        st.markdown('<div class="section">Pollutants now</div>',
                    unsafe_allow_html=True)
        st.markdown(pol, unsafe_allow_html=True)

    wx = cells(WEATHER, latest)
    if wx:
        st.markdown('<div class="section">Conditions</div>',
                    unsafe_allow_html=True)
        st.markdown(wx, unsafe_allow_html=True)

    if skill:
        st.markdown('<div class="section">Model in service</div>',
                    unsafe_allow_html=True)
        ref = skill.get(24) or skill[min(skill)]
        hz_ref = 24 if 24 in skill else min(skill)
        gain = ("" if ref["gain"] != ref["gain"]
                else f'{ref["gain"]:+.1f}% vs naive')
        metrics = "".join(
            f'<div class="metric"><div class="k">{k}</div>'
            f'<div class="v">{v}</div></div>'
            for k, v in [
                ("RMSE", f'{ref["rmse"]:.2f}'),
                ("MAE", f'{ref["mae"]:.2f}' if ref["mae"] == ref["mae"] else "—"),
                ("R²", f'{ref["r2"]:.3f}' if ref["r2"] == ref["r2"] else "—"),
                ("Skill", gain or "—"),
            ])
        st.markdown(
            f'<div class="model"><div><div class="name">'
            f'{ref["model"].replace("_", " ").title()}</div>'
            f'<div class="sub">selected at +{hz_ref}h · predicts change, '
            f'not level</div></div>{metrics}</div>',
            unsafe_allow_html=True)
        st.markdown('<div class="note">Scores come from expanding-window '
                    'cross-validation: every test hour falls after the data '
                    'the model was fitted on.</div>', unsafe_allow_html=True)

    imp, src = load_importance()
    if imp is not None and len(imp):
        st.markdown('<div class="section">What drives the forecast</div>',
                    unsafe_allow_html=True)
        top = float(imp.max()) or 1.0
        parts = []
        for name, val in imp.items():
            w = max(2.0, float(val) / top * 100.0)
            parts.append(
                f'<div class="feat"><span class="fn">{name}</span>'
                f'<span class="fb" style="width:{w:.0f}%"></span></div>')
        st.markdown("".join(parts), unsafe_allow_html=True)
        kind = ("SHAP attribution, gradient boosting"
                if "shap" in src else "Ridge coefficient magnitude")
        st.markdown(f'<div class="note">{kind}. Recent AQI history dominates; '
                    'raw pollutant readings add less than they appear to, '
                    'because they move together.</div>',
                    unsafe_allow_html=True)

    st.markdown(
        '<div class="foot">'
        'AQI follows the US EPA scale, computed from the 24-hour average '
        'PM2.5 concentration. Errors are cross-validated on future data the '
        'models never saw, and the shaded band is one typical error either '
        'side. Skill falls with distance: a day ahead the models clearly beat '
        'assuming nothing changes, three days ahead they barely do, which is '
        'why the forecast line fades as it runs out. Pollutants from '
        'OpenWeather (CAMS); weather from Open-Meteo (ERA5). History shown '
        f'from {REGIME_START}, when the pollutant feed\'s hour-to-hour '
        'variability changed sharply.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
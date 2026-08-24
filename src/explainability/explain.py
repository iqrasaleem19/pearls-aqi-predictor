"""
Explainability for the AQI forecasting models.

Two complementary analyses, because the deployed models are linear and
the interesting question is not just "which features matter" but "why
does a linear model win at all".

1. COEFFICIENTS of the deployed linear models.
   For Ridge and ElasticNet the coefficients ARE the explanation - SHAP
   on a linear model just recovers coefficient x (value - mean), so it
   adds nothing. What is informative is how the coefficient profile
   SHIFTS with horizon: short horizons should lean on recent lags,
   long horizons on the weekly rolling mean. That shift is the model
   learning that momentum decays and mean reversion takes over.

2. SHAP on hist_gbr.
   The tree model loses to linear models at every horizon. SHAP
   dependence plots test why. If the relationship between each feature
   and the target is close to a straight line with little interaction
   structure, there is no nonlinearity for trees to exploit, which
   explains the whole results table rather than just decorating it.

   This is the honest framing: explainability applied to the model we
   did NOT deploy, in order to justify the one we did.

Outputs go to reports/explainability/.

Usage:
    python -m src.explainability.explain
    python -m src.explainability.explain --horizon 24
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training_pipeline.train_hourly import (  # noqa: E402
    load_hourly, add_features, make_candidates,
    FEATURE_COLUMNS, HORIZONS, REPORT_HORIZONS,
    MeanEnsemble, BoostedResidual, ZeroDeltaBaseline,
    DiurnalNaive, DampedReversion,
)

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "reports" / "explainability"

# Muted palette - these figures go in a report, not a dashboard.
POS_COLOR = "#c2410c"
NEG_COLOR = "#1d4ed8"


def get_linear_step(model):
    """
    Digs the fitted linear estimator out of whatever wrapper it is in.
    Returns (estimator, scaler) or (None, None) if the model has no
    single linear component - e.g. the ensemble, which averages a
    linear and a tree model and has no one coefficient vector.
    """
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler")
        for key in ("ridge", "net"):
            if key in model.named_steps:
                return model.named_steps[key], scaler
    if isinstance(model, BoostedResidual) and hasattr(model, "linear_"):
        return get_linear_step(model.linear_)
    return None, None


def coefficient_table(horizons=REPORT_HORIZONS) -> pd.DataFrame:
    """
    Refits a Ridge at each horizon and returns standardised coefficients.

    Refitting rather than reading the saved models is deliberate: the
    saved winner differs by horizon (ensemble at 24h, ridge at 72h), so
    their coefficients are not comparable. Holding the model class fixed
    isolates the effect of the horizon itself.
    """
    hourly = add_features(load_hourly("2025-07-01"))
    rows = {}

    for hz in horizons:
        cols = FEATURE_COLUMNS + [f"delta_{hz}"]
        clean = hourly.dropna(subset=cols)
        model = make_candidates(len(clean))["ridge"]
        model.fit(clean[FEATURE_COLUMNS], clean[f"delta_{hz}"])

        est, _ = get_linear_step(model)
        if est is None:
            continue
        # Features are standardised inside the pipeline, so coefficients
        # are directly comparable across features without rescaling.
        rows[f"+{hz}h"] = pd.Series(est.coef_, index=FEATURE_COLUMNS)

    return pd.DataFrame(rows)


def plot_coefficients(coefs: pd.DataFrame, path: Path, top_n: int = 14) -> None:
    """Horizontal bars per horizon, features ordered by overall influence."""
    order = coefs.abs().mean(axis=1).sort_values(ascending=False).head(top_n).index
    sub = coefs.loc[order[::-1]]

    fig, axes = plt.subplots(1, len(sub.columns), figsize=(4.2 * len(sub.columns), 6.5),
                             sharey=True)
    if len(sub.columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, sub.columns):
        vals = sub[col]
        colors = [POS_COLOR if v > 0 else NEG_COLOR for v in vals]
        ax.barh(range(len(vals)), vals.values, color=colors, height=0.68)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=9)
        ax.axvline(0, color="#3f3f46", lw=0.8)
        ax.set_title(col, fontsize=11, weight="bold")
        ax.set_xlabel("standardised coefficient", fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Ridge coefficients on the AQI delta, by forecast horizon",
                 fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def shap_analysis(horizon: int, n_sample: int = 1500) -> None:
    """
    SHAP on hist_gbr - the model that LOSES. The question being asked is
    whether it lost because there is no nonlinear structure to find.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed - run: pip install shap", file=sys.stderr)
        return

    hourly = add_features(load_hourly("2025-07-01"))
    cols = FEATURE_COLUMNS + [f"delta_{horizon}"]
    clean = hourly.dropna(subset=cols)

    X = clean[FEATURE_COLUMNS]
    y = clean[f"delta_{horizon}"]

    model = make_candidates(len(clean))["hist_gbr"]
    model.fit(X, y)

    # Sample for speed; SHAP on 8k rows x 26 features is slow and the
    # extra precision does not change the conclusion.
    Xs = X.sample(min(n_sample, len(X)), random_state=42).sort_index()

    print(f"  computing SHAP values on {len(Xs)} rows...")
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xs, check_additivity=False)

    shap.summary_plot(sv, Xs, show=False, max_display=14, plot_size=(9, 6.5))
    plt.title(f"SHAP: gradient boosting, +{horizon}h delta", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_summary_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dependence plots for the strongest features. A near-straight line
    # here is the evidence that trees have nothing extra to offer.
    importance = pd.Series(np.abs(sv.values).mean(axis=0), index=FEATURE_COLUMNS)
    top = importance.sort_values(ascending=False).head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), top):
        i = FEATURE_COLUMNS.index(feat)
        ax.scatter(Xs[feat], sv.values[:, i], s=5, alpha=0.35, color=NEG_COLOR)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.grid(alpha=0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Straight-line fit through the SHAP cloud. R2 near 1 means the
        # model's learned response to this feature is essentially linear.
        m, b = np.polyfit(Xs[feat], sv.values[:, i], 1)
        xs = np.linspace(Xs[feat].min(), Xs[feat].max(), 50)
        ax.plot(xs, m * xs + b, color=POS_COLOR, lw=1.6)
        pred = m * Xs[feat] + b
        ss_res = ((sv.values[:, i] - pred) ** 2).sum()
        ss_tot = ((sv.values[:, i] - sv.values[:, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.set_title(f"{feat}  (linearity R2 = {r2:.2f})", fontsize=10)

    fig.suptitle(
        f"SHAP dependence, +{horizon}h - straight lines mean no nonlinearity to exploit",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"shap_dependence_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    importance.sort_values(ascending=False).to_csv(
        OUT_DIR / f"shap_importance_h{horizon}.csv", header=["mean_abs_shap"])
    print(f"  top features: {', '.join(top)}")


def main():
    ap = argparse.ArgumentParser(description="Explainability for AQI models")
    ap.add_argument("--horizon", type=int, default=24,
                    help="Horizon for the SHAP analysis")
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Coefficient analysis (deployed linear models)...")
    coefs = coefficient_table()
    if coefs.empty:
        print("  no linear models available", file=sys.stderr)
    else:
        coefs.to_csv(OUT_DIR / "ridge_coefficients.csv")
        plot_coefficients(coefs, OUT_DIR / "coefficients_by_horizon.png")

        print("\n  Strongest features per horizon:")
        for col in coefs.columns:
            top = coefs[col].abs().sort_values(ascending=False).head(4)
            named = ", ".join(f"{k} ({coefs.loc[k, col]:+.2f})" for k in top.index)
            print(f"    {col:6s} {named}")

    if not args.skip_shap:
        print(f"\nSHAP analysis (hist_gbr, +{args.horizon}h)...")
        shap_analysis(args.horizon)

    print(f"\nWritten to: {OUT_DIR}")


if __name__ == "__main__":
    main()
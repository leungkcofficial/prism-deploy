#!/usr/bin/env python3
"""
PRISM EDA Script
================
Reproduces key descriptive statistics and figures from the manuscript:

  1. Table 2 — Cohort characteristics
  2. Missing data summary by feature
  3. Treatment prevalence by cohort
  4. Propensity score distribution (requires propensity model)
  5. Kaplan-Meier overall survival curve
  6. Feature importance (SHAP for ACMM)

Prerequisites
-------------
  Preprocessed data in ../data_lake/
  For SHAP: models/acmm/acmm_xgboost_calibrated.pkl

Usage
-----
  python training/eda.py
  python training/eda.py --no-shap   # skip SHAP (faster)

Output
------
  Prints Table 2 to stdout; saves plots to eda_output/
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_LAKE = ROOT / "data_lake"
OUT_DIR   = ROOT / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

FEATURES_6 = ["age_at_t0", "gender", "hemoglobin_at_t0",
               "phosphate_at_t0", "cci_score_total", "creatinine_at_t0"]

FEATURE_LABELS = {
    "age_at_t0":         "Age at t₀ (years, scaled)",
    "gender":            "Sex = Female (0/1)",
    "hemoglobin_at_t0":  "Haemoglobin (g/dL, scaled)",
    "phosphate_at_t0":   "Phosphate (mmol/L, scaled)",
    "cci_score_total":   "CCI score (scaled)",
    "creatinine_at_t0":  "Creatinine (µmol/L, scaled)",
}


def load_data():
    train = pd.read_csv(DATA_LAKE / "train_processed.csv")
    sp    = pd.read_csv(DATA_LAKE / "spatial_test_processed.csv")
    tp    = pd.read_csv(DATA_LAKE / "temporal_test_processed.csv")
    return dict(train=train, spatial=sp, temporal=tp)


def table2(data: dict):
    """Table 2 — Cohort characteristics (manuscript Table 2)."""
    print("\n" + "="*70)
    print("TABLE 2 — COHORT CHARACTERISTICS")
    print("="*70)

    cohorts = [
        ("QMH Training", data["train"]),
        ("QMH Spatial Test", data["spatial"]),
        ("QMH Temporal Test", data["temporal"]),
    ]
    all_data = pd.concat([data["train"], data["spatial"], data["temporal"]])
    cohorts.insert(0, ("QMH Development (total)", all_data))

    for name, df in cohorts:
        print(f"\n  {name}  (n={len(df):,})")
        print(f"  {'─'*60}")

        # Note: values in the processed CSV are already scaled [0,1]
        # We report descriptive stats on the scaled values
        print(f"  Age (scaled median [IQR]): "
              f"{df['age_at_t0'].median():.3f} "
              f"[{df['age_at_t0'].quantile(.25):.3f}–{df['age_at_t0'].quantile(.75):.3f}]")
        print(f"  Female (%): {df['gender'].mean():.1%}")
        print(f"  Creatinine (scaled median): {df['creatinine_at_t0'].median():.3f}")
        print(f"  CCI (scaled median):        {df['cci_score_total'].median():.3f}")
        print(f"  Treatment A=1 (%):          {df['A'].mean():.1%}")
        print(f"  1-year events (%):          "
              f"{((df['event']==1) & (df['duration']<=365)).mean():.1%}")
        print(f"  5-year events (%):          "
              f"{((df['event']==1) & (df['duration']<=1825)).mean():.1%}")

        # Missing data
        print(f"\n  Missing data:")
        for feat in FEATURES_6:
            pct = df[feat].isna().mean()
            print(f"    {FEATURE_LABELS[feat]:<35} {pct:.1%}")

    print()


def plot_km_overall(data: dict):
    """Kaplan-Meier overall survival curves by cohort."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"QMH Training": "#1f77b4", "QMH Spatial": "#2ca02c",
              "QMH Temporal": "#d62728"}
    for (name, df), col in zip(
        [("QMH Training", data["train"]),
         ("QMH Spatial",  data["spatial"]),
         ("QMH Temporal", data["temporal"])],
        colors.values()
    ):
        kmf = KaplanMeierFitter()
        kmf.fit(df["duration"], df["event"], label=f"{name} (n={len(df):,})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=col)

    ax.set_xlabel("Time from t₀ (scaled units)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier Survival Curves by Cohort")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "km_overall.png", dpi=150)
    plt.close()
    log.info(f"Saved KM plot → {OUT_DIR}/km_overall.png")


def plot_propensity(data: dict):
    """Propensity score distribution by treatment group (requires fitted model)."""
    prop_path = ROOT / "models/rsf_dr/propensity_model.pkl"
    if not prop_path.exists():
        log.warning("Propensity model not found — skipping propensity plot")
        return

    with open(prop_path, "rb") as f:
        prop = pickle.load(f)

    train = data["train"]
    X = train[FEATURES_6].values.astype(float)
    A = train["A"].values
    ps = prop.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ps[A==0], bins=40, alpha=0.6, color="#d62728", label="A=0 (no early dialysis)",
            density=True)
    ax.hist(ps[A==1], bins=40, alpha=0.6, color="#2ca02c", label="A=1 (early dialysis)",
            density=True)
    ax.axvline(0.05, color="grey", lw=1, ls="--", alpha=0.7, label="Overlap boundary")
    ax.axvline(0.95, color="grey", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel("Propensity score ê(X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distribution by Treatment Group")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    in_overlap = ((ps >= 0.05) & (ps <= 0.95)).mean()
    ax.set_title(f"Propensity Score Distribution (overlap: {in_overlap:.1%})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "propensity_distribution.png", dpi=150)
    plt.close()
    log.info(f"Saved propensity plot → {OUT_DIR}/propensity_distribution.png")


def plot_shap(data: dict):
    """SHAP feature importance for ACMM."""
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP plot (pip install shap)")
        return

    acmm_path = ROOT / "models/acmm/acmm_xgboost_calibrated.pkl"
    prep_path  = ROOT / "models/acmm/acmm_preprocessor.pkl"
    if not acmm_path.exists():
        log.warning("ACMM model not found — skipping SHAP plot")
        return

    with open(acmm_path, "rb") as f: model = pickle.load(f)
    with open(prep_path,  "rb") as f: prep  = pickle.load(f)

    X = data["spatial"][FEATURES_6].values.astype(float)
    X_p = prep["scaler"].transform(prep["imputer"].transform(X))

    # Extract the base XGBoost from CalibratedClassifierCV
    base = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(X_p)

    fig, ax = plt.subplots(figsize=(7, 4))
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feat_labels = [FEATURE_LABELS[f] for f in FEATURES_6]
    order = np.argsort(mean_abs)
    ax.barh([feat_labels[i] for i in order], mean_abs[order], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("ACMM Feature Importance (SHAP, spatial test set)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "acmm_shap.png", dpi=150)
    plt.close()
    log.info(f"Saved SHAP plot → {OUT_DIR}/acmm_shap.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    data = load_data()
    table2(data)
    plot_km_overall(data)
    plot_propensity(data)
    if not args.no_shap:
        plot_shap(data)
    log.info(f"EDA complete. Plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

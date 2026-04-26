#!/usr/bin/env python3
"""
PRISM Training Script — ACMM (Model 4)
=======================================
Trains the All-Cause Mortality Model: a treatment-agnostic XGBoost binary
classifier predicting 1-year all-cause mortality, calibrated via isotonic
regression (§2.6.4 of the manuscript).

The ACMM is INDEPENDENT of the causal models:
  - Uses a separate BayesianRidge MICE imputer (not ExtraTrees)
  - Answers: "what is this patient's absolute 1-year mortality probability?"
  - Does NOT incorporate treatment information
  - Provides the ACMM threshold (30%) for Zone A/B vs C/D separation

Performance:
  AUC 0.793 (95% CI 0.773–0.816) spatial
  AUC 0.790 (95% CI 0.775–0.804) temporal
  AUC 0.743 (95% CI 0.727–0.758) external (TMH, n=6,039)
  O:E ratio 0.98 (QMH spatial)

Prerequisites
-------------
  Preprocessed data in ../data_lake/

Usage
-----
  python training/train_acmm.py

Output
------
  models/acmm/acmm_xgboost_calibrated.pkl
  models/acmm/acmm_preprocessor.pkl

References
----------
  [6] Chen T, Guestrin C. XGBoost. KDD 2016.
  [7] Zadrozny B, Elkan C. Transforming classifier scores into accurate
      multiclass probability estimates. KDD 2002.
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_LAKE = ROOT / "data_lake"
OUT_DIR   = ROOT / "models/acmm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
HORIZON_1Y  = 365.0   # 1-year mortality outcome

FEATURES_6 = ["age_at_t0", "gender", "hemoglobin_at_t0",
               "phosphate_at_t0", "cci_score_total", "creatinine_at_t0"]


def load_data():
    train = pd.read_csv(DATA_LAKE / "train_processed.csv")
    sp    = pd.read_csv(DATA_LAKE / "spatial_test_processed.csv")
    tp    = pd.read_csv(DATA_LAKE / "temporal_test_processed.csv")

    def _label(df):
        return ((df["event"] == 1) & (df["duration"] <= HORIZON_1Y)).astype(int)

    y_tr = _label(train); y_sp = _label(sp); y_tp = _label(tp)
    log.info(f"1-year event rate — train: {y_tr.mean():.1%}  "
             f"spatial: {y_sp.mean():.1%}  temporal: {y_tp.mean():.1%}")
    return (train[FEATURES_6].values, y_tr.values,
            sp[FEATURES_6].values,    y_sp.values,
            tp[FEATURES_6].values,    y_tp.values)


def fit_preprocessor(X_raw: np.ndarray):
    """
    ACMM-specific preprocessing (§2.4.3):
      - BayesianRidge MICE imputer (not ExtraTrees — smaller, faster for ACMM)
      - MinMaxScaler across all 6 features
    Fitted on training set only to prevent leakage.
    """
    log.info("Fitting ACMM preprocessor (BayesianRidge MICE + MinMaxScaler) ...")
    imputer = IterativeImputer(
        estimator=BayesianRidge(), max_iter=10, random_state=RANDOM_SEED)
    imputer.fit(X_raw)
    X_imp = imputer.transform(X_raw)

    scaler = MinMaxScaler()
    scaler.fit(X_imp)

    return dict(imputer=imputer, scaler=scaler)


def bootstrap_auc(y_true, y_prob, n_boot=1000, seed=RANDOM_SEED):
    """Bootstrap 95% CI for AUC-ROC."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(len(y_true), size=len(y_true))
        if y_true[idx].nunique() < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    np.random.seed(RANDOM_SEED)
    X_tr, y_tr, X_sp, y_sp, X_tp, y_tp = load_data()

    # Preprocess
    prep   = fit_preprocessor(X_tr)
    X_tr_p = prep["scaler"].transform(prep["imputer"].transform(X_tr))
    X_sp_p = prep["scaler"].transform(prep["imputer"].transform(X_sp))
    X_tp_p = prep["scaler"].transform(prep["imputer"].transform(X_tp))

    # XGBoost base classifier
    log.info("Training XGBoost ...")
    xgb_base = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = (y_tr == 0).sum() / (y_tr == 1).sum(),
        eval_metric       = "auc",
        random_state      = RANDOM_SEED,
        verbosity         = 0,
    )

    # Isotonic calibration (§2.6.4)
    log.info("Calibrating with isotonic regression ...")
    model = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    model.fit(X_tr_p, y_tr)

    # Evaluate
    for X, y, tag in [(X_sp_p, y_sp, "Spatial"), (X_tp_p, y_tp, "Temporal")]:
        prob = model.predict_proba(X)[:, 1]
        auc  = roc_auc_score(y, prob)
        ci   = bootstrap_auc(pd.Series(y), prob)
        log.info(f"AUC {tag}: {auc:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f})")

        # Observed-to-expected
        obs = y.mean()
        exp = prob.mean()
        log.info(f"O:E {tag}: {obs/exp:.3f}  (obs={obs:.3f}, exp={exp:.3f})")

    # Save
    with open(OUT_DIR / "acmm_xgboost_calibrated.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(OUT_DIR / "acmm_preprocessor.pkl", "wb") as f:
        pickle.dump(prep, f)
    log.info(f"Saved ACMM model + preprocessor to {OUT_DIR}")


if __name__ == "__main__":
    main()

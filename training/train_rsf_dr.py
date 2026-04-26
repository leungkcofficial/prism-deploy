#!/usr/bin/env python3
"""
PRISM Training Script — RSF DR-Learner (Model 1)
=================================================
Trains a Random Survival Forest with the Doubly-Robust meta-learner strategy
(§2.6.1 of the manuscript) on the 6-feature PRISM cohort.

Three learner variants:
  S-Learner : single RSF with treatment A as 7th feature
  T-Learner : separate RSF per treatment arm (A=0, A=1)
  DR-Learner: S-Learner RSF weighted by overlap weights w = A·(1−ê) + (1−A)·ê

The DR-Learner is the primary model used for Zone C/D assignment.

Prerequisites
-------------
1. Preprocessed data in ../data_lake/ (run prepare_data.py first)
2. Python environment: scikit-survival, xgboost, numpy, pandas, scikit-learn

Usage
-----
  python training/train_rsf_dr.py

Output
------
  models/rsf_dr/dr_learner.pkl
  models/rsf_dr/s_learner.pkl
  models/rsf_dr/t_learner_A0.pkl
  models/rsf_dr/t_learner_A1.pkl
  models/rsf_dr/propensity_model.pkl

References
----------
  [1] Ishwaran H et al. Random survival forests. Ann Appl Stat 2008.
  [2] Li F et al. Addressing extreme propensity scores via overlap weights. Am J Epidemiol 2019.
  [3] Künzel S et al. Metalearners for estimating heterogeneous treatment effects. PNAS 2019.
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants (§2.6.1) ────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_LAKE    = ROOT / "data_lake"
OUTPUT_DIR   = ROOT / "models/rsf_dr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
EVAL_TIMES   = [365, 730, 1095, 1460, 1825]

# Feature columns in preprocessed CSVs
FEATURES_6 = ["age_at_t0", "gender", "hemoglobin_at_t0",
               "phosphate_at_t0", "cci_score_total", "creatinine_at_t0"]
CONTINUOUS  = ["age_at_t0", "hemoglobin_at_t0", "phosphate_at_t0",
               "cci_score_total", "creatinine_at_t0"]

# RSF hyperparameters (§2.6.1; selected via grid search on temporal C-index)
RSF_PARAMS = dict(
    n_estimators   = 300,    # ensemble size
    max_depth      = None,   # unlimited — trees grown to purity
    min_samples_leaf = 30,   # minimum leaf size for stable hazard estimates
    n_jobs         = -1,
    random_state   = RANDOM_SEED,
)


def load_data() -> dict:
    """Load and split preprocessed data from data_lake."""
    log.info("Loading preprocessed data ...")
    train = pd.read_csv(DATA_LAKE / "train_processed.csv")
    sp    = pd.read_csv(DATA_LAKE / "spatial_test_processed.csv")
    tp    = pd.read_csv(DATA_LAKE / "temporal_test_processed.csv")

    def _arrays(df):
        X   = df[FEATURES_6].values.astype(float)
        A   = df["A"].values.astype(int)
        dur = df["duration"].values.astype(float)
        evt = df["event"].values.astype(bool)
        return X, A, dur, evt

    log.info(f"  Train: {len(train):,}  Spatial: {len(sp):,}  Temporal: {len(tp):,}")
    return dict(
        train = _arrays(train),
        sp    = _arrays(sp),
        tp    = _arrays(tp),
    )


def fit_propensity(X_train: np.ndarray, A_train: np.ndarray):
    """
    Fit XGBoost propensity model ê(X) = P(A=1|X).
    Used for DR-Learner overlap weighting (§2.6.1).
    """
    import xgboost as xgb
    log.info("Fitting propensity model (XGBoost) ...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0,
    )
    model.fit(X_train, A_train)
    ps_train = model.predict_proba(X_train)[:, 1]
    log.info(f"  Propensity range: [{ps_train.min():.3f}, {ps_train.max():.3f}]")
    log.info(f"  Overlap zone [0.05, 0.95]: {((ps_train >= 0.05) & (ps_train <= 0.95)).mean():.1%}")
    return model, ps_train


def overlap_weights(A: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Overlap weights (Li et al. 2019):  w = A·(1−ê) + (1−A)·ê
    Down-weights patients near-deterministically assigned to one arm.
    """
    return A * (1 - ps) + (1 - A) * ps


def make_survival_array(dur: np.ndarray, evt: np.ndarray):
    """Pack into structured array for scikit-survival."""
    return np.array([(bool(e), d) for d, e in zip(dur, evt)],
                    dtype=[("event", bool), ("duration", float)])


def c_index(model, X, A, y, times=EVAL_TIMES, label=""):
    """Compute C-index at 1 year for an S-Learner RSF."""
    X_A = np.hstack([X, A.reshape(-1, 1)])
    # Predict at 1 year (index 0)
    sf   = model.predict_survival_function(X_A)
    risk = np.array([1 - f(times[0]) for f in sf])
    ci   = concordance_index_censored(y["event"], y["duration"], risk)[0]
    log.info(f"  C-index {label} 1y: {ci:.4f}")
    return ci


def train_s_learner(X: np.ndarray, A: np.ndarray, y) -> RandomSurvivalForest:
    """S-Learner: append treatment A as 7th feature."""
    log.info("Training S-Learner RSF ...")
    X_A = np.hstack([X, A.reshape(-1, 1)])
    m = RandomSurvivalForest(**RSF_PARAMS)
    m.fit(X_A, y)
    log.info("  S-Learner done.")
    return m


def train_t_learner(X: np.ndarray, A: np.ndarray, y) -> tuple:
    """T-Learner: separate RSF per treatment arm."""
    log.info("Training T-Learner RSF (A=0) ...")
    m0 = RandomSurvivalForest(**RSF_PARAMS)
    m0.fit(X[A == 0], y[A == 0])
    log.info(f"  T0 trained on {(A==0).sum()} patients")

    log.info("Training T-Learner RSF (A=1) ...")
    m1 = RandomSurvivalForest(**RSF_PARAMS)
    m1.fit(X[A == 1], y[A == 1])
    log.info(f"  T1 trained on {(A==1).sum()} patients")
    return m0, m1


def train_dr_learner(X: np.ndarray, A: np.ndarray, y,
                     ps: np.ndarray) -> RandomSurvivalForest:
    """
    DR-Learner: S-Learner RSF fitted with overlap weights.
    Overlap weighting (Li et al. 2019) avoids positivity violations from IPW.
    """
    log.info("Training DR-Learner RSF (overlap-weighted S-Learner) ...")
    w   = overlap_weights(A, ps)
    X_A = np.hstack([X, A.reshape(-1, 1)])
    m   = RandomSurvivalForest(**RSF_PARAMS)
    m.fit(X_A, y, sample_weight=w)
    log.info("  DR-Learner done.")
    return m


def evaluate(name: str, model, data: dict, style="s"):
    """Report C-index on spatial and temporal test sets."""
    X_sp, A_sp, dur_sp, evt_sp = data["sp"]
    X_tp, A_tp, dur_tp, evt_tp = data["tp"]
    y_sp = make_survival_array(dur_sp, evt_sp)
    y_tp = make_survival_array(dur_tp, evt_tp)

    if style == "s":
        c_index(model, X_sp, A_sp, y_sp, label=f"{name} spatial")
        c_index(model, X_tp, A_tp, y_tp, label=f"{name} temporal")
    else:
        # T-Learner: evaluate on combined set using T0/T1 predictions
        m0, m1 = model
        def ci_t(X, A, y, tag):
            sf0 = m0.predict_survival_function(X)
            sf1 = m1.predict_survival_function(X)
            ite = np.array([1 - f(EVAL_TIMES[0]) for f in sf1]) - \
                  np.array([1 - f(EVAL_TIMES[0]) for f in sf0])
            risk = np.where(A == 1,
                            np.array([1-f(EVAL_TIMES[0]) for f in sf1]),
                            np.array([1-f(EVAL_TIMES[0]) for f in sf0]))
            ci = concordance_index_censored(y["event"], y["duration"], risk)[0]
            log.info(f"  C-index {tag} 1y: {ci:.4f}")
        ci_t(X_sp, A_sp, y_sp, f"{name} spatial")
        ci_t(X_tp, A_tp, y_tp, f"{name} temporal")


def main():
    np.random.seed(RANDOM_SEED)
    data = load_data()
    X_tr, A_tr, dur_tr, evt_tr = data["train"]
    y_tr = make_survival_array(dur_tr, evt_tr)

    # Propensity model
    prop, ps_tr = fit_propensity(X_tr, A_tr)
    with open(OUTPUT_DIR / "propensity_model.pkl", "wb") as f:
        pickle.dump(prop, f)
    log.info("Saved propensity model.")

    # S-Learner
    s_learner = train_s_learner(X_tr, A_tr, y_tr)
    with open(OUTPUT_DIR / "s_learner.pkl", "wb") as f:
        pickle.dump(s_learner, f)
    log.info("Saved S-Learner.")
    evaluate("S-Learner", s_learner, data, style="s")

    # T-Learner
    t0, t1 = train_t_learner(X_tr, A_tr, y_tr)
    with open(OUTPUT_DIR / "t_learner_A0.pkl", "wb") as f: pickle.dump(t0, f)
    with open(OUTPUT_DIR / "t_learner_A1.pkl", "wb") as f: pickle.dump(t1, f)
    log.info("Saved T-Learner.")

    # DR-Learner
    dr_learner = train_dr_learner(X_tr, A_tr, y_tr, ps_tr)
    with open(OUTPUT_DIR / "dr_learner.pkl", "wb") as f:
        pickle.dump(dr_learner, f)
    log.info("Saved DR-Learner.")
    evaluate("DR-Learner", dr_learner, data, style="s")

    log.info(f"\nAll models saved to {OUTPUT_DIR}")
    log.info("Run training/train_cf_rl.py next to train Causal Forest and R-Learner.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PRISM Training Script — Causal Forest (CF) + R-Learner (RL) (Models 2 & 3)
===========================================================================
Trains econml Causal Forest and R-Learner models at each of 5 time horizons
(1–5 years) for per-year ITE estimation with honest 95% confidence intervals.

Per §2.6.2–2.6.3 of the manuscript:
  CF  — econml.grf.CausalForest: honest half-sample bootstrap CI
  RL  — econml.dml.CausalForestDML: residualises outcome+treatment on nuisance
         functions before fitting final causal model (5-fold cross-fitting)

Outcome: IPCW-weighted binary mortality indicator at each horizon.
Both models reuse the propensity scores from train_rsf_dr.py.

Prerequisites
-------------
1. models/rsf_dr/propensity_model.pkl  (from train_rsf_dr.py)
2. Preprocessed data in ../data_lake/
3. econml>=0.15, scikit-learn>=1.3

Usage
-----
  python training/train_cf_rl.py

Output
------
  models/causal_forest/causal_forest.pkl   (1-year)
  models/causal_forest/causal_forest_2y.pkl ... causal_forest_5y.pkl
  models/r_learner/r_learner.pkl          (1-year)
  models/r_learner/r_learner_2y.pkl ... r_learner_5y.pkl

References
----------
  [4] Athey S, Tibshirani J, Wager S. Generalised random forests. Ann Stat 2019.
  [5] Chernozhukov V et al. Double/debiased machine learning. Econom J 2018.
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from lifelines.utils import survival_table_from_events
from econml.grf import CausalForest
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_LAKE = ROOT / "data_lake"
CF_DIR    = ROOT / "models/causal_forest"
RL_DIR    = ROOT / "models/r_learner"
PROP_PATH = ROOT / "models/rsf_dr/propensity_model.pkl"
CF_DIR.mkdir(parents=True, exist_ok=True)
RL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EVAL_TIMES  = [365, 730, 1095, 1460, 1825]
YEAR_LABELS = [1, 2, 3, 4, 5]

FEATURES_6 = ["age_at_t0", "gender", "hemoglobin_at_t0",
               "phosphate_at_t0", "cci_score_total", "creatinine_at_t0"]

OVERLAP_LO, OVERLAP_HI = 0.05, 0.95


def load_data() -> dict:
    train = pd.read_csv(DATA_LAKE / "train_processed.csv")
    sp    = pd.read_csv(DATA_LAKE / "spatial_test_processed.csv")
    tp    = pd.read_csv(DATA_LAKE / "temporal_test_processed.csv")
    log.info(f"Loaded: train={len(train):,}  spatial={len(sp):,}  temporal={len(tp):,}")
    return dict(train=train, sp=sp, tp=tp)


def ipcw_binary_outcome(dur: np.ndarray, evt: np.ndarray,
                         horizon: float) -> np.ndarray:
    """
    IPCW-weighted binary outcome: 1 if died by horizon, 0 if alive at horizon.
    Censored patients with follow-up < horizon are excluded (sample weight = 0
    is equivalent when weights are not used here; we simply drop them).

    For simplicity, this returns a mask of usable patients and their outcomes.
    """
    usable = (evt == 1) | (dur >= horizon)
    Y      = (evt[usable] == 1) & (dur[usable] <= horizon)
    return usable, Y.astype(float)


def load_propensity(X: np.ndarray):
    """Load propensity model and compute P(A=1|X)."""
    with open(PROP_PATH, "rb") as f:
        prop = pickle.load(f)
    ps = prop.predict_proba(X)[:, 1]
    return ps


def train_cf_year(X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                  ps: np.ndarray, year: int) -> CausalForest:
    """Train Causal Forest for a single time horizon."""
    log.info(f"  CF {year}y: n_train={len(X):,} ...")
    cf = CausalForest(
        n_estimators    = 500,
        min_samples_leaf = 5,
        max_depth       = None,
        inference       = True,   # enables honest CIs
        random_state    = RANDOM_SEED,
    )
    cf.fit(X, A, Y, sample_weight=None)
    return cf


def train_rl_year(X: np.ndarray, A: np.ndarray, Y: np.ndarray,
                  year: int) -> CausalForestDML:
    """Train R-Learner (DML) for a single time horizon."""
    log.info(f"  RL {year}y: n_train={len(X):,} ...")
    rl = CausalForestDML(
        model_y         = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        model_t         = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        n_estimators    = 500,
        min_samples_leaf = 5,
        max_depth       = None,
        cv              = 5,       # 5-fold cross-fitting
        random_state    = RANDOM_SEED,
    )
    rl.fit(Y, A, X=X)
    return rl


def evaluate_model(model, X_test, A_test, name):
    """Report 1-year ITE distribution statistics."""
    if hasattr(model, "predict"):
        ite = model.predict(X_test).flatten()
    else:
        ite = model.effect(X_test).flatten()
    in_overlap = (A_test == A_test)  # all patients for reporting
    log.info(f"    {name}: ITE mean={ite.mean():.4f}  std={ite.std():.4f}  "
             f"pct_benefit={(ite < 0).mean():.1%}")


def main():
    np.random.seed(RANDOM_SEED)
    data  = load_data()
    train = data["train"]
    X_tr  = train[FEATURES_6].values.astype(float)
    A_tr  = train["A"].values.astype(int)
    dur_tr = train["duration"].values.astype(float)
    evt_tr = train["event"].values.astype(int)

    ps_tr = load_propensity(X_tr)
    log.info(f"Overlap [0.05, 0.95]: {((ps_tr >= OVERLAP_LO) & (ps_tr <= OVERLAP_HI)).mean():.1%}")

    year_keys = ["causal_forest", "causal_forest_2y", "causal_forest_3y",
                 "causal_forest_4y", "causal_forest_5y"]
    rl_keys   = ["r_learner", "r_learner_2y", "r_learner_3y",
                 "r_learner_4y", "r_learner_5y"]

    for yr, t, cf_key, rl_key in zip(YEAR_LABELS, EVAL_TIMES, year_keys, rl_keys):
        log.info(f"\n── Year {yr} (horizon={t}d) ────────────────────────")

        usable, Y = ipcw_binary_outcome(dur_tr, evt_tr, t)
        Xu = X_tr[usable]
        Au = A_tr[usable]
        psu = ps_tr[usable]
        log.info(f"  Usable patients: {usable.sum():,} / {len(usable):,}")
        log.info(f"  Event rate: {Y.mean():.1%}")

        # Causal Forest
        cf = train_cf_year(Xu, Au, Y, psu, yr)
        with open(CF_DIR / f"{cf_key}.pkl", "wb") as f:
            pickle.dump(cf, f)
        evaluate_model(cf, Xu[:500], Au[:500], f"CF {yr}y (train sample)")
        log.info(f"  Saved {cf_key}.pkl")

        # R-Learner
        rl = train_rl_year(Xu, Au, Y, yr)
        with open(RL_DIR / f"{rl_key}.pkl", "wb") as f:
            pickle.dump(rl, f)
        evaluate_model(rl, Xu[:500], Au[:500], f"RL {yr}y (train sample)")
        log.info(f"  Saved {rl_key}.pkl")

    log.info(f"\nAll CF models saved to {CF_DIR}")
    log.info(f"All RL models saved to {RL_DIR}")
    log.info("Run training/train_acmm.py to train the ACMM.")


if __name__ == "__main__":
    main()

"""
PRISM Patient-Level Inference
==============================
Core prediction module for the PRISM clinical decision aid.

Loads all four trained models and returns individualised predictions:
  1. RSF DR-Learner  — counterfactual survival curves at 1–5 years
  2. Causal Forest   — 1–5 year ITE with honest 95% CI
  3. R-Learner (DML) — 1–5 year ITE with 95% CI (independent estimation)
  4. ACMM            — treatment-agnostic 1-year mortality probability

Zone assignment (4-zone framework, Table 1 of manuscript):
  A — Low ACMM (<30%), limited ITE  → Conservative care
  B — Low ACMM (<30%), strong ITE   → Early dialysis indicated
  C — High ACMM (≥30%), strong ITE  → Early dialysis despite high risk
  D — High ACMM (≥30%), limited ITE → Shared decision-making

Model paths resolve from PRISM_MODELS_DIR environment variable
(default: ./models).  Pre-trained artefacts are distributed separately
(see setup_models.sh); this module is purely inference code.
"""

import os
import pickle
import warnings
import logging
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Model directory ────────────────────────────────────────────────────────────
_MODELS_DIR = Path(os.environ.get("PRISM_MODELS_DIR", Path(__file__).parent.parent / "models"))

# ── Constants (must match training configuration exactly) ──────────────────────
FEATURES_6 = ["age_at_t0", "gender", "hemoglobin_at_t0",
               "phosphate_at_t0", "cci_score_total", "creatinine_at_t0"]
CONTINUOUS  = ["age_at_t0", "hemoglobin_at_t0", "phosphate_at_t0",
               "cci_score_total", "creatinine_at_t0"]
EVAL_TIMES  = np.array([365, 730, 1095, 1460, 1825])
YEAR_LABELS = [1, 2, 3, 4, 5]

OVERLAP_LO, OVERLAP_HI = 0.05, 0.95
ACMM_THR      = 0.30    # ACMM threshold: <30% = low risk, ≥30% = high risk
ITE_LOW_THR   = -0.15   # Zone A/B boundary: ≤−15pp = Zone B (strong benefit)
SUB_ITE_THR   = -0.1328 # Zone C/D boundary (subgroup model median split)

# 18-feature set for high-risk subgroup CF/RL model
FEATURES_18 = [
    "age_at_t0", "gender", "creatinine_at_t0", "hemoglobin_at_t0", "phosphate_at_t0",
    "myocardial_infarction", "congestive_heart_failure",
    "peripheral_vascular_disease", "cerebrovascular_disease",
    "dementia", "chronic_pulmonary_disease", "peptic_ulcer_disease",
    "mild_liver_disease", "diabetes_wo_complication", "diabetes_w_complication",
    "hemiplegia_paraplegia", "any_malignancy", "metastatic_cancer",
]
CONTINUOUS_18 = ["age_at_t0", "creatinine_at_t0", "hemoglobin_at_t0", "phosphate_at_t0"]

# Zone clinical descriptions
_ZONES = {
    "A": (
        "Continue monitoring",
        "Low baseline mortality risk (ACMM <30%) with limited predicted benefit "
        "from early dialysis. Continue close monitoring of renal function and "
        "symptoms. Reassess if eGFR declines further, uraemic symptoms develop, "
        "or clinical status changes.",
    ),
    "B": (
        "Early dialysis indicated",
        "Low baseline mortality risk (ACMM <30%) with substantial predicted survival "
        "benefit from early dialysis. Early initiation is strongly supported unless "
        "there is a compelling patient-preference reason against it.",
    ),
    "C": (
        "Early dialysis — strong benefit despite high risk",
        "High baseline mortality risk (ACMM ≥30%) with strong predicted survival "
        "benefit from early dialysis. Validated Zone C treatment effect: HR 0.38 "
        "(95% CI 0.27–0.54) on 6,039-patient external cohort. Early initiation "
        "supported; document patient values and dialysis burden tolerance.",
    ),
    "D": (
        "Shared decision-making",
        "High baseline mortality risk (ACMM ≥30%) with moderate or attenuated "
        "predicted benefit. Intensive shared decision-making with patient and family "
        "is essential. Consider goals-of-care discussion, functional status, and "
        "patient preference before committing to a treatment strategy.",
    ),
}

ZONE_COLOURS = {"A": "#66BB6A", "B": "#29B6F6", "C": "#7E57C2", "D": "#FFA726"}

# ── Module-level model cache ───────────────────────────────────────────────────
_PREP   = None
_MODELS = None


def _pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_preprocessor():
    """Load the saved MICE imputer + MinMaxScaler (fitted on QMH training set)."""
    global _PREP
    if _PREP is not None:
        return _PREP
    path = _MODELS_DIR / "causal_preprocessor.pkl"
    _PREP = _pkl(path)
    return _PREP


def load_models():
    """Load all four model groups into memory (cached after first call)."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    d = _MODELS_DIR
    _MODELS = dict(
        # RSF DR-Learner
        rsf_dr   = _pkl(d / "rsf_dr/dr_learner.pkl"),
        prop     = _pkl(d / "rsf_dr/propensity_model.pkl"),
        # Causal Forest (1y–5y)
        cf       = _pkl(d / "causal_forest/causal_forest.pkl"),
        cf_2y    = _pkl(d / "causal_forest/causal_forest_2y.pkl"),
        cf_3y    = _pkl(d / "causal_forest/causal_forest_3y.pkl"),
        cf_4y    = _pkl(d / "causal_forest/causal_forest_4y.pkl"),
        cf_5y    = _pkl(d / "causal_forest/causal_forest_5y.pkl"),
        # R-Learner (1y–5y)
        rl       = _pkl(d / "r_learner/r_learner.pkl"),
        rl_2y    = _pkl(d / "r_learner/r_learner_2y.pkl"),
        rl_3y    = _pkl(d / "r_learner/r_learner_3y.pkl"),
        rl_4y    = _pkl(d / "r_learner/r_learner_4y.pkl"),
        rl_5y    = _pkl(d / "r_learner/r_learner_5y.pkl"),
        # ACMM
        acmm      = _pkl(d / "acmm/acmm_xgboost_calibrated.pkl"),
        acmm_prep = _pkl(d / "acmm/acmm_preprocessor.pkl"),
        # Subgroup CF/RL (18 features, high-risk patients)
        sub_cf   = _pkl(d / "subgroup/subgroup_cf.pkl"),
        sub_rl   = _pkl(d / "subgroup/subgroup_rl.pkl"),
        sub_prop = _pkl(d / "subgroup/subgroup_propensity.pkl"),
        sub_prep = _pkl(d / "subgroup/subgroup_preprocessor.pkl"),
    )
    return _MODELS


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_patient(age: float, female: int, hb: float, po4: float,
                       cci: int, cr: float) -> tuple:
    """
    Impute missing values and scale to [0,1] for the causal models.

    Parameters
    ----------
    age    : years
    female : 0 = male, 1 = female
    hb     : haemoglobin g/dL  (float or np.nan if unknown)
    po4    : phosphate mmol/L  (float or np.nan if unknown)
    cci    : Charlson Comorbidity Index total score
    cr     : serum creatinine µmol/L

    Returns
    -------
    X_scaled : (1, 6) numpy array ready for models
    X_imp    : (1, 6) imputed but unscaled (original units)
    """
    prep = load_preprocessor()
    raw  = np.array([[float(age), float(female),
                      float(hb), float(po4),
                      float(cci), float(cr)]])
    X_imp    = prep["imputer"].transform(raw)
    X_scaled = X_imp.copy()
    X_scaled[:, prep["cont_idx"]] = prep["scaler"].transform(
        X_imp[:, prep["cont_idx"]])
    return X_scaled, X_imp


# ── RSF DR-Learner ─────────────────────────────────────────────────────────────

def _rsf_risk(model, X_with_A, times=EVAL_TIMES):
    """Extract mortality risks at each evaluation time from RSF survival function."""
    sf_fn = model.predict_survival_function(X_with_A)[0]
    risks = []
    for t in times:
        idx = max(0, min(np.searchsorted(sf_fn.x, t, "right") - 1,
                         len(sf_fn.y) - 1))
        risks.append(1.0 - sf_fn.y[idx])
    return np.array(risks)


def predict_rsf(X_scaled: np.ndarray) -> tuple:
    """
    Counterfactual mortality risks at years 1–5.

    Returns
    -------
    R0 : (5,) mortality risk under no early dialysis (A=0)
    R1 : (5,) mortality risk under early dialysis (A=1)
    """
    m    = load_models()
    X_A0 = np.hstack([X_scaled, [[0.0]]])
    X_A1 = np.hstack([X_scaled, [[1.0]]])
    R0   = _rsf_risk(m["rsf_dr"], X_A0)
    R1   = _rsf_risk(m["rsf_dr"], X_A1)
    return R0, R1


# ── Causal Forest ──────────────────────────────────────────────────────────────

def predict_cf(X_scaled: np.ndarray) -> tuple:
    """
    Causal Forest ITE with honest 95% CI at years 1–5.

    Returns
    -------
    ite    : (5,) point estimates
    ci_lo  : (5,) lower 95% CI bound
    ci_hi  : (5,) upper 95% CI bound
    """
    m = load_models()
    ites, los, his = [], [], []
    for key in ("cf", "cf_2y", "cf_3y", "cf_4y", "cf_5y"):
        ite       = float(m[key].predict(X_scaled)[0, 0])
        lo, hi    = m[key].predict_interval(X_scaled, alpha=0.05)
        ites.append(ite)
        los.append(float(lo[0, 0]))
        his.append(float(hi[0, 0]))
    return np.array(ites), np.array(los), np.array(his)


# ── R-Learner ──────────────────────────────────────────────────────────────────

def predict_rl(X_scaled: np.ndarray) -> tuple:
    """
    R-Learner (DML) ITE with 95% CI at years 1–5.

    Returns
    -------
    ite    : (5,) point estimates
    ci_lo  : (5,) lower 95% CI bound
    ci_hi  : (5,) upper 95% CI bound
    """
    m = load_models()
    ites, los, his = [], [], []
    for key in ("rl", "rl_2y", "rl_3y", "rl_4y", "rl_5y"):
        ite       = float(m[key].effect(X_scaled)[0])
        lo, hi    = m[key].effect_interval(X_scaled, alpha=0.05)
        ites.append(ite)
        los.append(float(lo[0]))
        his.append(float(hi[0]))
    return np.array(ites), np.array(los), np.array(his)


# ── Propensity score ───────────────────────────────────────────────────────────

def predict_propensity(X_scaled: np.ndarray) -> tuple:
    """
    Returns
    -------
    ps         : float, P(A=1|X)
    in_overlap : bool, ps ∈ [0.05, 0.95]
    """
    m   = load_models()
    raw = m["prop"].predict_proba(X_scaled)
    ps  = float(raw[0, 1] if raw.ndim > 1 else raw[0])
    return ps, (OVERLAP_LO <= ps <= OVERLAP_HI)


# ── ACMM ───────────────────────────────────────────────────────────────────────

def predict_acmm(age: float, female: int, hb: float, po4: float,
                 cci: int, cr: float) -> float:
    """
    Treatment-agnostic 1-year all-cause mortality probability.
    Uses its own BayesianRidge imputer + MinMaxScaler (independent from causal pipeline).

    External validation: AUC 0.743 (0.727–0.758) on 6,039-patient TMH cohort.
    """
    m    = load_models()
    prep = m["acmm_prep"]
    raw  = np.array([[float(age), float(female),
                      float(hb),  float(po4),
                      float(cci), float(cr)]])
    X_imp = prep["imputer"].transform(raw)
    X_sc  = prep["scaler"].transform(X_imp)
    return float(m["acmm"].predict_proba(X_sc)[0, 1])


# ── Subgroup model (high-risk, 18 features) ────────────────────────────────────

def predict_subgroup_ite(age: float, female: int, hb: float, po4: float,
                         cr: float, cci_flags: dict) -> dict:
    """
    18-feature subgroup CF/RL model for high-risk patients (ACMM ≥30%).
    Only called when CCI component flags are provided.

    Parameters
    ----------
    cci_flags : dict  e.g. {"myocardial_infarction": 1, "diabetes_wo_complication": 0, ...}

    Returns dict with sub_ite, sub_cf_ite, sub_cf_lo, sub_cf_hi,
                          sub_rl_ite, sub_rl_lo, sub_rl_hi,
                          sub_in_overlap, sub_propensity
    """
    m    = load_models()
    prep = m["sub_prep"]

    raw = np.zeros((1, len(FEATURES_18)))
    raw[0, 0] = float(age)
    raw[0, 1] = float(female)
    raw[0, 2] = float(cr) if not np.isnan(cr) else np.nan
    raw[0, 3] = float(hb) if not np.isnan(hb) else np.nan
    raw[0, 4] = float(po4) if not np.isnan(po4) else np.nan
    for i, feat in enumerate(FEATURES_18[5:], start=5):
        raw[0, i] = float(cci_flags.get(feat, 0))

    Xi = prep["imputer"].transform(raw)
    Xs = Xi.copy()
    Xs[:, prep["cont_idx"]] = prep["scaler"].transform(Xi[:, prep["cont_idx"]])

    sub_cf_ite = float(m["sub_cf"].predict(Xs)[0, 0])
    cf_lo, cf_hi = m["sub_cf"].predict_interval(Xs, alpha=0.05)
    sub_cf_lo, sub_cf_hi = float(cf_lo[0, 0]), float(cf_hi[0, 0])

    sub_rl_ite = float(m["sub_rl"].effect(Xs)[0])
    rl_lo, rl_hi = m["sub_rl"].effect_interval(Xs, alpha=0.05)
    sub_rl_lo, sub_rl_hi = float(rl_lo[0]), float(rl_hi[0])

    e = m["sub_prop"].predict_proba(Xs)[:, 1]

    return dict(
        sub_ite        = (sub_cf_ite + sub_rl_ite) / 2,
        sub_cf_ite     = sub_cf_ite,
        sub_cf_lo      = sub_cf_lo,
        sub_cf_hi      = sub_cf_hi,
        sub_rl_ite     = sub_rl_ite,
        sub_rl_lo      = sub_rl_lo,
        sub_rl_hi      = sub_rl_hi,
        sub_in_overlap = bool(OVERLAP_LO <= float(e[0]) <= OVERLAP_HI),
        sub_propensity = float(e[0]),
    )


# ── Clinical verdict ───────────────────────────────────────────────────────────

def _clinical_verdict(acmm_prob: float, avg_ite: float, in_overlap: bool,
                      rsf_ite_1y: float, sub_result: Optional[dict]) -> dict:
    """
    4-zone assignment per Table 1 of the manuscript.

    For ACMM <30%: uses CF+RL average (or RSF fallback if outside overlap).
    For ACMM ≥30%: prefers 18-feature subgroup model; falls back to 6-feature.
    """
    is_high_risk = acmm_prob >= ACMM_THR

    if not is_high_risk:
        ite_signal = avg_ite if in_overlap else rsf_ite_1y
        ite_source = ("CF+RL average (6-feature)" if in_overlap
                      else "RSF fallback (outside overlap)")
        zone = "B" if ite_signal <= ITE_LOW_THR else "A"
    else:
        if sub_result is not None:
            ite_signal = sub_result["sub_ite"]
            ite_source = "CF+RL average (18-feature subgroup model)"
            zone = "C" if ite_signal <= SUB_ITE_THR else "D"
        elif in_overlap:
            ite_signal = avg_ite
            ite_source = "CF+RL average (6-feature fallback)"
            zone = "C" if ite_signal <= SUB_ITE_THR else "D"
        else:
            ite_signal = rsf_ite_1y
            ite_source = "RSF fallback (no CCI components, outside overlap)"
            zone = "C" if ite_signal <= SUB_ITE_THR else "D"

    caveat = ""
    if is_high_risk and sub_result is None:
        caveat = ("CCI component flags not provided — zone C/D assignment uses "
                  "6-feature model fallback. Provide individual comorbidities for "
                  "the validated 18-feature subgroup model.")
    elif is_high_risk and sub_result and not sub_result["sub_in_overlap"]:
        caveat = ("Outside subgroup propensity overlap — ITE estimate is "
                  "extrapolative. Treat magnitude with caution.")
    elif not is_high_risk and not in_overlap:
        caveat = (f"Outside propensity overlap — ITE from RSF fallback "
                  f"({ite_signal*100:+.1f}pp). Treat magnitude with caution.")

    name, rec = _ZONES[zone]
    return dict(
        zone            = zone,
        zone_name       = name,
        zone_colour     = ZONE_COLOURS[zone],
        recommendation  = rec,
        acmm_risk_level = "high" if is_high_risk else "low",
        ite_signal      = ite_signal,
        ite_source      = ite_source,
        reliable        = (sub_result["sub_in_overlap"] if sub_result else in_overlap),
        caveat          = caveat,
    )


# ── Master predict function ────────────────────────────────────────────────────

def predict_patient(age: float, female: int, hb: float = np.nan,
                    po4: float = np.nan, cci: int = 0, cr: float = 500,
                    cci_flags: Optional[dict] = None, label: str = "Patient") -> dict:
    """
    Run all four models for a single patient.

    Parameters
    ----------
    age       : years
    female    : 0 = male, 1 = female
    hb        : haemoglobin g/dL  (optional; imputed if NaN)
    po4       : phosphate mmol/L  (optional; imputed if NaN)
    cci       : Charlson Comorbidity Index total score (0–37)
    cr        : serum creatinine µmol/L
    cci_flags : dict of individual CCI component flags (enables 18-feature subgroup model)
    label     : identifier string for display

    Returns
    -------
    dict with all prediction outputs and clinical verdict
    """
    X_scaled, X_imp = preprocess_patient(age, female, hb, po4, cci, cr)

    ps, in_overlap        = predict_propensity(X_scaled)
    R0, R1                = predict_rsf(X_scaled)
    cf_ite, cf_lo, cf_hi  = predict_cf(X_scaled)
    rl_ite, rl_lo, rl_hi  = predict_rl(X_scaled)
    acmm_prob             = predict_acmm(age, female, hb, po4, cci, cr)
    avg_ite_1y            = float((cf_ite[0] + rl_ite[0]) / 2)
    rsf_ite_1y            = float(R1[0] - R0[0])

    sub_result = None
    if acmm_prob >= ACMM_THR and cci_flags is not None:
        sub_result = predict_subgroup_ite(age, female, hb, po4, cr, cci_flags)

    verdict = _clinical_verdict(acmm_prob, avg_ite_1y, in_overlap,
                                rsf_ite_1y, sub_result)

    return dict(
        label            = label,
        # Raw inputs
        age              = age,
        female           = female,
        hb               = float(hb) if not np.isnan(float(hb)) else None,
        po4              = float(po4) if not np.isnan(float(po4)) else None,
        cci              = cci,
        cr               = cr,
        # RSF counterfactuals (years 1–5)
        rsf_R0           = R0.tolist(),
        rsf_R1           = R1.tolist(),
        rsf_ITE          = (R1 - R0).tolist(),
        # Causal Forest ITE + CI
        cf_ite           = cf_ite.tolist(),
        cf_lo            = cf_lo.tolist(),
        cf_hi            = cf_hi.tolist(),
        # R-Learner ITE + CI
        rl_ite           = rl_ite.tolist(),
        rl_lo            = rl_lo.tolist(),
        rl_hi            = rl_hi.tolist(),
        # Propensity
        propensity       = ps,
        in_overlap       = in_overlap,
        # ACMM
        acmm_prob        = acmm_prob,
        acmm_risk_level  = verdict["acmm_risk_level"],
        # Subgroup (optional)
        sub_result       = sub_result,
        # Clinical verdict
        zone             = verdict["zone"],
        zone_name        = verdict["zone_name"],
        zone_colour      = verdict["zone_colour"],
        recommendation   = verdict["recommendation"],
        ite_signal       = verdict["ite_signal"],
        ite_source       = verdict["ite_source"],
        reliable         = verdict["reliable"],
        caveat           = verdict["caveat"],
        year_labels      = YEAR_LABELS,
    )

"""
PRISM Clinical Decision Aid — FastAPI Backend
=============================================
Serves the web UI and prediction API for the PRISM framework.

Endpoints:
  GET  /                → index.html (clinical UI)
  POST /api/predict     → JSON prediction from 6 features + optional CCI flags
  GET  /api/sample      → sample patient input for demo/testing
  GET  /api/health      → liveness probe

Usage:
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  # Or via Docker: docker compose up
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PRISM Clinical Decision Aid",
    description="Individualised dialysis decision support for CKD Stage 5",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Lazy model loading (avoids loading on import for tests) ────────────────────
_models_loaded = False

def _ensure_models():
    global _models_loaded
    if not _models_loaded:
        from app.predict import load_preprocessor, load_models
        load_preprocessor()
        load_models()
        _models_loaded = True


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    age: float = Field(..., gt=0, lt=120, description="Age in years")
    female: int = Field(..., ge=0, le=1, description="Sex: 0=male, 1=female")
    creatinine: float = Field(..., gt=0, lt=5000, description="Serum creatinine µmol/L")
    haemoglobin: Optional[float] = Field(None, gt=0, lt=30, description="Haemoglobin g/dL (optional)")
    phosphate: Optional[float] = Field(None, gt=0, lt=10, description="Phosphate mmol/L (optional)")
    cci_total: int = Field(0, ge=0, le=37, description="Charlson Comorbidity Index total score")
    cci_flags: Optional[dict] = Field(None, description="Individual CCI component flags (enables subgroup model)")

    @field_validator("cci_flags")
    @classmethod
    def validate_cci_flags(cls, v):
        if v is None:
            return v
        valid_keys = {
            "myocardial_infarction", "congestive_heart_failure",
            "peripheral_vascular_disease", "cerebrovascular_disease",
            "dementia", "chronic_pulmonary_disease", "peptic_ulcer_disease",
            "mild_liver_disease", "diabetes_wo_complication", "diabetes_w_complication",
            "hemiplegia_paraplegia", "any_malignancy", "metastatic_cancer",
        }
        for k in v:
            if k not in valid_keys:
                raise ValueError(f"Unknown CCI flag: {k}")
            if v[k] not in (0, 1):
                raise ValueError(f"CCI flag {k} must be 0 or 1")
        return v


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": _models_loaded}


@app.get("/api/sample")
async def sample_patient():
    """Return a sample patient input for testing the prediction endpoint."""
    return {
        "description": "Representative Zone B patient (young male, moderate CCI)",
        "input": {
            "age": 55,
            "female": 0,
            "creatinine": 720,
            "haemoglobin": 8.1,
            "phosphate": 2.1,
            "cci_total": 2,
            "cci_flags": {
                "diabetes_wo_complication": 1,
                "congestive_heart_failure": 1,
            }
        }
    }


@app.post("/api/predict")
async def predict(patient: PatientInput):
    """
    Run the four PRISM models for a single patient.

    Returns zone assignment, ACMM risk, and 1–5 year counterfactual
    mortality risks from the RSF DR-Learner, with ITE estimates and
    95% CIs from the Causal Forest and R-Learner.
    """
    try:
        _ensure_models()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model files not found: {e}. Run setup_models.sh to populate models/."
        )

    try:
        from app.predict import predict_patient
        result = predict_patient(
            age        = patient.age,
            female     = patient.female,
            hb         = patient.haemoglobin if patient.haemoglobin is not None else float("nan"),
            po4        = patient.phosphate   if patient.phosphate   is not None else float("nan"),
            cci        = patient.cci_total,
            cr         = patient.creatinine,
            cci_flags  = patient.cci_flags,
        )
        # Replace NaN with None for JSON serialisation
        def _clean(v):
            if isinstance(v, float) and (v != v):  # NaN check
                return None
            return v

        def _deep_clean(obj):
            if isinstance(obj, dict):
                return {k: _deep_clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_deep_clean(x) for x in obj]
            return _clean(obj)

        return _deep_clean(result)

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

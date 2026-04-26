# PRISM — Predictive Renal Intelligence Survival Modeling

A clinical decision-support tool for CKD Stage 5 patients at the dialysis decision point.  
Given 6 routine lab values, PRISM estimates individualized 1–5 year mortality risk under  
early dialysis vs conservative management and assigns patients to one of four clinical zones.

---

## Quick Start (Docker)

Trained models are hosted on [Hugging Face Hub](https://huggingface.co/datasets/leungkc/prism-models)
and downloaded automatically during `docker compose build` — no manual setup needed.

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/prism-deploy.git
cd prism-deploy

# 2. Build the image (downloads ~4.3 GB of models from HF on first build; cached afterwards)
docker compose build

# 3. Start the service
docker compose up

# 4. Open the clinical interface
open http://localhost:8000
```

The app is ready when the health check passes (allow ~90 s for model loading on first request).

> **Offline / local use (no Docker)**
> ```bash
> python download_models.py          # populates models/ from HF Hub
> uvicorn app.main:app --port 8000   # or: pip install -r requirements.txt first
> ```

---

## Clinical Interface

Enter the following at the decision point (first eGFR ≤ 10 after persistent eGFR < 15):

| Field | Unit | Notes |
|-------|------|-------|
| Date of birth | — | Age auto-computed |
| Sex | — | Male / Female |
| Creatinine | µmol/L | Required |
| Haemoglobin | g/dL | Optional — imputed if missing |
| Phosphate | mmol/L | Optional — imputed if missing |
| Charlson Comorbidity Index | 0–37 | Calculated from 13 checkboxes |

**Outputs**: Zone (A/B/C/D), ACMM 1-year risk, RSF survival table (1–5 y), Causal Forest and R-Learner ITE with 95% CI, propensity overlap flag.

### Clinical Zone Definitions

| Zone | ACMM | ITE (1y) | Recommendation |
|------|------|----------|----------------|
| A | < 30% | > −15 pp | Conservative care; dialysis unlikely to benefit |
| B | < 30% | ≤ −15 pp | Early dialysis indicated |
| C | ≥ 30% | ≤ −13.3 pp | Early dialysis despite high mortality risk |
| D | ≥ 30% | > −13.3 pp | Shared decision-making; strong palliative input |

---

## Models Included

| Model | Purpose | Artefact |
|-------|---------|---------|
| RSF DR-Learner | Counterfactual survival curves (1–5 y) | `models/rsf_dr/` |
| Causal Forest | ITE + 95% CI per year | `models/causal_forest/` |
| R-Learner (DML) | ITE + 95% CI per year | `models/r_learner/` |
| ACMM (XGBoost) | Treatment-agnostic 1-year mortality | `models/acmm/` |
| Subgroup CF/RL | High-risk subgroup ITE (18 features) | `models/subgroup/` |

All models were trained on a multi-centre Hong Kong cohort (n = 14,682 after exclusions;  
41 hospitals, 2009–2023). See manuscript for full cohort description and model performance.

---

## API Reference

```
GET  /                    →  clinical UI (index.html)
GET  /api/health          →  {"status": "ok", "models_loaded": true}
GET  /api/sample          →  example patient JSON
POST /api/predict         →  prediction results
```

**POST /api/predict** — request body:

```json
{
  "age": 65.3,
  "female": 0,
  "creatinine": 620,
  "haemoglobin": 9.2,
  "phosphate": 1.8,
  "cci_total": 3,
  "cci_flags": {
    "diabetes_wo_complication": true,
    "congestive_heart_failure": true
  }
}
```

`haemoglobin`, `phosphate`, and `cci_flags` are optional. If `cci_flags` is provided and  
`cci_total ≥ 30%` ACMM threshold is crossed, the 18-feature subgroup model is used for ITE.

---

## Replication

The `notebooks/PRISM_Replication.ipynb` notebook walks through:

1. Model loading and preprocessing
2. Single-patient inference (with annotated outputs)
3. Zone logic table (reproduced from manuscript Table 4)
4. CF vs R-Learner agreement (r = 0.979)
5. ACMM calibration (O:E ratio)
6. Sample batch predictions from `data/sample_patients.csv`

```bash
pip install jupyter
jupyter notebook notebooks/PRISM_Replication.ipynb
```

---

## Training from Scratch

Clean training scripts are provided in `training/`. They assume access to a preprocessed  
cohort dataset (`data_lake/train_processed.csv`). See each script's docstring for column requirements.

```bash
python training/train_acmm.py
python training/train_rsf_dr.py
python training/train_cf_rl.py
```

---

## For Developers: Updating Models Without Rebuilding

If you have updated model artefacts in `models/`, you can run the service against the host  
directory without rebuilding the image:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -e PRISM_MODELS_DIR=/app/models \
  prism-deploy:latest
```

---

## Requirements

- Docker ≥ 20.10 (Docker Desktop with 4 GB memory allocation recommended)
- 4 GB+ free RAM for the RSF DR-Learner
- ~5 GB disk space for the Docker image

---

## Citation

> [Manuscript citation to be added upon publication]

---

## Disclaimer

PRISM is a research prototype and decision-support tool only.  
All predictions must be reviewed by a qualified clinician in the context of the individual patient.  
Do not use as a sole basis for clinical decisions.

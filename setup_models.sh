#!/usr/bin/env bash
# setup_models.sh — Copy trained model artefacts from the PRISM dev project
# ==========================================================================
# Run this once from the prism_deploy/ directory before starting the app.
# Models are not included in git (too large); this script copies them locally.
#
# Usage:
#   ./setup_models.sh                            # uses default PRISM_DEV path
#   ./setup_models.sh /path/to/prism/results     # explicit results directory
#
# After running, models/ will contain:
#   rsf_dr/         DR-Learner + propensity (~361 MB)
#   causal_forest/  5 × CF models (~55 MB)
#   r_learner/      5 × RL models (~70 MB)
#   acmm/           ACMM + preprocessor (<1 MB)
#   subgroup/       Subgroup CF/RL models (~152 MB)
#   causal_preprocessor.pkl  MICE imputer + MinMaxScaler (~91 MB)
#
# Total: ~730 MB

set -euo pipefail

PRISM_RESULTS="${1:-/mnt/dump/yard/projects/prism/results}"

if [ ! -d "$PRISM_RESULTS" ]; then
  echo "ERROR: PRISM results directory not found: $PRISM_RESULTS"
  echo "Usage: ./setup_models.sh [/path/to/prism/results]"
  exit 1
fi

echo "Copying model files from: $PRISM_RESULTS"

mkdir -p models/{rsf_dr,causal_forest,r_learner,acmm,subgroup}

# RSF DR-Learner (§2.6.1)
echo "  → RSF DR-Learner models ..."
cp "$PRISM_RESULTS/treatment_window_comparison/window_90_models/dr_learner.pkl" \
   models/rsf_dr/dr_learner.pkl

# The propensity model is saved with a custom PropensityModel wrapper class from the
# dev project (src.propensity_model). We extract the underlying GradientBoostingClassifier
# so the deploy app has no dependency on the dev project's src/ package.
echo "  → Extracting raw propensity model (removing dev-project class dependency) ..."
PRISM_SRC="$(dirname "$PRISM_RESULTS")"
python3 - <<PYEOF
import sys, pickle
sys.path.insert(0, '$PRISM_SRC')
with open('$PRISM_RESULTS/treatment_window_comparison/window_90_models/propensity_model.pkl', 'rb') as f:
    prop = pickle.load(f)
# Extract underlying sklearn model (GradientBoostingClassifier)
underlying = prop.model if hasattr(prop, 'model') else prop
with open('models/rsf_dr/propensity_model.pkl', 'wb') as f:
    pickle.dump(underlying, f)
print('  Propensity model extracted OK.')
PYEOF

# Causal Forest (§2.6.2)
echo "  → Causal Forest models (1–5 year) ..."
cp "$PRISM_RESULTS/cate_learners/models/causal_forest.pkl"    models/causal_forest/
cp "$PRISM_RESULTS/cate_learners/models/causal_forest_2y.pkl" models/causal_forest/
cp "$PRISM_RESULTS/cate_learners/models/causal_forest_3y.pkl" models/causal_forest/
cp "$PRISM_RESULTS/cate_learners/models/causal_forest_4y.pkl" models/causal_forest/
cp "$PRISM_RESULTS/cate_learners/models/causal_forest_5y.pkl" models/causal_forest/

# R-Learner (§2.6.3)
echo "  → R-Learner models (1–5 year) ..."
cp "$PRISM_RESULTS/cate_learners/models/r_learner.pkl"    models/r_learner/
cp "$PRISM_RESULTS/cate_learners/models/r_learner_2y.pkl" models/r_learner/
cp "$PRISM_RESULTS/cate_learners/models/r_learner_3y.pkl" models/r_learner/
cp "$PRISM_RESULTS/cate_learners/models/r_learner_4y.pkl" models/r_learner/
cp "$PRISM_RESULTS/cate_learners/models/r_learner_5y.pkl" models/r_learner/

# ACMM (§2.6.4)
echo "  → ACMM model ..."
cp "$PRISM_RESULTS/acmm_model/models/acmm_xgboost_calibrated.pkl" models/acmm/
cp "$PRISM_RESULTS/acmm_model/models/acmm_preprocessor.pkl"        models/acmm/

# Subgroup model (§2.7)
echo "  → Subgroup CATE models (18-feature) ..."
cp "$PRISM_RESULTS/subgroup_cate/models/subgroup_cf.pkl"          models/subgroup/
cp "$PRISM_RESULTS/subgroup_cate/models/subgroup_rl.pkl"          models/subgroup/
cp "$PRISM_RESULTS/subgroup_cate/models/subgroup_propensity.pkl"  models/subgroup/
cp "$PRISM_RESULTS/subgroup_cate/models/subgroup_preprocessor.pkl" models/subgroup/

# Causal preprocessor (MICE imputer + MinMaxScaler for 6-feature causal pipeline)
echo "  → Causal preprocessor ..."
if [ -f "$PRISM_RESULTS/causal_preprocessor.pkl" ]; then
  cp "$PRISM_RESULTS/causal_preprocessor.pkl" models/
else
  echo "  WARNING: causal_preprocessor.pkl not found."
  echo "  Run: cd $PRISM_RESULTS/../ && python scripts/save_causal_preprocessor.py"
fi

echo ""
echo "Done! Model sizes:"
du -sh models/rsf_dr/ models/causal_forest/ models/r_learner/ \
        models/acmm/ models/subgroup/ models/causal_preprocessor.pkl 2>/dev/null || true
echo ""
echo "Start the app:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "  # or: docker compose up"

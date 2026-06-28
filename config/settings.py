"""
config/settings.py
==================
Central configuration for the COF Iodine Uptake ML pipeline.
Modify this file to change dataset paths, CV folds, Optuna trials,
or any other experiment parameter without touching model/feature code.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

INPUT_CSV   = DATA_DIR / "cof_dataset_corrected_fixed.csv"   # ← place CSV here

OUTPUT_PREDICTIONS  = RESULTS_DIR / "cof_full_predictions_best.csv"
OUTPUT_TOP20        = RESULTS_DIR / "top20_cofs_best.csv"
OUTPUT_FEAT_IMP     = RESULTS_DIR / "feature_importance_best.csv"
OUTPUT_SUMMARY_JSON = RESULTS_DIR / "model_summary_best.json"
OUTPUT_DASHBOARD    = RESULTS_DIR / "cof_ml_dashboard_best.png"

# ── Cross-validation ────────────────────────────────────────────────────────
CV_FOLDS_EVAL   = 10   # folds used for final model evaluation
CV_FOLDS_TUNING = 3    # folds used inside Optuna objectives (faster)
CV_FOLDS_STACK  = 5    # folds used inside StackingRegressor
RANDOM_STATE    = 42

# ── KNN Imputer ─────────────────────────────────────────────────────────────
KNN_NEIGHBORS = 5

# ── Optuna trials ───────────────────────────────────────────────────────────
OPTUNA_TRIALS_XGB = 60
OPTUNA_TRIALS_LGB = 60
OPTUNA_TRIALS_RF  = 40

# ── Target & tier boundaries ────────────────────────────────────────────────
TARGET_COLUMN = "iodine_uptake_mg_g"
TIER_BINS     = [0, 3000, 5000, 7000, float("inf")]
TIER_LABELS   = ["Standard (<3 000)", "Good (3–5 k)", "High (5–7 k)", "Elite (>7 k)"]

# ── Dashboard style ─────────────────────────────────────────────────────────
FIGURE_DPI     = 160
BG_DARK        = "#0f1117"
BG_PANEL       = "#1a1d2e"
PALETTE = ["#5b8ff9", "#61d9aa", "#f6bd60", "#f08080",
           "#bf7fff", "#5dc8d4", "#ff9f7f"]

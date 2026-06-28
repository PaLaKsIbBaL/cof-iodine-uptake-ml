"""
main.py
=======
Entry point for the COF Iodine Uptake ML pipeline.

Usage
-----
    python main.py

All configuration (paths, CV folds, Optuna trials, etc.) lives in
config/settings.py — no need to edit this file for routine experiments.

Pipeline stages
---------------
1.  Load dataset                     (data/loader.py)
2.  Feature engineering + imputation (data/features.py)
3.  Bayesian hyperparameter tuning   (models/tuning.py)
4.  Build all estimators             (models/ensemble.py)
5.  10-fold CV evaluation            (models/evaluation.py)
6.  SHAP feature importance          (models/evaluation.py)
7.  Learning curve                   (models/evaluation.py)
8.  Annotate dataframe + save files  (utils/reporting.py)
9.  Generate dashboard figure        (visualization/dashboard.py)
"""

import warnings
warnings.filterwarnings("ignore")

# ── Internal modules ─────────────────────────────────────────────────────────
from data.loader   import load_dataset
from data.features import build_features

from models.tuning     import tune_all
from models.ensemble   import build_models
from models.evaluation import (
    evaluate_all,
    compute_shap_fitted,
    compute_learning_curve,
)

from utils.reporting import (
    annotate_dataframe,
    print_top20,
    print_summary_table,
    save_all,
)
from visualization.dashboard import plot_dashboard

from config.settings import OUTPUT_DASHBOARD
import xgboost as xgb


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_dataset()

    # ── 2. Feature engineering ────────────────────────────────────────────────
    X, y, feature_names = build_features(df)
    print(f"   Features: {X.shape[1]}  |  Samples: {X.shape[0]}")

    # ── 3. Hyperparameter tuning ──────────────────────────────────────────────
    best_params = tune_all(X, y)

    # ── 4. Build models ───────────────────────────────────────────────────────
    models = build_models(best_params)

    # ── 5. 10-fold CV evaluation ──────────────────────────────────────────────
    cv_results = evaluate_all(models, X, y)

    # Identify best model
    best_name = max(cv_results, key=lambda n: cv_results[n]["r2"])
    best_preds = cv_results[best_name]["preds"]
    best_r2    = cv_results[best_name]["r2"]

    # ── 6. SHAP (on tuned XGBoost — fastest TreeExplainer) ───────────────────
    xgb_model = xgb.XGBRegressor(
        **best_params["xgb"], random_state=42, verbosity=0, n_jobs=-1
    )
    xgb_model.fit(X, y)
    feat_imp = compute_shap_fitted(xgb_model, X, feature_names)

    # ── 7. Learning curve ─────────────────────────────────────────────────────
    print(f"\n📈 Computing learning curve for '{best_name}'…")
    train_sizes, train_scores, val_scores = compute_learning_curve(
        models[best_name], X, y
    )

    # ── 8. Annotate + save CSVs / JSON ───────────────────────────────────────
    df = annotate_dataframe(df, best_preds, y)
    print_top20(df)
    print_summary_table(cv_results, best_name, X)
    top20 = save_all(df, feat_imp, cv_results, best_params, best_name, X)

    # ── 9. Dashboard ──────────────────────────────────────────────────────────
    plot_dashboard(
        cv_results   = cv_results,
        best_name    = best_name,
        y            = y,
        feat_imp     = feat_imp,
        top20        = top20,
        train_sizes  = train_sizes,
        train_scores = train_scores,
        val_scores   = val_scores,
        output_path  = OUTPUT_DASHBOARD,
    )

    print("\n🎉 Pipeline complete.")


if __name__ == "__main__":
    main()

# COF Iodine Uptake Prediction — ML Pipeline

**Machine learning prediction of iodine uptake capacity in Covalent Organic Frameworks (COFs)**  
*Research-paper-ready modular pipeline with Bayesian hyperparameter optimisation and stacking ensemble*

---

## Overview

This repository accompanies the manuscript:

> *"Machine Learning-Accelerated Prediction of Iodine Uptake in Covalent Organic Frameworks"*

The pipeline trains and evaluates six regression models to predict iodine uptake (mg/g) from
structural and chemical descriptors. A 5-model stacking ensemble with Ridge meta-learner achieves
the best cross-validated R² in our benchmarks.

---

## Repository Structure

```
cof_iodine_ml/
│
├── main.py                    # ← Single entry point; run this
│
├── config/
│   └── settings.py            # All hyperparameters, paths, CV folds
│
├── data/
│   ├── loader.py              # CSV loading + sanity checks
│   └── features.py            # Feature engineering (46 descriptors)
│
├── models/
│   ├── tuning.py              # Optuna Bayesian search (XGB / LGB / RF)
│   ├── ensemble.py            # Model definitions & stacking
│   └── evaluation.py          # 10-fold CV, SHAP, learning curves
│
├── visualization/
│   └── dashboard.py           # 3×3 research dashboard (Figure 1)
│
├── utils/
│   └── reporting.py           # Console tables + CSV/JSON output
│
├── data/                      # ← Place your CSV here
│   └── cof_dataset_corrected_fixed.csv
│
├── results/                   # Auto-created on first run
│   ├── cof_ml_dashboard_best.png
│   ├── cof_full_predictions_best.csv
│   ├── top20_cofs_best.csv
│   ├── feature_importance_best.csv
│   └── model_summary_best.json
│
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone / download the repository
cd cof_iodine_ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
cp /path/to/cof_dataset_corrected_fixed.csv data/

# 4. Run the full pipeline
python main.py
```

All outputs are written to `results/`.

---

## Feature Engineering (46 descriptors)

| Group | Count | Description |
|---|---|---|
| Raw structural / textural | 3 | BET surface area, pore volume, pore size |
| Binary chemical flags | 4 | N-site, S-site, ionic, TTF/S donor |
| Encoded categoricals | 3 | Linkage type, topology, data quality (ordinal) |
| Molecular descriptors | 14 | MW, logP, TPSA, H-donors/acceptors, ring counts, etc. |
| Original interaction terms | 9 | N×BET, S×pore, ionic×logP, aromatic ratio, etc. |
| **New chemistry features** | **11** | BET×pore, π-surface, heteroatom score, conjugation score, etc. |

Missing values are imputed with KNN (k=5) prior to model training.

---

## Models

| Model | Notes |
|---|---|
| XGBoost | Bayesian-tuned, 60 Optuna trials |
| LightGBM | Bayesian-tuned, 60 Optuna trials |
| Random Forest | Bayesian-tuned, 40 Optuna trials |
| Extra Trees | Fixed architecture (800 trees) |
| Hist GradBoost | Fixed architecture (600 iterations) |
| **Stacking Ensemble** | XGB + LGB + RF + ET + HGBR → Ridge meta-learner (`passthrough=True`) |

All models are evaluated with **10-fold cross-validation**. Predictions are clipped to [0, ∞).

---

## Outputs

| File | Description |
|---|---|
| `cof_ml_dashboard_best.png` | 3×3 figure: model comparison, scatter, residuals, SHAP, error dist., top-20 bar, learning curve |
| `cof_full_predictions_best.csv` | Full dataset with CV predictions, error %, tier labels |
| `top20_cofs_best.csv` | Top 20 COFs ranked by actual uptake |
| `feature_importance_best.csv` | Mean \|SHAP\| values for all 46 features |
| `model_summary_best.json` | All metrics + best hyperparameters (reproducibility) |

---

## Configuration

Edit `config/settings.py` to change any experiment parameter without touching model code:

```python
OPTUNA_TRIALS_XGB = 60   # increase for more thorough search
CV_FOLDS_EVAL     = 10   # outer CV folds
KNN_NEIGHBORS     = 5    # imputation neighbours
INPUT_CSV         = ...  # path to dataset
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2025cof,
  title   = {Machine Learning-Accelerated Prediction of Iodine Uptake in Covalent Organic Frameworks},
  author  = {Your Name et al.},
  journal = {Journal Name},
  year    = {2025},
}
```

---

## License

MIT

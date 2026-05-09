# Loan Approval — Two-Stage ML Application

## Overview
A two-stage machine learning pipeline that first predicts whether a loan application will be approved or rejected, and if approved, estimates the eligible loan amount.

- **Stage 1 — Classification**: Predict `approved` / `rejected` using applicant features including CIBIL score, income, assets, and loan details
- **Stage 2 — Regression**: For approved applicants, predict the loan amount they are eligible for (trained on approved loans only)

Both stages use **Gradient Boosting** models selected from systematic experimentation.

## Key Findings (from EDA)
- **CIBIL score** is the dominant predictor — approved loans cluster around 618–803, rejected around 364–493
- Education and self-employment status have no meaningful effect on approval (~62% approval rate regardless)
- Income and asset values are highly correlated with each other but not with CIBIL score
- Loan amount prediction is largely driven by income (near-linear relationship)

## Project Structure
```
├── 1_EDA.ipynb              # Exploratory data analysis
├── 2_Experiments.ipynb      # Model comparison across classifiers and regressors
├── 3_Modeling.ipynb         # Final pipeline with hyperparameter tuning + model saving
├── main.py                  # CLI application for predictions
├── streamlit_app.py         # Streamlit web UI
├── config.yaml              # Model paths and default UI inputs
├── app/
│   ├── loader.py            # Loads saved model pipelines
│   ├── predict.py           # Two-stage prediction logic
│   └── utils.py             # Input formatting helpers
├── models/
│   ├── stage_1_gb_classifier.pkl
│   └── stage_2_gb_regressor.pkl
└── archive/
    └── loan_approval_dataset.csv
```

## Workflow
```
1_EDA.ipynb → 2_Experiments.ipynb → 3_Modeling.ipynb → models/*.pkl → main.py / streamlit_app.py
```

1. **EDA** — understand the data, distributions, correlations, and engineered features
2. **Experiments** — compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost for both tasks
3. **Modeling** — tune the winning model (Gradient Boosting) with GridSearchCV and save artifacts
4. **Inference** — load saved models via CLI (`main.py`) or web UI (`streamlit_app.py`)

## Quick Start

**1. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train and save models** — run all cells in `3_Modeling.ipynb`

**4. Run the CLI**
```bash
python main.py
```

**5. Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

## Model Performance

| Task | Model | Metric |
|---|---|---|
| Classification | Gradient Boosting | F1: 0.986 |
| Regression | Gradient Boosting | R²: 0.879, MAE: ~2.4M |

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- xgboost, joblib, pyyaml, streamlit

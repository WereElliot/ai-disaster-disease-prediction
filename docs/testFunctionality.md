# Verification Procedure: Phases 1 to 5
This document provides a step-by-step guide to verify the functionality of the AI-Based Disaster and Disease Outbreak Prediction Platform from data ingestion to model evaluation.

---

## 1. Environment Setup
**Goal**: Ensure all dependencies are installed and the environment is ready.
- **Action**: Run the following command:
  ```bash
  pip install -r requirements.txt
  pip install pyyaml requests requests-cache scikit-learn xgboost pandas numpy pyarrow fastparquet scipy matplotlib seaborn shap lime
  ```
- **Verification**: Ensure no `ModuleNotFoundError` occurs when running the scripts.

---

## 2. Phase 1: Data Ingest & Pipeline
**Goal**: Download, clean, merge, and engineer features for the dataset.
- **Action**: Run the ingestion pipeline:
  ```bash
  $env:PYTHONPATH = "."; python src/main.py --mode ingest
  ```
- **Expected Output**:
  - `data/raw/`: Contains `climate.json`, `health.json`, and `disasters.json`.
  - `data/processed/final_dataset.parquet`: A single file containing ~35,000 processed samples with engineered features.
  - Console output: "Pipeline completed in X seconds".

---

## 3. Phase 2: EDA & Baselines
**Goal**: Review data insights and establish initial benchmarks.
- **Action**:
  1. Open `notebooks/eda.ipynb` or `notebooks/01_data_overview.ipynb` to view summary statistics.
  2. Review `docs/EDA_Report.md` for feature importance and data quality findings.
- **Expected Output**:
  - `docs/EDA_Report.md` exists and contains a list of top features (e.g., Temperature, Humidity).

---

## 4. Phase 4: Core Hybrid Model Training
**Goal**: Train the Hybrid LSTM + XGBoost model using the 70/15/15 split.
- **Action**: Run the training pipeline:
  ```bash
  $env:PYTHONPATH = "."; python src/main.py --mode train
  ```
- **Expected Output**:
  - `models/dl_feature_extractor.pkl`: Saved DL/LSTM proxy model.
  - `models/xgb_classifier.pkl`: Saved XGBoost classifier.
  - Console output showing "Best XGB Params" and "Models saved successfully".

---

## 5. Phase 5: Explainable AI (XAI) Layer
**Goal**: Generate SHAP and LIME interpretations for the model.
- **Action**: Run the XAI layer:
  ```bash
  $env:PYTHONPATH = "."; python src/main.py --mode explain
  ```
- **Expected Output**:
  - `docs/xai/shap_summary.png`: Global feature importance plot.
  - `docs/xai/shap_bar.png`: Feature impact bar chart.
  - `docs/xai/lime_local_explanation.html`: Interactive local explanation for an individual prediction.

---

## 6. Phase 6: Model Evaluation Framework
**Goal**: Validate the hybrid model against baselines on the 15% test set.
- **Action**: Run the evaluation suite:
  ```bash
  $env:PYTHONPATH = "."; python src/main.py --mode evaluate
  ```
- **Expected Output**:
  - `docs/Evaluation_Report.md`: Contains F1, Accuracy, Precision, Recall, and ROC-AUC scores.
  - `docs/evaluation/confusion_matrix.png`: Plot of classification accuracy.
  - `docs/evaluation/roc_curve.png`: ROC plot for the hybrid model.
  - `docs/evaluation/error_distribution.png`: Histogram of prediction errors.

---

## Summary Checklist
| Phase | Task | Success Criteria |
|---|---|---|
| 1 | Ingest | `final_dataset.parquet` exists |
| 2 | EDA | `EDA_Report.md` generated |
| 3 | Train | Models saved in `models/` |
| 4 | Explain | SHAP/LIME plots in `docs/xai/` |
| 5 | Evaluate | F1 > 0.85 (Target) and plots in `docs/evaluation/` |

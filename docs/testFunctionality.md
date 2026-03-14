# Verification Procedure: Phases 1 to 5
This document provides a step-by-step guide to verify the functionality of the AI-Based Disaster and Disease Outbreak Prediction Platform from data ingestion to model evaluation.

---

## 1. Environment Setup
**Goal**: Ensure all dependencies are installed and the environment is ready.
- **Action**: Run the following command:
  ```powershell
  pip install -r requirements.txt
  ```
- **Verification**: Ensure no `ModuleNotFoundError` occurs when running the scripts.

---

## 2. Full Pipeline Execution (Recommended)
**Goal**: Run all phases (1-5) sequentially to verify the entire system.
- **Action**: Run the full pipeline command:
  ```powershell
  $env:PYTHONPATH = '.'; python src/main.py --all
  ```
- **Expected Results**:
  - **Runtime**: ~25-30 seconds.
  - **Console Output**: "Verification Summary: All phases completed successfully. Reports saved in docs/."
  - **Artifacts**: All reports and plots generated in `docs/` and models saved in `models/`.

---

## 3. Phase 1: Data Ingest & Pipeline
**Goal**: Download, clean, merge, and engineer features for the dataset.
- **Action**: Run the ingestion pipeline:
  ```powershell
  $env:PYTHONPATH = "."; python src/main.py --mode ingest
  ```
- **Expected Output**:
  - `data/raw/`: Contains `climate.json`, `health.json`, and `disasters.json`.
  - `data/processed/final_dataset.parquet`: Processed samples with engineered features (lags 1-3).
  - Console output: "Pipeline completed in X seconds".

---

## 4. Phase 2: EDA & Baselines
**Goal**: Review data insights and establish initial benchmarks.
- **Action**: Run the EDA phase:
  ```powershell
  $env:PYTHONPATH = "."; python src/main.py --mode eda
  ```
- **Expected Output**:
  - `docs/plots/`: Contains `cases_over_time.png` and `correlation_heatmap.png`.
  - `docs/EDA_Report.md`: Contains summary statistics and correlation findings.

---

## 5. Phase 3: Hybrid Model Training
**Goal**: Train the Hybrid LSTM Proxy + XGBoost model.
- **Action**: Run the training pipeline:
  ```powershell
  $env:PYTHONPATH = "."; python src/main.py --mode train
  ```
- **Expected Output**:
  - `models/dl_feature_extractor.pkl`: Saved MLP-LSTM proxy model.
  - `models/xgb_classifier.pkl`: Saved XGBoost classifier.
  - Console output showing "Hybrid training completed. Best F1: X.XXXX".

---

## 6. Phase 4: Explainable AI (XAI) Layer
**Goal**: Generate SHAP and LIME interpretations for the model.
- **Action**: Run the XAI layer:
  ```powershell
  $env:PYTHONPATH = "."; python src/main.py --mode explain
  ```
- **Expected Output**:
  - `docs/xai/shap_summary.png`: Global feature importance plot.
  - `docs/xai/shap_bar.png`: Feature impact bar chart (using symlog scale).
  - `docs/xai/lime_local_explanation.html`: Local explanation for a high-risk prediction.

---

## 7. Phase 5: Model Evaluation Framework
**Goal**: Validate the hybrid model against baselines.
- **Action**: Run the evaluation suite:
  ```powershell
  $env:PYTHONPATH = "."; python src/main.py --mode evaluate
  ```
- **Expected Output**:
  - `docs/Evaluation_Report.md`: Final F1, Accuracy, Precision, Recall, and ROC-AUC scores.
  - `docs/evaluation/`: Contains `confusion_matrix.png` and `roc_curve.png`.

---

## Summary Checklist
| Phase | Task | Success Criteria |
|---|---|---|
| 1 | Ingest | `final_dataset.parquet` exists |
| 2 | EDA | `EDA_Report.md` & plots generated |
| 3 | Train | Models saved in `models/`, F1 > 0.85 |
| 4 | Explain | SHAP/LIME plots in `docs/xai/` |
| 5 | Evaluate | Final report in `docs/Evaluation_Report.md` |

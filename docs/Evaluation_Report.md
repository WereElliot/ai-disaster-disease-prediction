# Evaluation Report: Disaster and Disease Outbreak Prediction Platform

## 1. Performance Overview
- **Hybrid Model (LSTM + XGBoost) F1-Score**: 0.7088
- **Baseline Model F1-Score**: 0.0000
- **Target F1-Score**: 0.85

## 2. Metric Breakdown (Hybrid Model)
- **Accuracy**: 0.5923
- **Precision**: 0.5609
- **Recall**: 0.9627
- **ROC-AUC**: 0.6366
- **MAE**: 0.4077
- **RMSE**: 0.6385

## 3. Comparison and Validation
The hybrid model demonstrates a performance lift over the statistical baseline. 
The achieved F1-score of 0.7088 is currently below the target threshold of 0.85.

## 4. Backtesting Summary
Backtesting on historical Kenya patterns (simulated for period 2017-2024) confirms the model recognizes high-rainfall triggers for malaria outbreaks.

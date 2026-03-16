import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any

class StatisticalBaseline:
    """Moving Average baseline for time-series forecasting."""
    def __init__(self, window: int = 12):
        self.window = window

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        # Vectorized moving average
        return df[target].rolling(window=self.window).mean().shift(1).fillna(df[target].mean()).values

class BaselineLR:
    """Logistic Regression baseline for classification."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # Convert target to binary for classification (outbreak vs no outbreak)
        y_binary = (y_train > np.median(y_train)).astype(int)
        self.model.fit(X_train, y_binary)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

def evaluate_baselines(df: pd.DataFrame, target: str) -> Dict[str, float]:
    """Run and evaluate baselines on the provided dataset."""
    X = df.drop(columns=[target, 'date', 'location'], errors='ignore').values
    y = df[target].values
    
    # Statistical Baseline
    sb = StatisticalBaseline(window=12)
    y_pred_sb = sb.predict(df, target)
    sb_mae = np.mean(np.abs(y - y_pred_sb))
    
    # LR Baseline
    y_binary = (y > np.median(y)).astype(int)
    lr = BaselineLR()
    lr.train(X, y)
    y_pred_lr = lr.predict(X)
    
    metrics = {
        "sb_mae": float(sb_mae),
        "lr_accuracy": float(accuracy_score(y_binary, y_pred_lr)),
        "lr_f1": float(f1_score(y_binary, y_pred_lr))
    }
    
    return metrics

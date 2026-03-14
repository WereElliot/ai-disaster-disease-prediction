import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from typing import Dict, Any, List, Tuple

def filter_by_pearson(df: pd.DataFrame, target: str, threshold: float = 0.7) -> List[str]:
    """Calculate Pearson correlation (>0.7 threshold)."""
    correlations = df.corr()[target].abs().sort_values(ascending=False)
    selected_features = correlations[correlations > threshold].index.tolist()
    return selected_features

def select_features_rfe(df: pd.DataFrame, target: str, config: Dict[str, Any]) -> List[str]:
    """RFE with XGBoost."""
    X = df.drop(columns=[target, 'date', 'location'], errors='ignore')
    y = df[target]
    
    n_features = config['data']['processing']['features']['rfe_n_features']
    n_features = min(n_features, X.shape[1])
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    
    selector = RFE(estimator=model, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.support_].tolist()
    return selected_features

def engineer_features(df: pd.DataFrame, target: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Orchestrate feature selection pipeline."""
    # 1. Initial Filtering (Pearson)
    threshold = config['data']['processing']['features']['pearson_threshold']
    # pearson_selected = filter_by_pearson(df, target, threshold)
    
    # 2. RFE Selection
    best_features = select_features_rfe(df, target, config)
    
    # 3. Final dataset with selected features + target + metadata
    selected_df = df[best_features + [target, 'date', 'location']]
    
    return selected_df

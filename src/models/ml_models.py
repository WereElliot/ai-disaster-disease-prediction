from xgboost import XGBClassifier
from typing import Dict, Any

def get_xgb_classifier(config: Dict[str, Any]) -> XGBClassifier:
    """
    XGBoost Classifier for final prediction (classification).
    
    Parameters:
    - n_estimators: 200
    - max_depth: 5
    - learning_rate: 0.1
    """
    xgb_config = config['models']['xgboost']
    
    return XGBClassifier(
        n_estimators=xgb_config['n_estimators'],
        max_depth=xgb_config['max_depth'],
        learning_rate=xgb_config['learning_rate'],
        random_state=config['models']['hybrid']['random_seed'],
        n_jobs=-1, # Optimized for maximum parallelization
        eval_metric='logloss'
    )

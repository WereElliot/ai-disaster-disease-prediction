import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from src.utils.config import load_config
from src.models.dl_models import build_lstm_model, create_lstm_dataset
from src.models.ml_models import get_xgb_classifier
from pathlib import Path

def add_lagged_features(df: pd.DataFrame, target: str, features: list, lags=[1, 2, 3]) -> pd.DataFrame:
    """Add lagged features to capture temporal patterns for climate and health."""
    df = df.copy()
    for lag in lags:
        for feat in features + [target]:
            df[f'{feat}_lag_{lag}'] = df.groupby('location')[feat].shift(lag)
    return df.dropna().reset_index(drop=True)

def train_hybrid_pipeline():
    """Execute Phase 3: Train Hybrid Model (MLP Proxy + XGBoost)."""
    config = load_config()
    np.random.seed(config['models']['hybrid']['random_seed'])

    # 1. Load and Prepare
    processed_path = Path(config['data']['paths']['processed']) / "final_dataset.parquet"
    df = pd.read_parquet(processed_path)

    target_col = 'cases'
    # Exclude target and raw disease columns to prevent leakage
    exclude_cols = [target_col, 'date', 'location', 'malaria_cases', 'cholera_cases', 'dengue_cases', 'diarrheal_cases']
    base_features = [c for c in df.columns if c not in exclude_cols]
    
    # Feature Engineering (Lags)
    df_lagged = add_lagged_features(df, target_col, base_features)
    features = [c for c in df_lagged.columns if c not in exclude_cols]

    # Uniform Scaling BEFORE splitting
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numeric_cols = df_lagged.select_dtypes(include=[np.number]).columns
    # Ensure all features and target are in numeric_cols
    df_lagged[numeric_cols] = scaler.fit_transform(df_lagged[numeric_cols])

    # 2. Split (on already scaled data)
    test_val_size = config['models']['hybrid']['test_size'] + config['models']['hybrid']['val_size']
    train_df, test_val_df = train_test_split(df_lagged, test_size=test_val_size, shuffle=False)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, shuffle=False)

    # 3. Proxy Model (Feature Extraction)
    lookback = config['models']['lstm']['lookback']
    # Reshaping for MLP fallback (flattening)
    X_train_dl, y_train_dl = create_lstm_dataset(train_df[features + [target_col]].values, lookback)
    X_val_dl, y_val_dl = create_lstm_dataset(val_df[features + [target_col]].values, lookback)

    dl_model = build_lstm_model(input_shape=(lookback, len(features)), units=config['models']['lstm']['units'])
    dl_model.fit(X_train_dl, y_train_dl)

    # 4. XGBoost Hybrid
    train_dl_feats = dl_model.predict(X_train_dl).reshape(-1, 1)
    val_dl_feats = dl_model.predict(X_val_dl).reshape(-1, 1)

    # XGBoost trained on scaled features
    X_train_xgb = np.hstack([train_df[features].values[lookback:], train_dl_feats])
    # Target for classification: Outbreak detection (Above median of SCALED target)
    y_train_xgb = (train_df[target_col].values[lookback:] > train_df[target_col].median()).astype(int)

    xgb_base = get_xgb_classifier(config)

    # Simple grid search for F1 optimization
    param_grid = {
        'max_depth': [3, 6, 9],
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1]
    }

    grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train_xgb, y_train_xgb)

    best_model = grid.best_estimator_

    # 5. Save
    model_dir = Path(config['data']['paths']['models'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "dl_feature_extractor.pkl", "wb") as f:
        pickle.dump(dl_model, f)
    with open(model_dir / "xgb_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Save feature names for consistent prediction
    with open(model_dir / "features.pkl", "wb") as f:
        pickle.dump({
            'features': features, 
            'target': target_col, 
            'numeric_cols': numeric_cols.tolist(),
            'lookback': lookback
        }, f)

    print(f"Hybrid training completed. Best F1: {grid.best_score_:.4f}")
if __name__ == "__main__":
    train_hybrid_pipeline()

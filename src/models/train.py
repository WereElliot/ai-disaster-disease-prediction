import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from src.utils.config import load_config
from src.models.dl_models import build_lstm_model, create_lstm_dataset
from src.models.ml_models import get_xgb_classifier
from pathlib import Path

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠️  imbalanced-learn not installed. SMOTE will be skipped.")

def add_lagged_features(df: pd.DataFrame, target: str, features: list, lags=[1, 2, 3]) -> pd.DataFrame:
    """Add lagged features to capture temporal patterns for climate and health."""
    df = df.copy()
    
    # If dataset is very small, reduce lags
    if len(df) < 100:
        lags = [1]  # Only use lag-1 for small datasets
        print(f"⚠️  Small dataset detected. Using lags={lags} instead")
    
    for lag in lags:
        for feat in features + [target]:
            df[f'{feat}_lag_{lag}'] = df.groupby('location')[feat].shift(lag)
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    # If all rows were dropped, return original features with no lags
    if len(df) == 0:
        print(f"⚠️  All rows dropped due to NaN in lagged features. Using original features only.")
        df = pd.DataFrame()  # Return empty, will handle in train_hybrid_pipeline
    
    return df

def train_hybrid_pipeline():
    """Execute Phase 3: Train Hybrid Model (MLP Proxy + XGBoost)."""
    config = load_config()
    np.random.seed(config['models']['hybrid']['random_seed'])

    # 1. Load and Prepare
    processed_path = Path(config['data']['paths']['processed']) / "final_dataset.parquet"
    
    # If processed dataset doesn't exist, try to create it from sample data
    if not processed_path.exists():
        print(f"⚠️  Processed dataset not found at {processed_path}")
        print("Creating dataset from sample fixtures...")
        
        # Load sample data
        base_path = Path(__file__).parent.parent.parent
        climate_path = base_path / 'tests' / 'fixtures' / 'climate_sample.csv'
        disasters_path = base_path / 'tests' / 'fixtures' / 'disasters_sample.csv'
        health_path = base_path / 'tests' / 'fixtures' / 'health_sample.csv'
        
        if not all([climate_path.exists(), disasters_path.exists(), health_path.exists()]):
            raise FileNotFoundError(f"Sample fixtures not found. Expected:")
            print(f"  - {climate_path}")
            print(f"  - {disasters_path}")
            print(f"  - {health_path}")
        
        # Load and merge data
        climate_df = pd.read_csv(climate_path)
        disasters_df = pd.read_csv(disasters_path)
        health_df = pd.read_csv(health_path)
        
        print(f"✓ Loaded climate data: {climate_df.shape}")
        print(f"✓ Loaded disasters data: {disasters_df.shape}")
        print(f"✓ Loaded health data: {health_df.shape}")
        
        # For simplicity, use health data as primary (has 'cases' column)
        df = health_df.copy()
        
        # Rename 'location' column if it doesn't exist
        if 'location' not in df.columns and 'Location' in df.columns:
            df.rename(columns={'Location': 'location'}, inplace=True)
        if 'location' not in df.columns:
            df['location'] = 'Unknown'
        
        # Rename 'date' column if needed
        if 'date' not in df.columns and 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        if 'date' not in df.columns and 'date_start' in df.columns:
            df.rename(columns={'date_start': 'date'}, inplace=True)
        
        print(f"✓ Using health data as primary dataset")
    else:
        print(f"✓ Loading processed dataset: {processed_path}")
        df = pd.read_parquet(processed_path)
    
    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")

    target_col = 'cases'
    # Exclude target and raw disease columns to prevent leakage
    exclude_cols = [target_col, 'date', 'location', 'malaria_cases', 'cholera_cases', 'dengue_cases', 'diarrheal_cases']
    base_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.number, 'float64', 'int64']]
    
    print(f"✓ Using {len(base_features)} features for training")
    if len(base_features) == 0:
        print("⚠️  No numeric features found. Using synthetic features for demonstration.")
        # Create synthetic features for demo
        df['feature_1'] = np.random.randn(len(df))
        df['feature_2'] = np.random.randn(len(df))
        base_features = ['feature_1', 'feature_2']
    
    # Feature Engineering (Lags)
    df_lagged = add_lagged_features(df, target_col, base_features)
    
    # If lagged features failed, use original features without lags
    if df_lagged.empty:
        print("Using original features without lag transformation")
        df_lagged = df.dropna().reset_index(drop=True)
        if len(df_lagged) == 0:
            raise ValueError("Insufficient data after removing NaN values")
        features = base_features
    else:
        features = [c for c in df_lagged.columns if c not in exclude_cols]
    
    print(f"✓ Features for training: {features[:3]}{'...' if len(features) > 3 else ''} ({len(features)} total)")

    # Data Validation: Remove rows with NaNs or infinite values
    df_lagged = df_lagged.replace([np.inf, -np.inf], np.nan)
    df_lagged = df_lagged.dropna()
    
    # Uniform Scaling BEFORE splitting
    scaler = MinMaxScaler()
    numeric_cols = df_lagged.select_dtypes(include=[np.number]).columns
    df_lagged[numeric_cols] = scaler.fit_transform(df_lagged[numeric_cols])
    
    print(f"Data shape after cleaning: {df_lagged.shape}")
    print(f"Features: {len(features)}")

    # 2. Split (on already scaled data)
    test_val_size = config['models']['hybrid']['test_size'] + config['models']['hybrid']['val_size']
    train_df, test_val_df = train_test_split(df_lagged, test_size=test_val_size, shuffle=False)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, shuffle=False)

    # 3. Proxy Model (Feature Extraction)
    lookback = config['models']['lstm']['lookback']
    
    # Adjust lookback for small datasets
    if len(df_lagged) < lookback * 2:
        lookback = max(1, len(df_lagged) // 3)
        print(f"⚠️  Adjusting lookback to {lookback} due to small dataset size")
    
    # Reshaping for MLP fallback (flattening)
    try:
        X_train_dl, y_train_dl = create_lstm_dataset(train_df[features + [target_col]].values, lookback)
        X_val_dl, y_val_dl = create_lstm_dataset(val_df[features + [target_col]].values, lookback)
        
        if len(X_train_dl) == 0:
            print(f"⚠️  No training sequences created with lookback={lookback}")
            raise ValueError("Insufficient data for sequence creation")
    except Exception as e:
        print(f"Error creating sequences: {e}")
        print(f"  Data shape: {train_df.shape}")
        print(f"  Lookback: {lookback}")
        raise
    
    print(f"✓ Created {len(X_train_dl)} training sequences, {len(X_val_dl)} validation sequences")

    dl_model = build_lstm_model(input_shape=(lookback, len(features)), units=config['models']['lstm']['units'])
    dl_model.fit(X_train_dl, y_train_dl)
    print(f"✓ Proxy model (MLP) trained")

    # 4. XGBoost Hybrid
    train_dl_feats = dl_model.predict(X_train_dl).reshape(-1, 1)
    val_dl_feats = dl_model.predict(X_val_dl).reshape(-1, 1)

    # XGBoost trained on scaled features
    X_train_xgb = np.hstack([train_df[features].values[lookback:], train_dl_feats])
    
    # Target for classification: Outbreak detection (Above median of SCALED target)
    y_train_xgb = (train_df[target_col].values[lookback:] > train_df[target_col].median()).astype(int)
    
    # Apply SMOTE to handle class imbalance (only on training set)
    print(f"Class distribution before SMOTE: {np.bincount(y_train_xgb)}")
    if HAS_SMOTE:
        try:
            smote = SMOTE(random_state=config['models']['hybrid']['random_seed'], k_neighbors=3)
            X_train_xgb, y_train_xgb = smote.fit_resample(X_train_xgb, y_train_xgb)
            print(f"Class distribution after SMOTE: {np.bincount(y_train_xgb)}")
        except Exception as e:
            print(f"SMOTE failed: {e}. Proceeding without it.")
    
    xgb_base = get_xgb_classifier(config)

    # Grid search for F1 optimization with stratified k-fold
    param_grid = {
        'max_depth': [4, 7, 10],
        'n_estimators': [150, 250, 350],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['models']['hybrid']['random_seed'])
    grid = GridSearchCV(xgb_base, param_grid, cv=skf, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train_xgb, y_train_xgb)
    
    print(f"Best F1 params: {grid.best_params_}")
    print(f"Best F1 score (CV): {grid.best_score_:.4f}")

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

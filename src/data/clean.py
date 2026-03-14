import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from typing import Dict, Any

def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Perform production-ready data cleaning and preprocessing."""
    
    # 1. Z-score Outlier Removal (Vectorized)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not df.empty and len(df) > 1:
        z_threshold = config['data']['processing']['cleaning']['z_threshold']
        # Calculate z-scores, handling NaNs
        z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
        # Keep rows where NO column exceeds threshold
        mask = (z_scores < z_threshold).all(axis=1)
        df = df[mask]

    if df.empty:
        return df

    # 2. KNN Imputation (k=5)
    knn_neighbors = min(config['data']['processing']['cleaning']['knn_neighbors'], len(df))
    if knn_neighbors > 0:
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # 3. Min-Max Scaling (Vectorized)
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 4. Monthly Resampling (Temporal Alignment)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Keep non-numeric columns if they are constant (like 'location')
        # Otherwise, just resample numeric ones
        resampled_numeric = df.set_index('date').select_dtypes(include=[np.number]).resample('ME').mean()
        
        # Try to preserve 'location' if it's unique or just take the first
        if 'location' in df.columns:
            locations = df.set_index('date')['location'].resample('ME').first()
            resampled = pd.concat([resampled_numeric, locations], axis=1).reset_index()
        else:
            resampled = resampled_numeric.reset_index()
        
        return resampled

    return df

def preprocess_sources(raw_paths: Dict[str, str], config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Preprocess individual raw sources before merging."""
    # Simplified: Assuming dataframes are already created from raw files
    # In practice: pd.read_json/csv depending on source
    processed_dfs = {}
    for source, path in raw_paths.items():
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_json(path)
        processed_dfs[source] = clean_data(df, config)
    return processed_dfs

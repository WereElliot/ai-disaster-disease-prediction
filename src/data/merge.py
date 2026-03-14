from typing import List, Dict, Any
import pandas as pd
from functools import reduce

def merge_datasets(dataframes: List[pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
    """Perform temporal/spatial alignment on date + location keys."""
    
    # 1. Standardize keys
    for df in dataframes:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'location' not in df.columns:
            df['location'] = 'Global' # Placeholder for spatial alignment

    # 2. Sequential merge (Temporal + Spatial alignment)
    # Using 'outer' join to preserve all records initially, then handle NaNs
    merged_df = reduce(lambda left, right: pd.merge(
        left, right, on=['date', 'location'], how='outer'
    ), dataframes)

    # 3. Final cleaning after merge
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.sort_values(['date', 'location']).reset_index(drop=True)

    return merged_df

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def engineer_features(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """Add derived signals useful for models."""

    features = df.copy()
    if not features.index.size:
        return features

    if "temperature" in features.columns and "humidity" in features.columns:
        features["temp_humidity_index"] = features["temperature"] * features["humidity"]

    if "precipitation" in features.columns and "cases" in features.columns:
        features["rain_to_case_ratio"] = features["precipitation"].divide(
            features["cases"].replace(0, pd.NA)
        ).fillna(0)

    if "cases" in features.columns:
        features["cases_growth"] = features["cases"].diff().fillna(0)
        window = max(1, rolling_window)
        features[f"cases_roll_mean_{window}"] = (
            features["cases"].rolling(window=window, min_periods=1).mean()
        )

    if "affected_population" in features.columns and "cases" in features.columns:
        features["impact_ratio"] = features["cases"].divide(
            features["affected_population"].replace(0, pd.NA)
        ).fillna(0)

    return features


def persist_features(
    features_df: pd.DataFrame, processed_dir: str, filename: str = "features.csv"
) -> Path:
    """Persist engineered features under the processed features folder."""

    features_dir = Path(processed_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    target_path = features_dir / filename
    features_df.to_csv(target_path, index=False)
    return target_path

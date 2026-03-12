from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data.feature_engineer import engineer_features, persist_features
from src.utils.config import load_config


def feature_engineering(
    config: Dict[str, Any] | None = None, merged_path: str | None = None
) -> Path | None:
    """Apply feature engineering to the merged dataset."""

    if config is None:
        config = load_config()

    processed_dir = config["data"]["paths"]["processed"]
    feature_cfg = config["data"]["processing"]["features"]
    rolling_window = feature_cfg.get("rolling_window", 3)

    if merged_path is None:
        merged_path = Path(processed_dir) / "merged" / "merged_data.csv"

    if not Path(merged_path).exists():
        print("Merged dataset missing; cannot run feature engineering.")
        return None

    merged_df = pd.read_csv(merged_path)
    features_df = engineer_features(merged_df, rolling_window=rolling_window)
    saved = persist_features(features_df, processed_dir)
    print(f"Saved features to {saved}")
    return saved

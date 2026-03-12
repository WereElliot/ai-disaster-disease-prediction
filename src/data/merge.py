from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _determine_merge_keys(dataframes: List[pd.DataFrame], preferred_keys: Iterable[str]) -> List[str]:
    if not dataframes:
        return []

    available = set(dataframes[0].columns)
    keys = [key for key in preferred_keys if key in available]
    return keys


def merge_dataframes(
    dataframes: List[pd.DataFrame], merge_keys: Iterable[str], how: str = "outer"
) -> pd.DataFrame:
    """Merge a list of dataframes on the configured key columns."""

    if not dataframes:
        raise ValueError("No dataframes provided for merging.")

    keys = _determine_merge_keys(dataframes, merge_keys)
    if not keys:
        raise ValueError("Could not determine merge keys that exist in the data.")

    return reduce(lambda left, right: pd.merge(left, right, on=keys, how=how), dataframes)


def merge_cleaned_sources(
    clean_paths: Iterable[str], merge_keys: Iterable[str], processed_dir: str
) -> Path:
    """Load cleaned files, merge them, and persist the consolidated dataset."""

    loaded = [pd.read_csv(Path(path)) for path in clean_paths if Path(path).exists()]
    if not loaded:
        raise FileNotFoundError("No cleaned data files available for merging.")

    merged = merge_dataframes(loaded, merge_keys)
    merged_dir = Path(processed_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    target_path = merged_dir / "merged_data.csv"
    merged.to_csv(target_path, index=False)
    return target_path

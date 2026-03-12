from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.data.merge import merge_cleaned_sources
from src.utils.config import load_config


def merge_data(
    config: Dict[str, Any] | None = None, cleaned_paths: Mapping[str, str] | Iterable[str] | None = None
) -> Path | None:
    """Merge cleaned files into a single dataset."""

    if config is None:
        config = load_config()

    processed_dir = config["data"]["paths"]["processed"]
    merge_keys = config["data"]["processing"]["merge"]["keys"]

    paths = []
    if cleaned_paths is None:
        clean_dir = Path(processed_dir) / "clean"
        paths = [str(path) for path in clean_dir.glob("*.csv")]
    elif isinstance(cleaned_paths, Mapping):
        paths = list(cleaned_paths.values())
    else:
        paths = list(cleaned_paths)

    if not paths:
        print("No cleaned files supplied for merging.")
        return None

    merged_path = merge_cleaned_sources(paths, merge_keys, processed_dir)
    print(f"Merged dataset saved to {merged_path}")
    return merged_path

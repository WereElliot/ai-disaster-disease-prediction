from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.data.clean import clean_source_file
from src.utils.config import load_config


def _resolve_raw_path(source: Mapping[str, Any], raw_dir: Path) -> Path | None:
    filename = source.get("output") or f\"{source.get('name', 'source')}.csv\"
    candidate = raw_dir / filename
    return candidate if candidate.exists() else None


def clean_data(config: Dict[str, Any] | None = None, raw_paths: Mapping[str, str] | None = None) -> Dict[str, str]:
    """Clean configured raw sources and persist the cleaned versions."""

    if config is None:
        config = load_config()

    raw_dir = Path(config["data"]["paths"]["raw"])
    processed_dir = config["data"]["paths"]["processed"]
    date_cols = config["data"]["processing"]["cleaning"]["date_columns"]
    cleaned = {}

    for source in config["data"]["sources"]:
        source_name = source.get("name", "unnamed")
        raw_path = (
            Path(raw_paths[source_name]) if raw_paths and source_name in raw_paths else _resolve_raw_path(source, raw_dir)
        )
        if raw_path is None:
            print(f\"Skipping clean for {source_name}: raw file not found.\")
            continue

        try:
            cleaned_path = clean_source_file(source_name, str(raw_path), processed_dir, date_columns=date_cols)
        except Exception as exc:
            print(f\"Failed to clean {source_name}: {exc}\")
            continue

        cleaned[source_name] = str(cleaned_path)
        print(f\"Cleaned data written for {source_name} -> {cleaned_path}\")

    return cleaned

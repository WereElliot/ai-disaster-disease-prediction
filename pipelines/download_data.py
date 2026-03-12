from __future__ import annotations

from typing import Any, Dict

import requests

from src.data.download import copy_local_sample, get_save_path, ensure_directory
from src.utils.config import load_config


def download_data(config: Dict[str, Any] | None = None) -> Dict[str, str]:
    """Download each configured source into the raw data directory."""

    if config is None:
        config = load_config()

    raw_dir = ensure_directory(config["data"]["paths"]["raw"])
    saved_paths: Dict[str, str] = {}

    for source in config["data"].get("sources", []):
        source_name = source.get("name", "unnamed")
        filename = source.get("output", f"{source_name}.csv")
        target_path = get_save_path(str(raw_dir), filename)

        try:
            response = requests.get(source["url"], params=source.get("params"), timeout=30)
            response.raise_for_status()
            with open(target_path, "wb") as fh:
                fh.write(response.content)
        except requests.RequestException as exc:
            local_sample = source.get("local_sample")
            if local_sample and copy_local_sample(local_sample, str(target_path)):
                print(f"Downloaded {source_name} using local sample {local_sample}")
                saved_paths[source_name] = str(target_path)
                continue
            print(f"Skipping {source_name}: download failed ({exc}) and no sample available")
            continue

        print(f"Downloaded {source_name} to {target_path}")
        saved_paths[source_name] = str(target_path)

    return saved_paths

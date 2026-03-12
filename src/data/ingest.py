from typing import Dict, Any

from src.utils.config import load_config
from pipelines.download_data import download_data


def ingest_data(config: Dict[str, Any] | None = None) -> Dict[str, str]:
    """Run the download pipeline and return the paths that were saved."""
    if config is None:
        config = load_config()

    return download_data(config)

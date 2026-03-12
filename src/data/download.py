from pathlib import Path
import shutil


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists and return its Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_save_path(directory: str, filename: str) -> Path:
    """Return the resolved path for saving a download given a directory and filename."""
    return ensure_directory(directory) / Path(filename).name


def copy_local_sample(sample_path: str, dest_path: str) -> bool:
    """Copy a local sample file to the destination; return True if copy happened."""
    sample = Path(sample_path)
    if not sample.exists():
        return False
    destination = Path(dest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample, destination)
    return True

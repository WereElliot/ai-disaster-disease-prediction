from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DATE_COLUMNS = ["date"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z_]+", "_", regex=True)
        .str.strip("_")
    )
    return cleaned


def _fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        median = df[col].median(skipna=True)
        replacement = 0 if pd.isna(median) else median
        df[col].fillna(replacement, inplace=True)
    return df


def _fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        mode = df[col].mode(dropna=True)
        fallback = "" if mode.empty else mode.iloc[0]
        df[col].fillna(fallback, inplace=True)
    return df


def clean_dataframe(df: pd.DataFrame, date_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Standardize, deduplicate, and impute missing values."""

    working = _normalize_columns(df)
    working = working.drop_duplicates()
    date_columns = date_columns or DEFAULT_DATE_COLUMNS
    for column in date_columns:
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce")
    working = _fill_numeric(working)
    working = _fill_categorical(working)
    return working


def clean_source_file(
    source_name: str,
    raw_path: str,
    processed_dir: str,
    date_columns: Iterable[str] | None = None,
) -> Path:
    """Read a raw file, clean it, and write to the processed clean subfolder."""

    source = Path(raw_path)
    if not source.exists():
        raise FileNotFoundError(f"Raw source not found: {raw_path}")

    df = pd.read_csv(source)
    cleaned = clean_dataframe(df, date_columns=date_columns)
    clean_dir = Path(processed_dir) / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    target_path = clean_dir / f"{source_name}.csv"
    cleaned.to_csv(target_path, index=False)
    return target_path

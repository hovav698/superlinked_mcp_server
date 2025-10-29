"""
File processing utilities for loading and preparing data files.
Handles CSV/JSON loading, column-oriented JSON transformation, and DataFrame normalization.
"""
from pathlib import Path
import pandas as pd
from typing import Dict, Optional
import json
import logging
from config import WORK_DIR


def resolve_file_path(file_path: str) -> Path:
    """Resolve file path to absolute path relative to WORK_DIR."""
    path = Path(file_path)
    return path if path.is_absolute() else WORK_DIR / path


def _is_column_oriented_json(data) -> bool:
    """Check if JSON is column-oriented format: {"col": {"id": "val"}}."""
    if not isinstance(data, dict):
        return False
    return all(isinstance(v, dict) for v in data.values())


def _transform_column_oriented_json(data: dict) -> pd.DataFrame:
    """Transform column-oriented JSON to row-oriented DataFrame."""
    first_column = next(iter(data.values()))
    row_ids = list(first_column.keys())

    rows = []
    for row_id in row_ids:
        row = {'id': row_id}
        for column_name, column_data in data.items():
            row[column_name] = column_data.get(row_id, None)
        rows.append(row)

    logging.info(f"Transformed column-oriented JSON: {len(rows)} records")
    return pd.DataFrame(rows)


def _load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(file_path)


def _load_json(file_path: str) -> pd.DataFrame:
    """Load JSON file, auto-detecting and handling column-oriented format."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if _is_column_oriented_json(data):
        return _transform_column_oriented_json(data)
    else:
        return pd.read_json(file_path)


def load_file(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV or JSON file. Returns raw DataFrame without normalization."""
    if not file_path:
        return None

    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.csv':
            return _load_csv(file_path)
        elif file_ext == '.json':
            return _load_json(file_path)
        else:
            logging.warning(f"Unsupported file type: {file_ext}")
            return None

    except Exception as e:
        logging.error(f"Could not load file {file_path}: {e}")
        return None


def prepare_dataframe(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Prepare DataFrame for Superlinked ingestion.
    Ensures id column exists, normalizes types, filters to schema columns.
    Returns a normalized DataFrame ready for ingestion.
    """
    df = df.copy()

    # Ensure id column exists
    if 'id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'id'})

    # Convert datetimes to Unix timestamps
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype('int64') // 10**9

    # Filter to schema columns only
    schema_columns = ['id'] + list(column_mapping.keys())
    df_prepared = df[schema_columns].copy()

    # Ensure id is string
    df_prepared['id'] = df_prepared['id'].astype(str)

    # Ensure recency columns are integers
    for col, space_type in column_mapping.items():
        if space_type == 'recency' and col in df_prepared.columns:
            df_prepared[col] = df_prepared[col].astype(int)

    return df_prepared


def extract_categories_from_file(file_path: Optional[str], column_mapping: Dict[str, str]) -> Dict[str, list]:
    """Extract unique values from categorical columns. Returns empty dict if no categories found."""
    if not file_path:
        return {}

    category_columns = [col for col, space_type in column_mapping.items() if space_type == "category"]
    if not category_columns:
        return {}

    df = load_file(file_path)
    if df is None:
        return {}

    categories_dict = {}
    for col in category_columns:
        if col in df.columns:
            unique_values = df[col].dropna().unique().tolist()
            categories_dict[col] = [str(val) for val in unique_values]
        else:
            logging.warning(f"Category column '{col}' not found in file")

    return categories_dict

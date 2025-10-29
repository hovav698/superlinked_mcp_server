"""
Utility functions for file loading and data processing.
"""
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, List
import logging
from config import WORK_DIR


def resolve_file_path(file_path: str) -> Path:
    """
    Resolve file path to absolute path.
    If relative path is provided, resolve it relative to WORK_DIR.

    Args:
        file_path: Absolute or relative file path

    Returns:
        Absolute Path object
    """
    path = Path(file_path)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # Otherwise, resolve relative to WORK_DIR
    return WORK_DIR / path


def load_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data file and return as DataFrame.
    Supports CSV and JSON formats.
    Ensures 'id' column exists (required by Superlinked).
    """
    if not file_path:
        return None

    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            logging.warning(f"Unsupported file type: {file_ext}")
            return None

        # Ensure 'id' column exists (required by Superlinked)
        if 'id' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'id'})

        # Convert id to string type (required by Superlinked)
        df['id'] = df['id'].astype(str)

        return df
    except Exception as e:
        logging.error(f"Could not load file {file_path}: {e}")
        return None


def extract_categories_from_file(file_path: Optional[str], column_mapping: Dict[str, str]) -> Dict[str, list]:
    """
    Extract unique categories from categorical columns in the data file.

    Args:
        file_path: Path to the data file
        column_mapping: Mapping of column names to space types

    Returns:
        Dictionary mapping column names to their list of unique categories.
        Returns empty dict if no category columns or file cannot be loaded.
    """
    if not file_path:
        return {}

    # Get list of category columns
    category_columns = [col for col, space_type in column_mapping.items() if space_type == "category"]
    if not category_columns:
        return {}

    # Load file
    df = load_file(file_path)
    if df is None:
        return {}

    # Extract categories for each category column
    categories_dict = {}
    for col in category_columns:
        if col in df.columns:
            categories = df[col].dropna().unique().tolist()
            categories_dict[col] = [str(cat) for cat in categories]
        else:
            logging.warning(f"Category column '{col}' not found in file")

    return categories_dict


def prepare_dataframe_for_ingestion(df: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict]:
    """
    Prepare DataFrame for Superlinked ingestion.

    - Filters to only include schema columns (id + mapped columns)
    - Converts recency columns to integers (Unix timestamps)
    - Converts DataFrame to list of dictionaries

    Args:
        df: Source DataFrame with 'id' column
        column_mapping: Mapping of column names to space types

    Returns:
        List of dictionaries ready for source.put()
    """
    # Filter to only schema columns (id + mapped columns)
    schema_columns = ['id'] + list(column_mapping.keys())
    df_filtered = df[schema_columns].copy()

    # Convert recency columns to int (Unix timestamps)
    for col, space_type in column_mapping.items():
        if space_type == 'recency' and col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].astype(int)

    # Convert to list of dictionaries
    records = df_filtered.to_dict('records')

    return records

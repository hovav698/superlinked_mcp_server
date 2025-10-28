"""
Utility functions for file loading and data processing.
"""
from pathlib import Path
import pandas as pd
from typing import Dict, Optional
import logging


def load_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data file and return as DataFrame.
    Supports CSV and JSON formats.
    """
    if not file_path:
        return None

    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        else:
            logging.warning(f"Unsupported file type: {file_ext}")
            return None
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

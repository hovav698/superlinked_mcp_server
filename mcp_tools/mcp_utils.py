"""
Helper functions for Superlinked MCP Server with InMemoryExecutor.
"""
import pandas as pd
from typing import Dict


def detect_column_type(df: pd.DataFrame, col: str) -> str:
    """Auto-detect column type from pandas dtype."""
    dtype = str(df[col].dtype)

    # Check for timestamp
    if 'datetime' in dtype:
        return 'recency'

    # Check for numeric
    if 'float' in dtype or 'int' in dtype:
        # If it looks like a score (0-1 range), treat as number space
        if df[col].between(0, 1).all():
            return 'number'
        return 'number'

    # Check for categorical (string with low cardinality)
    if dtype == 'object' or 'str' in dtype:
        unique_ratio = len(df[col].dropna().unique()) / len(df[col].dropna())
        # If less than 10% unique values, likely categorical
        if unique_ratio < 0.1:
            return 'category'

    # Default to text
    return 'text_similarity'

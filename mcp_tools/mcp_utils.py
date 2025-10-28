"""
Helper functions for Superlinked MCP Server with InMemory VectorDB.
"""
import pandas as pd
import subprocess
import time
import requests
import sys
import os
from pathlib import Path
from typing import Dict
import json


# Import config from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


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


def start_superlinked_server(metadata: Dict):
    """
    Start the Superlinked server with index metadata.

    Args:
        metadata: Index metadata dict to pass to server via environment variable.
                  Format: {"index_name": {"csv_filename": "...", "column_mapping": {...}}}
    """

    # Kill existing server
    subprocess.run(["pkill", "-9", "-f", "superlinked.server"], capture_output=True)
    time.sleep(2)

    # Start new server
    env = os.environ.copy()
    env["APP_MODULE_PATH"] = "app"

    # Pass metadata as environment variable
    env["INDEX_METADATA"] = json.dumps(metadata)

    # Redirect output to file for debugging
    log_file_path = str(WORK_DIR / "server_startup.log")
    f = open(log_file_path, 'w')
    subprocess.Popen(
        [sys.executable, "-m", "superlinked.server"],
        env=env,
        cwd=str(WORK_DIR),
        stdout=f,
        stderr=f
    )
    # Don't close the file - let the subprocess keep it open



def ingest_csv_data(
    csv_path: str,
    index_name: str,
    column_mapping: Dict[str, str]
) -> Dict[str, int]:
    """
    Ingest CSV data via REST API.

    Args:
        csv_path: Path to CSV file
        index_name: Name of the index
        column_mapping: Mapping of columns to space types

    Returns:
        Dict with 'ingested' and 'errors' counts
    """
    df = pd.read_csv(csv_path)
    ingest_url = f"http://localhost:{SERVER_PORT}/api/v1/ingest/{index_name}_schema"

    ingested = 0
    errors = 0

    for _, row in df.iterrows():
        try:
            # Only include 'id' and columns in the mapping
            row_dict = {'id': str(row['id'])}

            for col in column_mapping.keys():
                if col in row.index:
                    value = row[col]
                    # Skip NaN values
                    if pd.isna(value):
                        continue
                    # Convert timestamps to int
                    if column_mapping[col] == 'recency':
                        row_dict[col] = int(value)
                    else:
                        row_dict[col] = value

            # Use DEBUG_TIMEOUT for debugging with breakpoints
            from config import DEBUG_MODE, DEBUG_TIMEOUT
            timeout_val = DEBUG_TIMEOUT if DEBUG_MODE else 5
            response = requests.post(ingest_url, json=row_dict, timeout=timeout_val)
            if response.status_code in [200, 202]:
                ingested += 1
            else:
                errors += 1
                print(f"  Error ingesting row {row_dict.get('id')}: {response.status_code}")
        except Exception as e:
            errors += 1
            print(f"  Error ingesting row: {e}")

    return {"ingested": ingested, "errors": errors}

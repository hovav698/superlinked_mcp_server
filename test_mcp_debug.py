"""
Debug script for MCP server.
Run this with VSCode debugger to step through MCP server code.
"""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "mcp_tools"))

# Import MCP server functions
from config import *
from mcp_tools.mcp_utils import (
    start_superlinked_server,
    ingest_csv_data
)
import pandas as pd
import json


def test_preview_csv():
    """Test CSV preview."""
    print("\n=== Testing CSV Preview ===")
    csv_path = "business_news.csv"

    df = pd.read_csv(csv_path)
    preview = {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(3).to_dict('records')
    }
    print(json.dumps(preview, indent=2))


def test_create_index():
    """Test index creation - SET BREAKPOINT HERE."""
    print("\n=== Testing Create Index ===")

    csv_path = "business_news.csv"
    column_mapping = {
        "headline": "text_similarity",
        "category": "category",
        "date": "recency"
    }

    # Set breakpoint on next line to debug
    index_name = Path(csv_path).stem
    df = pd.read_csv(csv_path)

    print(f"Index name: {index_name}")
    print(f"Rows: {len(df)}")
    print(f"Column mapping: {column_mapping}")

    # Prepare metadata to pass to server
    metadata = {
        index_name: {
            "csv_filename": Path(csv_path).name,
            "column_mapping": column_mapping
        }
    }

    # Start server with metadata
    print("Starting server with metadata...")
    start_superlinked_server(metadata)


    print("✓ Server started")

    # Wait for index initialization
    print("Waiting for index initialization...")
    import time
    time.sleep(3)

    # Manually ingest data - SET BREAKPOINT HERE to step into ingestion
    print(f"Ingesting {len(df)} rows...")
    result = ingest_csv_data(csv_path, index_name, column_mapping)

    print(f"✓ Ingested: {result['ingested']} rows")
    print(f"✗ Errors: {result['errors']} rows")

    return {
        "status": "success",
        "index_name": index_name,
        "columns": column_mapping,
        "rows": len(df),
        "ingested": result["ingested"],
        "errors": result["errors"]
    }


def test_query_index():
    """Test querying index - SET BREAKPOINT HERE."""
    print("\n=== Testing Query Index ===")

    import requests

    index_name = "business_news"
    query_text = "Elon Musk"
    limit = 3

    # Check server (long timeout for debugging with breakpoints)
    try:
        timeout_val = DEBUG_TIMEOUT if DEBUG_MODE else 2
        r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=timeout_val)
        if r.status_code == 200:
            print("✓ Server is running")
        else:
            print("❌ Server not responding")
            return
    except Exception as e:
        print(f"❌ Server error: {e}")
        return

    # Query - SET BREAKPOINT HERE
    url = f"http://localhost:{SERVER_PORT}/api/v1/search/{index_name}_query"
    payload = {"search_query": query_text, "limit": limit}

    print(f"Querying: {url}")
    print(f"Payload: {payload}")

    timeout_val = DEBUG_TIMEOUT if DEBUG_MODE else 30
    response = requests.post(url, json=payload, timeout=timeout_val)
    response.raise_for_status()

    results = response.json()
    entries = results.get('entries', [])

    parsed = []
    for entry in entries:
        parsed.append({
            "id": entry.get("id"),
            "score": entry.get("metadata", {}).get("score", 0),
            "fields": entry.get("fields", {})
        })

    print(json.dumps(parsed, indent=2))
    return parsed


if __name__ == "__main__":
    print("=" * 60)
    print("MCP Server Debug Script")
    print("=" * 60)
    print("\nSet breakpoints in the test functions or in mcp_utils.py")
    print("Then step through with F10 (step over) or F11 (step into)")
    print("=" * 60)

    # Uncomment the test you want to run:

    # Test 1: Preview CSV
    # test_preview_csv()

    # Test 2: Create Index (most common)
    result = test_create_index()
    print(f"\nResult: {json.dumps(result, indent=2)}")

    # Test 3: Query Index (after creating index)
    # test_query_index()

    print("\n" + "=" * 60)
    print("Debug session complete!")

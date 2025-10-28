"""
Superlinked MCP Server - Clean and organized.
Uses InMemory VectorDB with dynamic index creation.
"""
from fastmcp import FastMCP
import pandas as pd
from pathlib import Path
import json
import sys
import time

# Import config and utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from mcp_utils import (
    start_superlinked_server,
    wait_for_server,
    ingest_csv_data
)

mcp = FastMCP("Superlinked RAG Server")


@mcp.tool()
def preview_csv(csv_path: str, rows: int = 5) -> str:
    """
    Preview CSV file contents.

    Args:
        csv_path: Path to CSV file
        rows: Number of rows to preview (default: 5)
    """
    try:
        df = pd.read_csv(csv_path)
        preview = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(rows).to_dict('records')
        }
        return json.dumps(preview, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def create_index(csv_path: str, column_mapping: dict) -> str:
    """
    Create index from CSV with manual data ingestion.
    Space types: 'text_similarity', 'recency', 'number', 'category'

    Args:
        csv_path: Path to CSV file
        column_mapping: Which columns to index and their types
    """
    try:
        index_name = Path(csv_path).stem
        df = pd.read_csv(csv_path)

        final_mapping = {}
        for col, space_type in column_mapping.items():
            if col not in df.columns:
                return json.dumps({"error": f"Column '{col}' not found in CSV"})
            final_mapping[col] = space_type

        # Prepare metadata to pass to server via environment variable
        metadata = {
            index_name: {
                "csv_filename": Path(csv_path).name,
                "column_mapping": final_mapping
            }
        }

        # Start server and pass metadata directly
        print(f"Starting server for {index_name}...")
        start_superlinked_server(metadata)

        # Wait for server startup
        if not wait_for_server():
            return json.dumps({"error": "Server failed to start"})

        # Wait for index initialization
        print(f"Waiting for index initialization...")
        time.sleep(3)

        # Manually ingest data via REST API
        print(f"Ingesting {len(df)} rows...")
        result = ingest_csv_data(csv_path, index_name, final_mapping)

        return json.dumps({
            "status": "success",
            "index_name": index_name,
            "columns": final_mapping,
            "rows": len(df),
            "ingested": result["ingested"],
            "errors": result["errors"]
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_indexes() -> str:
    """List all available indexes. Note: Indexes are not persisted - created on demand."""
    try:
        import requests

        # Check if server is running
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
            if r.status_code == 200:
                return json.dumps({
                    "status": "Server running",
                    "message": "Indexes are created on-demand and not persisted. Use create_index to index data."
                }, indent=2)
        except:
            pass

        return json.dumps({
            "status": "Server not running",
            "message": "Use create_index to start server and index data."
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def query_index(index_name: str, query_text: str, limit: int = 5) -> str:
    """
    Query an index with natural language.

    Args:
        index_name: Name of the index to query
        query_text: Natural language query
        limit: Max results (default: 5)
    """
    try:
        import requests

        # Check if server is running
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
            if r.status_code != 200:
                return json.dumps({"error": "Server not running. Create an index first using create_index."})
        except:
            return json.dumps({"error": "Server not running. Create an index first using create_index."})

        # Query
        url = f"http://localhost:{SERVER_PORT}/api/v1/search/{index_name}_query"
        payload = {"search_query": query_text, "limit": limit}

        response = requests.post(url, json=payload, timeout=30)
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

        return json.dumps(parsed, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")

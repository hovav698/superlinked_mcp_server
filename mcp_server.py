"""
Simplified Superlinked MCP Server
All functionality inline - no separate utils needed.
"""
from fastmcp import FastMCP
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
import subprocess
import time
import requests
import sys
import os
import uuid

from config import *
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

mcp = FastMCP("Superlinked RAG Server")


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


def get_qdrant_client():
    """Get Qdrant client instance."""
    return QdrantClient(url=QDRANT_URL)


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
def create_index(csv_path: str, column_mapping: Dict[str, str], recreate: bool = False) -> str:
    """
    Create index from CSV. Auto-detects types if not specified.
    Space types: 'text_similarity', 'recency', 'number', 'category'

    Args:
        csv_path: Path to CSV file
        column_mapping: Which columns to index and their types
        recreate: If True, delete and recreate the index if it exists (default: False)
    """
    try:
        index_name = Path(csv_path).stem
        was_recreated = False
        client = get_qdrant_client()

        # Check if metadata for this index exists
        metadata_collection = "_sl_metadata"
        collections = [c.name for c in client.get_collections().collections]

        # Generate consistent UUID from index name
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, index_name))

        metadata_exists = False
        if metadata_collection in collections:
            try:
                existing = client.retrieve(metadata_collection, ids=[point_uuid])
                metadata_exists = len(existing) > 0
            except:
                pass

        if metadata_exists:
            if recreate:
                # Delete metadata and data collection if it exists
                client.delete(
                    collection_name=metadata_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="index_name",
                                match=MatchValue(value=index_name)
                            )
                        ]
                    )
                )
                if index_name in collections:
                    client.delete_collection(index_name)
                print(f"Deleted existing index: {index_name}")
                was_recreated = True
            else:
                return json.dumps({
                    "status": "exists",
                    "message": f"Index '{index_name}' already exists. Use recreate=True to overwrite."
                })

        # Load CSV to get schema info
        df = pd.read_csv(csv_path)

        # Auto-detect types if "auto"
        final_mapping = {}
        for col, space_type in column_mapping.items():
            if col not in df.columns:
                return json.dumps({"error": f"Column '{col}' not found in CSV"})

            if space_type == "auto":
                final_mapping[col] = detect_column_type(df, col)
            else:
                final_mapping[col] = space_type

        # Store metadata in a separate collection
        # Don't create the data collection - let Superlinked handle that
        metadata_collection = "_sl_metadata"

        # Create metadata collection if it doesn't exist
        collections = [c.name for c in client.get_collections().collections]
        if metadata_collection not in collections:
            client.create_collection(
                collection_name=metadata_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)
            )

        # Store metadata with index name as the ID
        metadata = {
            "_is_metadata": True,
            "index_name": index_name,
            "csv_filename": Path(csv_path).name,
            "column_mapping": final_mapping,
            "total_rows": len(df)
        }

        client.upsert(
            collection_name=metadata_collection,
            points=[PointStruct(
                id=point_uuid,  # Use UUID derived from index name
                vector=[0.0],  # Dummy vector
                payload=metadata
            )]
        )

        # Start server to load data
        print(f"Starting server to load {index_name}...")
        subprocess.run(["pkill", "-9", "-f", "superlinked.server"], capture_output=True)
        time.sleep(2)

        env = os.environ.copy()
        env["APP_MODULE_PATH"] = "app"

        subprocess.Popen(
            [sys.executable, "-m", "superlinked.server"],
            env=env,
            cwd=str(WORK_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server
        for _ in range(SERVER_STARTUP_TIMEOUT):
            try:
                r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
                if r.status_code == 200:
                    break
            except:
                time.sleep(1)

        # Trigger data load
        time.sleep(DATA_LOAD_WAIT)

        return json.dumps({
            "status": "success",
            "index_name": index_name,
            "columns": final_mapping,
            "rows": len(df),
            "recreated": was_recreated
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_indexes() -> str:
    """List all available indexes (Qdrant collections)."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections

        result = []
        for c in collections:
            # Get full collection info
            collection_info = client.get_collection(c.name)
            info = {
                "name": c.name,
                "vectors_count": collection_info.vectors_count or 0,
                "points_count": collection_info.points_count or 0
            }
            result.append(info)

        return json.dumps(result, indent=2)
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
        # Check server is running
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
            if r.status_code != 200:
                raise Exception("Server not running")
        except:
            # Start server
            env = os.environ.copy()
            env["APP_MODULE_PATH"] = "app"

            subprocess.Popen(
                [sys.executable, "-m", "superlinked.server"],
                env=env,
                cwd=str(WORK_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for startup
            for _ in range(SERVER_STARTUP_TIMEOUT):
                try:
                    r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
                    if r.status_code == 200:
                        break
                except:
                    time.sleep(1)

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

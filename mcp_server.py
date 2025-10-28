"""
Superlinked MCP Server - InMemoryExecutor version.
Uses InMemoryExecutor with direct Python API (no REST server).
"""
from fastmcp import FastMCP
import pandas as pd
from pathlib import Path
import json
import sys

# Import config and app
sys.path.insert(0, str(Path(__file__).parent))
from config import *
from app import create_app
from utils import load_file
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

mcp = FastMCP("Superlinked RAG Server")

# Store active app instances
_active_apps = {}


@mcp.tool()
def preview_file(file_path: str, rows: int = 5) -> str:
    """
    Preview data file contents to understand its structure before creating an index.

    Use this tool to:
    - Explore the columns and data types in a data file (CSV or JSON)
    - See sample data to help decide which columns to index
    - Verify the file is properly formatted before indexing
    - Determine total row count for indexing estimates

    This should be your first step before creating an index with create_index().

    Args:
        file_path: Path to data file - supports CSV and JSON (absolute or relative to working directory)
        rows: Number of sample rows to preview (default: 5)

    Returns:
        JSON with: total rows, column names, data types, and sample records
    """
    try:
        df = load_file(file_path)
        if df is None:
            return json.dumps({"error": f"Could not load file or unsupported format: {file_path}"})

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
def create_index(file_path: str, column_mapping: dict) -> str:
    """
    Create a semantic search index from structured data using Superlinked's InMemoryExecutor.

    Use this tool to:
    - Build a searchable vector index from structured data (CSV or JSON)
    - Combine multiple search dimensions (text similarity, recency, categories, numbers)

    When to use:
    - After previewing the file with preview_file() to understand the data structure
    - When you want to enable semantic/vector search over text fields
    - When you need to filter or rank by dates, categories, or numeric values

    Available space types for column_mapping:
    - 'text_similarity': For text fields (uses embeddings for semantic search)
    - 'recency': For timestamp/date fields (Unix timestamps, more recent = higher rank)
    - 'number': For numeric fields (prices, ratings, counts, etc.)
    - 'category': For categorical fields (tags, types, labels, etc.)

    The index is stored in memory and can be queried with query_index().

    Args:
        file_path: Path to data file to index (CSV or JSON)
        column_mapping: Dict mapping column names to space types
                       Example: {"description": "text_similarity", "timestamp": "recency"}

    Returns:
        JSON with: status, index name, column configuration, ingestion statistics
    """
    try:
        index_name = Path(file_path).stem
        df = load_file(file_path)

        if df is None:
            return json.dumps({"error": f"Could not load file or unsupported format: {file_path}"})

        # Validate columns
        final_mapping = {}
        for col, space_type in column_mapping.items():
            if col not in df.columns:
                return json.dumps({"error": f"Column '{col}' not found in file"})
            final_mapping[col] = space_type

        # Create app with InMemoryExecutor
        logging.info(f"Creating InMemoryExecutor app for {index_name}...")
        app, source, query = create_app(file_path, final_mapping)

        # Ingest data via InMemorySource
        logging.info(f"Ingesting {len(df)} rows...")
        ingested = 0
        errors = 0

        for _, row in df.iterrows():
            try:
                # Build row dict with id + mapped columns
                row_dict = {'id': str(row.get('id', row.name))}

                for col in final_mapping.keys():
                    if col in row.index and not pd.isna(row[col]):
                        value = row[col]
                        # Convert timestamps to int if needed
                        if final_mapping[col] == 'recency':
                            row_dict[col] = int(value)
                        else:
                            row_dict[col] = value

                # Use source.put() to ingest
                source.put([row_dict])
                ingested += 1
            except Exception as e:
                errors += 1
                logging.error(f"Error ingesting row: {e}")

        # Store app instance for querying
        _active_apps[index_name] = {
            'app': app,
            'query': query,
            'source': source
        }

        return json.dumps({
            "status": "success",
            "index_name": index_name,
            "columns": final_mapping,
            "rows": len(df),
            "ingested": ingested,
            "errors": errors
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})



@mcp.tool()
def query_index(index_name: str, query_text: str, limit: int = 5) -> str:
    """
    Query a Superlinked index using natural language semantic search.

    Use this tool to:
    - Search indexed data using natural language queries
    - Find semantically similar content (not just keyword matching)
    - Retrieve ranked results based on vector similarity and configured spaces

    When to use:
    - After creating an index with create_index()
    - When you need semantic search (e.g., "find articles about AI" matches "machine learning posts")
    - When you want results ranked by multiple factors (similarity + recency + category)

    Args:
        index_name: Name of the index to query (from create_index(), typically the filename without extension)
        query_text: Natural language search query
        limit: Maximum number of results to return (default: 5)

    Returns:
        JSON array of results with: id, similarity score, and all indexed fields
    """
    try:
        # Check if index exists
        if index_name not in _active_apps:
            return json.dumps({
                "error": f"Index '{index_name}' not found. Create it first using create_index.",
                "available_indexes": list(_active_apps.keys())
            })

        # Get app and query
        app_data = _active_apps[index_name]
        app = app_data['app']
        query = app_data['query']

        # Execute query directly via Python API
        result = app.query(query, search_query=query_text, limit=limit)

        # Parse QueryResult - it's iterable with ('entries', [ResultEntry, ...])
        parsed = []
        for key, value in result:
            if key == 'entries':
                for entry in value:
                    parsed.append({
                        "id": entry.id,
                        "score": entry.metadata.score,
                        "fields": entry.fields
                    })
                break

        return json.dumps(parsed, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")

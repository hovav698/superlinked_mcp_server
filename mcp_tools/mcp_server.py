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
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from app import create_app

# Helper to print to stderr (not stdout, which breaks MCP protocol)
def log(msg):
    print(msg, file=sys.stderr)

mcp = FastMCP("Superlinked RAG Server")

# Store active app instances
_active_apps = {}


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
    Create index from CSV using InMemoryExecutor.
    Space types: 'text_similarity', 'recency', 'number', 'category'

    Args:
        csv_path: Path to CSV file
        column_mapping: Which columns to index and their types
    """
    try:
        index_name = Path(csv_path).stem
        df = pd.read_csv(csv_path)

        # Validate columns
        final_mapping = {}
        for col, space_type in column_mapping.items():
            if col not in df.columns:
                return json.dumps({"error": f"Column '{col}' not found in CSV"})
            final_mapping[col] = space_type

        # Create app with InMemoryExecutor
        log(f"Creating InMemoryExecutor app for {index_name}...")
        app, source, query = create_app(index_name, final_mapping, csv_path)

        # Ingest data via InMemorySource
        log(f"Ingesting {len(df)} rows...")
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
                log(f"  Error ingesting row: {e}")

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
    Query an index with natural language.

    Args:
        index_name: Name of the index to query
        query_text: Natural language query
        limit: Max results (default: 5)
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

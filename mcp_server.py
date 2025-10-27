"""
Superlinked MCP Server
Provides tools for creating and querying RAG indexes using Superlinked and Qdrant.
"""
from fastmcp import FastMCP
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import json

from superlinked_utils import (
    list_indexes,
    create_index_and_load,
    query_index
)

# Initialize FastMCP server
mcp = FastMCP("Superlinked RAG Server")


@mcp.tool()
def preview_csv(csv_path: str) -> str:
    """
    Preview the first few rows of a CSV file to understand its structure and columns.

    ⚠️ CRITICAL: If list_available_indexes returns NO index for this CSV file, you MUST call this tool BEFORE any other
    Claude Desktop internal tool (like Read, Glob, etc.). Do NOT use Read or any other built-in tool to inspect the CSV.

    This MCP tool is specifically designed to analyze CSV structure for indexing and MUST be used instead of
    Claude Desktop's internal file reading tools when preparing to create an index.

    This tool is REQUIRED before calling create_index to determine:
    - Which columns exist in the CSV
    - The data type of each column (text, timestamp, number)
    - Sample values to understand the data format
    - Appropriate column-to-space type mappings for indexing

    MANDATORY USAGE PATTERN:
    1. User asks about indexing/querying a CSV file
    2. Call list_available_indexes FIRST to check if index exists
    3. ⚠️ If index does NOT exist, call preview_csv IMMEDIATELY - do NOT use Read, Glob, or other tools
    4. Use the column information from preview_csv to decide column_mapping for create_index
    5. Then call create_index with the correct mappings

    This tool shows the first 10 rows, column names, data types, and sample values.

    Args:
        csv_path: Absolute path to the CSV file to preview

    Returns:
        JSON string containing: first 10 rows as dict, column names, data types, and sample values

    Example:
        preview_csv("/path/to/data.csv")
    """
    try:
        # Validate path
        path = Path(csv_path)
        if not path.exists():
            return json.dumps({
                "error": f"File not found: {csv_path}",
                "suggestion": "Please provide the full absolute path to the CSV file"
            })

        if not path.suffix.lower() == '.csv':
            return json.dumps({
                "error": f"File is not a CSV: {csv_path}",
                "suggestion": "Only CSV files are supported"
            })

        # Read CSV
        df = pd.read_csv(csv_path)

        # Get column info
        column_info = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_value = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            column_info[col] = {
                "dtype": dtype,
                "sample": sample_value
            }

        # Prepare response
        result = {
            "filename": path.name,
            "total_rows": len(df),
            "columns": list(df.columns),
            "column_info": column_info,
            "preview": df.head(10).to_dict(orient='records')
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__
        })


@mcp.tool()
def create_index(
    csv_path: str,
    column_mapping: Dict[str, str],
    weights: Optional[Union[Dict[str, float], str]] = None,
    recreate: bool = False
) -> str:
    """
    Create a Superlinked index for a CSV file with specified column-to-space mappings.

    IMPORTANT: Only call this tool AFTER:
    1. list_available_indexes confirms the index does NOT exist (or you want to recreate it)
    2. preview_csv has been called to inspect the CSV structure

    This tool creates a vector index in Qdrant using Superlinked's multi-space indexing.
    The index name will be the same as the CSV filename (without extension).

    If the index already exists and recreate=False, this tool will return an error message.
    If recreate=True, the existing index will be deleted and recreated with new settings.

    ⚠️ CRITICAL: Space Types - ONLY these values are allowed:
    The column_mapping values MUST be one of these three exact strings (case-sensitive):
    - "text_similarity": For text fields that should be semantically searchable
    - "recency": For timestamp fields to favor recent documents
    - "number": For numerical fields to optimize by value (e.g., usefulness scores)

    NO OTHER VALUES ARE ALLOWED. Using any other value will result in an error.

    Args:
        csv_path: Absolute path to the CSV file
        column_mapping: Dictionary mapping column names to space types.
                       ⚠️ VALUES MUST BE EXACTLY: "text_similarity", "recency", or "number" (no other values allowed)
                       Example: {"body": "text_similarity", "created_at": "recency", "usefulness": "number"}
        weights: Optional dictionary of weights for each column (default: 1.0 for text, 0.5 for others).
                Example: {"body": 1.0, "created_at": 0.5, "usefulness": 0.5}
        recreate: Optional boolean to force recreation of existing index (default: False).
                 If True, deletes existing index and creates new one with updated mappings/weights.
                 Use this when you want to:
                 - Change column mappings
                 - Update weights
                 - Reload data with different settings
                 Example: recreate=True

    Returns:
        JSON string with status, index name, and document count

    Example:
        create_index(
            "/path/to/data.csv",
            {"body": "text_similarity", "created_at": "recency"},
            {"body": 1.0, "created_at": 0.5},
            recreate=False
        )
    """
    # Handle weights being passed as JSON string from MCP
    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid weights format. Expected dictionary, got malformed JSON string: {weights}",
                "json_error": str(e)
            })

    try:
        # Validate path
        path = Path(csv_path)
        if not path.exists():
            return json.dumps({
                "error": f"File not found: {csv_path}"
            })

        # Validate column_mapping
        valid_types = ["text_similarity", "recency", "number"]
        for col, space_type in column_mapping.items():
            if space_type not in valid_types:
                return json.dumps({
                    "error": f"Invalid space type '{space_type}' for column '{col}'",
                    "valid_types": valid_types
                })

        # Check if at least one text_similarity space exists
        if "text_similarity" not in column_mapping.values():
            return json.dumps({
                "error": "At least one column must be mapped to 'text_similarity'",
                "suggestion": "Add a text column with type 'text_similarity'"
            })

        # Create index (with optional recreation)
        result = create_index_and_load(csv_path, column_mapping, weights, recreate)

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__
        })


@mcp.tool()
def list_available_indexes() -> str:
    """
    ⚠️ CRITICAL: ALWAYS call this tool FIRST in Claude Desktop if there is ANY question about a specific index or file.

    This tool MUST be called BEFORE any other tool (including Claude Desktop's internal tools like Read, Glob, etc.)
    when the user asks about indexing or querying any file.

    List all existing indexes that have been created.

    MANDATORY WORKFLOW - CALL THIS FIRST:
    1. User asks ANY question about indexing/querying a CSV file or mentions a specific file
    2. ⚠️ Call list_available_indexes FIRST - do NOT use any other tools yet
    3. If index exists (CSV filename matches an index name), skip directly to query_indexed_data
    4. If index does NOT exist, then call preview_csv to inspect the CSV structure
    5. Then call create_index to create the new index

    The index name matches the CSV filename without extension.
    Example: "sample_data.csv" creates index named "sample_data"

    Returns:
        JSON string with list of index names

    Example:
        list_available_indexes()  # Returns: {"indexes": ["sample_data", "employees", "products"]}
    """
    try:
        indexes = list_indexes()

        return json.dumps({
            "indexes": indexes,
            "count": len(indexes)
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__
        })


@mcp.tool()
def query_indexed_data(
    index_name: str,
    query: str,
    limit: int = 5,
    weights: Optional[Union[Dict[str, float], str]] = None
) -> str:
    """
    Query an existing index with natural language to retrieve relevant documents.

    IMPORTANT: Before calling this tool, use list_available_indexes to confirm the index exists.
    If the index doesn't exist, you'll need to create it first.

    This tool searches the specified index using semantic similarity and returns
    the most relevant documents with their scores and content.

    The index_name should match the CSV filename (without extension).
    Example: For "sample_data.csv", use index_name="sample_data"

    Args:
        index_name: Name of the index to query (same as CSV filename without extension)
        query: Natural language query string
        limit: Maximum number of results to return (default: 5)
        weights: Optional dictionary to override default space weights.
                Example: {"body": 1.5, "created_at": 0.3}

    Returns:
        JSON string with array of results containing scores and document fields

    Example:
        query_indexed_data(
            "employees",
            "vacation policy",
            limit=3
        )
    """
    # Handle weights being passed as JSON string from MCP
    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid weights format. Expected dictionary, got malformed JSON string: {weights}",
                "json_error": str(e)
            })

    try:
        # Execute query (query_index will validate index exists)
        results = query_index(index_name, query, limit, weights)

        return json.dumps({
            "index_name": index_name,
            "query": query,
            "limit": limit,
            "results": results,
            "result_count": len(results)
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "type": type(e).__name__
        })


if __name__ == "__main__":
    # Run the MCP server with stdio transport for Claude Desktop
    mcp.run(transport="stdio")

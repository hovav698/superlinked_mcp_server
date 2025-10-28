"""
Dynamic Superlinked app with InMemory VectorDB.
Reads index config from environment variable (passed directly from MCP tool).
"""
from superlinked import framework as sl
from datetime import timedelta
from pathlib import Path
import json
import os
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from config import *


def load_csv_if_needed(csv_filename: Optional[str], column_mapping: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Load CSV file if it's needed for category detection."""
    if not csv_filename:
        return None

    # Only load CSV if we have category spaces
    has_category = any(space_type == "category" for space_type in column_mapping.values())
    if not has_category:
        return None

    try:
        return pd.read_csv(csv_filename)
    except Exception as e:
        print(f"⚠ Could not load CSV {csv_filename}: {e}")
        return None


def build_schema_fields(column_mapping: Dict[str, str]) -> Dict[str, type]:
    """Build schema field definitions from column mapping."""
    schema_fields = {"id": sl.IdField}

    for col, space_type in column_mapping.items():
        if space_type == "text_similarity":
            schema_fields[col] = sl.String
        elif space_type == "recency":
            schema_fields[col] = sl.Timestamp
        elif space_type == "number":
            schema_fields[col] = sl.Float
        elif space_type == "category":
            schema_fields[col] = sl.String

    return schema_fields


def create_schema_class(index_name: str, schema_fields: Dict[str, type]) -> sl.Schema:
    """Dynamically create a schema class and instantiate it."""
    SchemaClass = type(
        f"{index_name.capitalize()}Schema",
        (sl.Schema,),
        {"__annotations__": schema_fields}
    )
    return SchemaClass()


def create_text_space(schema: sl.Schema, col: str) -> Tuple[sl.Space, Dict]:
    """Create a text similarity space for a column."""
    space = sl.TextSimilaritySpace(
        text=getattr(schema, col),
        model=EMBEDDING_MODEL
    )
    weight = {space: sl.Param(f"{col}_weight", default=TEXT_WEIGHT)}
    return space, weight


def create_recency_space(schema: sl.Schema, col: str) -> Tuple[sl.Space, Dict]:
    """Create a recency space for a timestamp column."""
    space = sl.RecencySpace(
        timestamp=getattr(schema, col),
        period_time_list=[sl.PeriodTime(timedelta(days=300))],
        negative_filter=-0.25
    )
    weight = {space: sl.Param(f"{col}_weight", default=RECENCY_WEIGHT)}
    return space, weight


def create_number_space(schema: sl.Schema, col: str) -> Tuple[sl.Space, Dict]:
    """Create a number space for a numeric column."""
    space = sl.NumberSpace(
        number=getattr(schema, col),
        min_value=0.0,
        max_value=1.0,
        mode=sl.Mode.MAXIMUM
    )
    weight = {space: sl.Param(f"{col}_weight", default=NUMBER_WEIGHT)}
    return space, weight


def create_category_space(schema: sl.Schema, col: str, df: Optional[pd.DataFrame]) -> Optional[Tuple[sl.Space, Dict]]:
    """Create a category space for a categorical column."""
    if df is None or col not in df.columns:
        print(f"⚠ Skipping category space for {col}: no data available")
        return None

    categories = df[col].dropna().unique().tolist()
    categories = [str(cat) for cat in categories]

    space = sl.CategoricalSimilaritySpace(
        category_input=getattr(schema, col),
        categories=categories,
        negative_filter=0.0,
        uncategorized_as_category=True
    )
    weight = {space: sl.Param(f"{col}_weight", default=CATEGORY_WEIGHT)}
    return space, weight


def create_spaces(schema: sl.Schema, column_mapping: Dict[str, str], df: Optional[pd.DataFrame]) -> Tuple[List, Dict, Any]:
    """Create all spaces based on column mapping."""
    spaces = []
    weights = {}
    text_space = None  # Track first text space for query building

    for col, space_type in column_mapping.items():
        result = None

        if space_type == "text_similarity":
            space, weight = create_text_space(schema, col)
            # Keep reference to first text space - needed for .similar() in query
            if not text_space:
                text_space = space
            result = (space, weight)

        elif space_type == "recency":
            result = create_recency_space(schema, col)

        elif space_type == "number":
            result = create_number_space(schema, col)

        elif space_type == "category":
            result = create_category_space(schema, col, df)

        if result:
            space, weight = result
            spaces.append(space)  # All spaces go here (including text_space)
            weights.update(weight)

    return spaces, weights, text_space  # text_space needed separately for query


def create_query(index: sl.Index, schema: sl.Schema, weights: Dict, text_space: Any, index_name: str) -> Optional[sl.RestQuery]:
    """Create a REST query for the index."""
    if not text_space:
        return None

    query = (
        sl.Query(index, weights=weights)
        .find(schema)
        .similar(text_space.text, sl.Param("search_query"))
        .select_all()
        .limit(sl.Param("limit"))
    )

    return sl.RestQuery(
        sl.RestDescriptor(f"{index_name}_query"),
        query
    )


def create_data_sources(schema: sl.Schema) -> List:
    """
    Create data sources for the schema.

    Only RestSource - data loaded manually via ingest_csv_data().
    DataLoaderSource doesn't work reliably (CSV not loading).
    """
    # RestSource creates HTTP endpoint for manual data ingestion
    return [sl.RestSource(schema)]


def process_single_index(index_name: str, metadata: Dict[str, Any]) -> Tuple[Optional[sl.Index], Optional[sl.RestQuery], List]:
    """Process a single index from metadata and return index, query, and sources."""
    csv_filename = metadata.get("csv_filename")
    column_mapping = metadata.get("column_mapping", {})

    print(f"Processing index: {index_name}, CSV: {csv_filename}")

    if not column_mapping:
        return None, None, []

    # Load CSV if needed for category detection
    df = load_csv_if_needed(csv_filename, column_mapping)

    # Create schema
    schema_fields = build_schema_fields(column_mapping)
    schema = create_schema_class(index_name, schema_fields)

    # Create spaces
    spaces, weights, text_space = create_spaces(schema, column_mapping, df)

    if not spaces:
        print(f"⚠ No valid spaces created for {index_name}")
        return None, None, []

    # Create index
    index = sl.Index(spaces)

    # Create query
    query = create_query(index, schema, weights, text_space, index_name)

    # Create data sources (REST endpoint only - manual ingestion)
    sources = create_data_sources(schema)

    print(f"✓ Loaded index: {index_name}")
    return index, query, sources


def initialize_server():
    """Initialize Superlinked server. Loads indexes from environment variable if present."""
    all_indices = []
    all_queries = []
    all_sources = []

    # Read metadata from environment variable (passed by MCP tool)
    metadata_json = os.environ.get('INDEX_METADATA')
    if metadata_json:
        try:
            all_metadata = json.loads(metadata_json)

            # Get the single index (only one per server instance)
            if len(all_metadata) != 1:
                print(f"⚠ Warning: Expected 1 index, found {len(all_metadata)}")

            index_name, index_config = next(iter(all_metadata.items()))
            print(f"✓ Loading index: {index_name}")

            index, query, sources = process_single_index(index_name, index_config)

            if index:
                all_indices.append(index)
            if query:
                all_queries.append(query)
            all_sources.extend(sources)

        except Exception as e:
            print(f"⚠ Could not load index: {e}")

    # Create executor
    vector_db = sl.InMemoryVectorDatabase()

    executor = sl.RestExecutor(
        sources=all_sources,
        indices=all_indices,
        queries=all_queries,
        vector_database=vector_db
    )

    sl.SuperlinkedRegistry.register(executor)

    if all_indices:
        print(f"✓ Server ready with {len(all_indices)} index(es)")
    else:
        print("✓ Server ready (empty - use MCP create_index tool)")


# Initialize the server
initialize_server()

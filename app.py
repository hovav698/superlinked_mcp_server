"""
Dynamic Superlinked app with InMemoryExecutor.
Creates an in-memory search app without REST API server.
"""
from superlinked import framework as sl
from datetime import timedelta
from pathlib import Path
import pandas as pd
import sys
from typing import Dict, Any, Optional, List, Tuple
from config import *

# Helper to print to stderr (not stdout, which would break MCP protocol)
def log(msg):
    print(msg, file=sys.stderr)


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
        log(f"⚠ Could not load CSV {csv_filename}: {e}")
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
        log(f"⚠ Skipping category space for {col}: no data available")
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


def create_query(index: sl.Index, schema: sl.Schema, weights: Dict, text_space: Any) -> Optional[sl.Query]:
    """Create a query for the index."""
    if not text_space:
        return None

    query = (
        sl.Query(index, weights=weights)
        .find(schema)
        .similar(text_space.text, sl.Param("search_query"))
        .select_all()
        .limit(sl.Param("limit"))
    )

    return query


def create_data_source(schema: sl.Schema) -> sl.InMemorySource:
    """
    Create an in-memory data source for the schema.
    Data will be loaded directly via source.put() method.
    """
    return sl.InMemorySource(schema)


def create_index_components(index_name: str, column_mapping: Dict[str, str], csv_path: Optional[str] = None):
    """
    Create all components for an index: schema, index, query, and source.

    Args:
        index_name: Name of the index
        column_mapping: Mapping of columns to space types
        csv_path: Optional path to CSV for category detection

    Returns:
        Tuple of (schema, index, query, source, text_space)
    """
    if not column_mapping:
        raise ValueError("column_mapping is required")

    # Load CSV if needed for category detection
    df = load_csv_if_needed(csv_path, column_mapping)

    # Create schema
    schema_fields = build_schema_fields(column_mapping)
    schema = create_schema_class(index_name, schema_fields)

    # Create spaces
    spaces, weights, text_space = create_spaces(schema, column_mapping, df)

    if not spaces:
        raise ValueError(f"No valid spaces created for {index_name}")

    # Create index
    index = sl.Index(spaces)

    # Create query
    query = create_query(index, schema, weights, text_space)

    # Create data source
    source = create_data_source(schema)

    log(f"✓ Created index components for: {index_name}")
    return schema, index, query, source, text_space


def create_app(index_name: str, column_mapping: Dict[str, str], csv_path: Optional[str] = None):
    """
    Create an InMemoryExecutor app for the given index configuration.

    Args:
        index_name: Name of the index
        column_mapping: Mapping of columns to space types
        csv_path: Optional path to CSV for category detection

    Returns:
        Tuple of (app, source, query) where:
        - app: The InMemoryApp instance with .query() method
        - source: The InMemorySource for ingesting data via source.put()
        - query: The Query object for passing to app.query()
    """
    schema, index, query, source, text_space = create_index_components(
        index_name, column_mapping, csv_path
    )

    # Create executor and run it to get an app
    executor = sl.InMemoryExecutor(
        sources=[source],
        indices=[index]
    )

    app = executor.run()

    log(f"✓ Created InMemoryExecutor app for: {index_name}")
    return app, source, query

"""
Dynamic Superlinked app with InMemoryExecutor.
Creates an in-memory search app without REST API server.
"""
from superlinked import framework as sl
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from config import EMBEDDING_MODEL, SPACE_TYPE_TO_SCHEMA_FIELD, SPACE_TYPE_WEIGHTS
from utils import extract_categories_from_file

logging.basicConfig(level=logging.INFO, format='%(message)s')


def create_schema(index_name: str, column_mapping: Dict[str, str]) -> sl.Schema:
    """Build schema field definitions from column mapping and create schema instance."""
    schema_fields = {"id": sl.IdField}

    for col, space_type in column_mapping.items():
        field_type_name = SPACE_TYPE_TO_SCHEMA_FIELD.get(space_type)
        if field_type_name:
            schema_fields[col] = getattr(sl, field_type_name)

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
    weight = {space: sl.Param(f"{col}_weight", default=SPACE_TYPE_WEIGHTS["text_similarity"])}
    return space, weight


def create_recency_space(schema: sl.Schema, col: str) -> Tuple[sl.Space, Dict]:
    """Create a recency space for a timestamp column."""
    space = sl.RecencySpace(
        timestamp=getattr(schema, col),
        period_time_list=[sl.PeriodTime(timedelta(days=300))],
        negative_filter=-0.25
    )
    weight = {space: sl.Param(f"{col}_weight", default=SPACE_TYPE_WEIGHTS["recency"])}
    return space, weight


def create_number_space(schema: sl.Schema, col: str) -> Tuple[sl.Space, Dict]:
    """Create a number space for a numeric column."""
    space = sl.NumberSpace(
        number=getattr(schema, col),
        min_value=0.0,
        max_value=1.0,
        mode=sl.Mode.MAXIMUM
    )
    weight = {space: sl.Param(f"{col}_weight", default=SPACE_TYPE_WEIGHTS["number"])}
    return space, weight


def create_category_space(schema: sl.Schema, col: str, categories: Optional[List[str]]) -> Optional[Tuple[sl.Space, Dict]]:
    """Create a category space for a categorical column."""
    if not categories:
        logging.warning(f"Skipping category space for {col}: no categories available")
        return None

    space = sl.CategoricalSimilaritySpace(
        category_input=getattr(schema, col),
        categories=categories,
        negative_filter=0.0,
        uncategorized_as_category=True
    )
    weight = {space: sl.Param(f"{col}_weight", default=SPACE_TYPE_WEIGHTS["category"])}
    return space, weight


def create_spaces(schema: sl.Schema, column_mapping: Dict[str, str], categories_dict: Dict[str, list]) -> Tuple[List, Dict, Any]:
    """Create all spaces based on column mapping."""
    space_creators = {
        "text_similarity": create_text_space,
        "recency": create_recency_space,
        "number": create_number_space,
        "category": create_category_space
    }

    spaces = []
    weights = {}
    text_space = None  # Track first text space for query building

    for col, space_type in column_mapping.items():
        creator = space_creators.get(space_type)
        if not creator:
            continue

        # Call creator with appropriate arguments
        if space_type == "category":
            categories = categories_dict.get(col, [])
            result = creator(schema, col, categories)
        else:
            result = creator(schema, col)

        if result:
            space, weight = result
            spaces.append(space)
            weights.update(weight)

            # Keep reference to first text space - needed for .similar() in query
            if space_type == "text_similarity" and not text_space:
                text_space = space

    return spaces, weights, text_space  # text_space needed separately for query


def create_app(file_path: str, column_mapping: Dict[str, str]):
    """
    Create an InMemoryExecutor app for the given index configuration.

    Args:
        file_path: Path to data file (CSV or JSON, used for index name and category detection)
        column_mapping: Mapping of columns to space types

    Returns:
        Tuple of (app, source, query) where:
        - app: The InMemoryApp instance with .query() method
        - source: The InMemorySource for ingesting data via source.put()
        - query: The Query object for passing to app.query()
    """
    if not column_mapping:
        raise ValueError("column_mapping is required")

    if not file_path:
        raise ValueError("file_path is required")

    # Derive index name from filename
    index_name = Path(file_path).stem

    # Extract categories from file for category spaces
    categories_dict = extract_categories_from_file(file_path, column_mapping)

    # Create schema
    schema = create_schema(index_name, column_mapping)

    # Create spaces
    spaces, weights, text_space = create_spaces(schema, column_mapping, categories_dict)

    if not spaces:
        raise ValueError(f"No valid spaces created for {index_name}")

    # Create index
    index = sl.Index(spaces)

    # Create query
    query = None
    if text_space:
        query = (
            sl.Query(index, weights=weights)
            .find(schema)
            .similar(text_space.text, sl.Param("search_query"))
            .select_all()
            .limit(sl.Param("limit"))
        )

    # Create data source
    source = sl.InMemorySource(schema)

    # Create executor and run it to get an app
    executor = sl.InMemoryExecutor(
        sources=[source],
        indices=[index]
    )

    app = executor.run()

    logging.info(f"Created InMemoryExecutor app for: {index_name}")
    return app, source, query

"""
Dynamic Superlinked app that reads from Qdrant collections.
Auto-creates indexes for all collections with metadata.
"""
from superlinked import framework as sl
from datetime import timedelta
from qdrant_client import QdrantClient
from pathlib import Path
from config import *

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL)

# Get metadata from special metadata collection
try:
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
except Exception as e:
    print(f"⚠ Could not connect to Qdrant: {e}")
    collection_names = []

all_indices = []
all_queries = []
all_sources = []

# Read all metadata from _sl_metadata collection
metadata_collection = "_sl_metadata"
all_metadata = []

if metadata_collection in collection_names:
    try:
        # Scroll through all points in metadata collection
        scroll_result = client.scroll(
            collection_name=metadata_collection,
            limit=100
        )
        all_metadata = scroll_result[0]
        print(f"✓ Found {len(all_metadata)} metadata entries")
    except Exception as e:
        print(f"⚠ Could not read metadata: {e}")

print(f"Processing {len(all_metadata)} metadata entries...")
for metadata_point in all_metadata:
    try:
        metadata = metadata_point.payload
        if not metadata.get("_is_metadata"):
            continue

        index_name = metadata.get("index_name")
        csv_filename = metadata.get("csv_filename")
        column_mapping = metadata.get("column_mapping", {})

        print(f"Processing index: {index_name}, CSV: {csv_filename}")

        if not column_mapping:
            continue

        # Load CSV to detect categories if needed
        import pandas as pd
        df = None
        if csv_filename and any(space_type == "category" for space_type in column_mapping.values()):
            try:
                df = pd.read_csv(csv_filename)
            except Exception as e:
                print(f"⚠ Could not load CSV {csv_filename}: {e}")
                continue

        # Dynamically create schema
        schema_fields = {"id": sl.IdField}
        for col, space_type in column_mapping.items():
            if space_type == "text_similarity":
                schema_fields[col] = sl.String
            elif space_type == "recency":
                schema_fields[col] = sl.Timestamp
            elif space_type == "number":
                schema_fields[col] = sl.Float
            elif space_type == "category":
                schema_fields[col] = sl.String  # Category input is a String field

        SchemaClass = type(
            f"{index_name.capitalize()}Schema",
            (sl.Schema,),
            {"__annotations__": schema_fields}
        )
        schema = SchemaClass()

        # Create spaces
        spaces = []
        weights = {}
        text_space = None

        for col, space_type in column_mapping.items():
            if space_type == "text_similarity":
                space = sl.TextSimilaritySpace(
                    text=sl.chunk(getattr(schema, col), CHUNK_SIZE, CHUNK_OVERLAP),
                    model=EMBEDDING_MODEL
                )
                weights[space] = sl.Param(f"{col}_weight", default=TEXT_WEIGHT)
                if not text_space:
                    text_space = space

            elif space_type == "recency":
                space = sl.RecencySpace(
                    timestamp=getattr(schema, col),
                    period_time_list=[sl.PeriodTime(timedelta(days=300))],
                    negative_filter=-0.25
                )
                weights[space] = sl.Param(f"{col}_weight", default=RECENCY_WEIGHT)

            elif space_type == "number":
                space = sl.NumberSpace(
                    number=getattr(schema, col),
                    min_value=0.0,
                    max_value=1.0,
                    mode=sl.Mode.MAXIMUM
                )
                weights[space] = sl.Param(f"{col}_weight", default=NUMBER_WEIGHT)

            elif space_type == "category":
                # Get unique categories from CSV data
                if df is not None and col in df.columns:
                    categories = df[col].dropna().unique().tolist()
                    categories = [str(cat) for cat in categories]  # Ensure strings

                    space = sl.CategoricalSimilaritySpace(
                        category_input=getattr(schema, col),
                        categories=categories,
                        negative_filter=0.0,
                        uncategorized_as_category=True
                    )
                    weights[space] = sl.Param(f"{col}_weight", default=CATEGORY_WEIGHT)
                else:
                    print(f"⚠ Skipping category space for {col}: no data available")
                    continue

            spaces.append(space)

        # Create index
        index = sl.Index(spaces)
        all_indices.append(index)

        # Create query
        if text_space:
            query = (
                sl.Query(index, weights=weights)
                .find(schema)
                .similar(text_space.text, sl.Param("search_query"))
                .select_all()
                .limit(sl.Param("limit"))
            )
            all_queries.append(
                sl.RestQuery(
                    sl.RestDescriptor(f"{index_name}_query"),
                    query
                )
            )

        # Data sources
        all_sources.append(sl.RestSource(schema))

        if csv_filename:
            # Construct full path if only filename is stored
            csv_path = Path(csv_filename)
            if not csv_path.is_absolute():
                csv_path = WORK_DIR / csv_filename

            print(f"Loading CSV from: {csv_path} (exists: {csv_path.exists()})")

            if csv_path.exists():
                loader = sl.DataLoaderSource(
                    schema,
                    sl.DataLoaderConfig(str(csv_path), sl.DataFormat.CSV)
                )
                all_sources.append(loader)
            else:
                print(f"⚠ CSV file not found: {csv_path}")

        print(f"✓ Loaded index: {index_name}")

    except Exception as e:
        print(f"⚠ Skipping {index_name}: {e}")

# Create executor if we have indexes
if all_indices:
    vector_db = sl.QdrantVectorDatabase(QDRANT_URL, "")

    executor = sl.RestExecutor(
        sources=all_sources,
        indices=all_indices,
        queries=all_queries,
        vector_database=vector_db
    )

    sl.SuperlinkedRegistry.register(executor)
    print(f"✓ Server ready with {len(all_indices)} index(es)")
else:
    print("⚠ No indexes found - create one with create_index tool")

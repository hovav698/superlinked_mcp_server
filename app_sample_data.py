"""
Auto-generated Superlinked app for index: sample_data
"""
from superlinked import framework as sl
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class Sample_dataSchema(sl.Schema):
    """Schema for sample_data data."""
    id: sl.IdField
    body: sl.String
    created_at: sl.Timestamp
    usefulness: sl.Float


# Create schema instance
schema = Sample_dataSchema()


# Body space for semantic similarity
body_space = sl.TextSimilaritySpace(
    text=sl.chunk(schema.body, chunk_size=100, chunk_overlap=20),
    model="sentence-transformers/all-mpnet-base-v2"
)

# Created_at space for temporal relevance
created_at_space = sl.RecencySpace(
    timestamp=schema.created_at,
    period_time_list=[sl.PeriodTime(timedelta(days=300))],
    negative_filter=-0.25
)

# Usefulness space for numerical optimization
usefulness_space = sl.NumberSpace(
    number=schema.usefulness,
    min_value=0.0,
    max_value=1.0,
    mode=sl.Mode.MAXIMUM
)

# Build index with all spaces
index = sl.Index([body_space, created_at_space, usefulness_space])

# Create query with parameterized weights
query = (
    sl.Query(
        index,
        weights={
            body_space: sl.Param("body_weight", default=2.0),
            created_at_space: sl.Param("created_at_weight", default=0.3),
            usefulness_space: sl.Param("usefulness_weight", default=0.5)
        },
    )
    .find(schema)
    .similar(body_space.text, sl.Param("search_query"))
    .select_all()
    .limit(sl.Param("limit"))
)

# Data sources
rest_source = sl.RestSource(schema)

# CSV data loader
data_loader_config = sl.DataLoaderConfig(
    "sample_data.csv",
    sl.DataFormat.CSV,
)
csv_loader_source = sl.DataLoaderSource(schema, data_loader_config)

# Use Qdrant vector database
vector_database = sl.QdrantVectorDatabase("http://localhost:6333", "")

# Create REST executor
executor = sl.RestExecutor(
    sources=[rest_source, csv_loader_source],
    indices=[index],
    queries=[
        sl.RestQuery(
            sl.RestDescriptor("sample_data_query"),
            query
        )
    ],
    vector_database=vector_database
)

# Register executor
sl.SuperlinkedRegistry.register(executor)

"""
Superlinked app configuration for Qdrant-backed RAG pipeline.
Single file containing all configuration and data loading.

To run:
1. Ensure Qdrant is running: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
2. Start server: APP_MODULE_PATH=app python -m superlinked.server
3. Query via REST API or use query.py
"""
from superlinked import framework as sl
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class ParagraphSchema(sl.Schema):
    """Schema for paragraph data with metadata."""
    id: sl.IdField
    body: sl.String
    created_at: sl.Timestamp
    usefulness: sl.Float


# Create schema instance
paragraph = ParagraphSchema()

# Create three semantic spaces
# 1. TextSimilaritySpace for semantic relevance with chunking
relevance_space = sl.TextSimilaritySpace(
    text=sl.chunk(paragraph.body, chunk_size=100, chunk_overlap=20),
    model="sentence-transformers/all-mpnet-base-v2"
)

# 2. RecencySpace for temporal decay
recency_space = sl.RecencySpace(
    timestamp=paragraph.created_at,
    period_time_list=[sl.PeriodTime(timedelta(days=300))],
    negative_filter=-0.25
)

# 3. NumberSpace for usefulness scores
usefulness_space = sl.NumberSpace(
    number=paragraph.usefulness,
    min_value=0.0,
    max_value=1.0,
    mode=sl.Mode.MAXIMUM
)

# Build index with all three spaces
paragraph_index = sl.Index([relevance_space, recency_space, usefulness_space])

# Create query with parameterized weights
knowledgebase_query = (
    sl.Query(
        paragraph_index,
        weights={
            relevance_space: sl.Param("relevance_weight"),
            recency_space: sl.Param("recency_weight"),
            usefulness_space: sl.Param("usefulness_weight"),
        },
    )
    .find(paragraph)
    .similar(relevance_space.text, sl.Param("search_query"))
    .select_all()
    .limit(sl.Param("limit"))
)

# Data sources
# 1. REST source for manual ingestion
rest_source = sl.RestSource(paragraph)

# 2. DataLoader source for automatic CSV loading
data_loader_config = sl.DataLoaderConfig(
    "sample_data.csv",
    sl.DataFormat.CSV,
)
csv_loader_source = sl.DataLoaderSource(paragraph, data_loader_config)

# Use Qdrant vector database (empty string for local instance with no auth)
vector_database = sl.QdrantVectorDatabase("http://localhost:6333", "")

# Create REST executor with Qdrant
executor = sl.RestExecutor(
    sources=[rest_source, csv_loader_source],
    indices=[paragraph_index],
    queries=[
        sl.RestQuery(
            sl.RestDescriptor("sample_data_query"),
            knowledgebase_query
        )
    ],
    vector_database=vector_database
)

# Register executor
sl.SuperlinkedRegistry.register(executor)

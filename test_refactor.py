"""Test refactored metadata storage"""
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import QDRANT_URL
import uuid

# Setup
csv_path = "business_news.csv"
index_name = Path(csv_path).stem
column_mapping = {
    'headline': 'text_similarity',
    'short_description': 'text_similarity',
    'category': 'category',
    'date': 'recency'
}

client = QdrantClient(url=QDRANT_URL)
metadata_collection = "_sl_metadata"

# Load CSV
df = pd.read_csv(csv_path)
print(f"Loaded CSV: {len(df)} rows")

# Create metadata collection if needed
collections = [c.name for c in client.get_collections().collections]
if metadata_collection not in collections:
    client.create_collection(
        collection_name=metadata_collection,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE)
    )
    print(f"Created metadata collection: {metadata_collection}")

# Generate consistent UUID from index name
point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, index_name))
print(f"Using UUID: {point_uuid} for {index_name}")

# Store metadata
metadata = {
    "_is_metadata": True,
    "index_name": index_name,
    "csv_filename": Path(csv_path).name,
    "column_mapping": column_mapping,
    "total_rows": len(df)
}

client.upsert(
    collection_name=metadata_collection,
    points=[PointStruct(
        id=point_uuid,
        vector=[0.0],
        payload=metadata
    )]
)
print(f"✓ Stored metadata for {index_name}")

# Verify
stored = client.retrieve(metadata_collection, ids=[point_uuid])
print(f"✓ Verified metadata: {stored[0].payload}")

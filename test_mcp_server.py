"""
Test script for simplified MCP server functionality.
Tests the logic without calling MCP decorated functions.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import subprocess
import time
import requests
import sys
import os

from config import *

print("=" * 60)
print("Testing Simplified MCP Server Logic")
print("=" * 60)

# Test 1: Preview CSV
print("\n[Test 1] Preview CSV")
print("-" * 60)
try:
    df = pd.read_csv("sample_data.csv")
    preview = {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(3).to_dict('records')
    }
    print(f"✓ Found {preview['rows']} rows")
    print(f"✓ Columns: {', '.join(preview['columns'])}")
    print(f"✓ Data types: {preview['dtypes']}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Create Index with Metadata
print("\n[Test 2] Create Index with Metadata")
print("-" * 60)
try:
    client = QdrantClient(url=QDRANT_URL)
    index_name = "sample_data"

    # Delete if exists
    try:
        client.delete_collection(index_name)
        print(f"ℹ️  Deleted existing collection")
    except:
        pass

    # Correct column names
    column_mapping = {
        "body": "text_similarity",
        "created_at": "recency",
        "usefulness": "number"
    }

    # Create collection
    client.create_collection(
        collection_name=index_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    # Store metadata as a special point (UUID: all zeros)
    metadata_uuid = "00000000-0000-0000-0000-000000000000"
    metadata = {
        "_is_metadata": True,
        "csv_filename": "sample_data.csv",
        "column_mapping": column_mapping,
        "total_rows": len(df)
    }

    client.upsert(
        collection_name=index_name,
        points=[PointStruct(
            id=metadata_uuid,
            vector=np.zeros(768).tolist(),
            payload=metadata
        )]
    )

    print(f"✓ Collection created: {index_name}")
    print(f"✓ Column mapping: {column_mapping}")
    print(f"✓ Metadata stored as special point")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Verify Metadata Retrieval
print("\n[Test 3] Verify Metadata Retrieval")
print("-" * 60)
try:
    metadata_point = client.retrieve(
        collection_name=index_name,
        ids=[metadata_uuid]
    )
    if metadata_point and metadata_point[0].payload.get("_is_metadata"):
        stored_metadata = metadata_point[0].payload
        print(f"✓ Retrieved metadata from special point")
        print(f"  CSV filename: {stored_metadata.get('csv_filename')}")
        print(f"  Column mapping: {stored_metadata.get('column_mapping')}")
        print(f"  Total rows: {stored_metadata.get('total_rows')}")
    else:
        print("❌ No metadata point found")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Start Server and Load Data
print("\n[Test 4] Start Server and Load Data")
print("-" * 60)
try:
    # Kill existing servers
    subprocess.run(["pkill", "-9", "-f", "superlinked.server"], capture_output=True)
    time.sleep(2)

    # Start server
    env = os.environ.copy()
    env["APP_MODULE_PATH"] = "app"

    process = subprocess.Popen(
        [sys.executable, "-m", "superlinked.server"],
        env=env,
        cwd=str(WORK_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print(f"ℹ️  Starting server (PID: {process.pid})...")

    # Wait for server to be ready
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                print(f"✓ Server ready on port {SERVER_PORT}")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server failed to start")
        raise Exception("Server startup timeout")

    # Wait for data load
    print(f"ℹ️  Waiting {DATA_LOAD_WAIT}s for data load...")
    time.sleep(DATA_LOAD_WAIT)

    # Check if data was loaded
    info = client.get_collection(index_name)
    print(f"✓ Points in collection: {info.points_count}")
    print(f"✓ Vectors in collection: {info.vectors_count}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Query Index
print("\n[Test 5] Query Index")
print("-" * 60)
try:
    url = f"http://localhost:{SERVER_PORT}/api/v1/search/{index_name}_query"
    payload = {"search_query": "vacation policy", "limit": 3}

    print(f"Query: '{payload['search_query']}'")
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()

    results = response.json()
    entries = results.get('entries', [])

    print(f"✓ Found {len(entries)} results:")
    for i, entry in enumerate(entries, 1):
        fields = entry.get('fields', {})
        score = entry.get('metadata', {}).get('score', 0)
        print(f"  {i}. Score: {score:.4f}")
        print(f"     ID: {entry.get('id', 'N/A')}")
        print(f"     Body: {fields.get('body', 'N/A')[:80]}...")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: List Indexes
print("\n[Test 6] List Indexes")
print("-" * 60)
try:
    collections = client.get_collections().collections
    print(f"✓ Found {len(collections)} collection(s):")
    for c in collections:
        collection_info = client.get_collection(c.name)
        print(f"  - {c.name}: {collection_info.points_count} points, {collection_info.vectors_count} vectors")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)

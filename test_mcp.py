#!/usr/bin/env python3
"""
Test script for Superlinked MCP Server
Tests the complete workflow: preview -> create index -> list -> query
"""
import json
import time
from pathlib import Path

from superlinked_utils import (
    list_indexes,
    create_index_and_load,
    query_index
)

def test_workflow():
    """Test the complete MCP workflow."""

    print("=" * 80)
    print("Testing Superlinked MCP Server")
    print("=" * 80)

    csv_path = str(Path("sample_data.csv").absolute())

    # Test 1: List existing indexes
    print("\n[Test 1] Listing existing indexes...")
    indexes = list_indexes()
    print(f"✓ Found {len(indexes)} indexes: {indexes}")

    # Test 2: Preview CSV (simple pandas read)
    print("\n[Test 2] Previewing CSV...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"✓ CSV has {len(df)} rows and columns: {list(df.columns)}")
    print(f"  First row sample:")
    print(f"    id: {df.iloc[0]['id']}")
    print(f"    body (truncated): {df.iloc[0]['body'][:80]}...")

    # Test 3: Create index
    print("\n[Test 3] Creating index...")
    column_mapping = {
        "body": "text_similarity",
        "created_at": "recency",
        "usefulness": "number"
    }

    weights = {
        "body": 1.0,
        "created_at": 0.5,
        "usefulness": 0.5
    }

    try:
        result = create_index_and_load(csv_path, column_mapping, weights)
        print(f"✓ Index creation result:")
        print(json.dumps(result, indent=2))

        if result["status"] == "exists":
            print("  Note: Index already exists, skipping creation")
        else:
            print(f"  Server running on port {result['server_port']}")
            print(f"  Loaded {result['document_count']} documents")

    except Exception as e:
        print(f"✗ Error creating index: {e}")
        return False

    # Test 4: List indexes again
    print("\n[Test 4] Listing indexes after creation...")
    indexes = list_indexes()
    print(f"✓ Found {len(indexes)} indexes: {indexes}")

    # Test 5: Query the index
    print("\n[Test 5] Querying index...")
    index_name = "sample_data"
    query_text = "vacation policy"

    try:
        results = query_index(index_name, query_text, limit=3)
        print(f"✓ Query results for '{query_text}':")
        print(f"  Found {len(results)} results")

        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            doc_id = result.get("id", "N/A")
            body = result.get("fields", {}).get("body", "")
            body_preview = body[:100] + "..." if len(body) > 100 else body

            print(f"\n  Result {i}:")
            print(f"    ID: {doc_id}")
            print(f"    Score: {score:.4f}")
            print(f"    Body: {body_preview}")

    except Exception as e:
        print(f"✗ Error querying index: {e}")
        print(f"  This might be because the server needs more time to start")
        print(f"  Or the weights don't match the column names")
        return False

    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_workflow()
    exit(0 if success else 1)

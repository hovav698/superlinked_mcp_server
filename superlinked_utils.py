"""
Superlinked utility functions for dynamic schema and index creation using REST API.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from qdrant_client import QdrantClient
import json
import subprocess
import time
import requests
import os
import signal
import sys


# Qdrant configuration
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RECENCY_PERIOD_DAYS = 300
BASE_PORT = 8080

# Working directory for generated files (configurable via env var)
WORK_DIR = Path(os.getenv("SUPERLINKED_WORK_DIR", Path(__file__).parent.absolute()))


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(url=QDRANT_URL)


def list_qdrant_collections() -> List[str]:
    """
    List all collections in Qdrant.

    Returns:
        List of collection names
    """
    client = get_qdrant_client()
    try:
        collections = client.get_collections()
        return [col.name for col in collections.collections]
    except Exception:
        return []


def list_indexes() -> List[str]:
    """
    List all created indexes by scanning for app_{index_name}.py files.

    Returns:
        List of index names
    """
    app_files = WORK_DIR.glob("app_*.py")
    index_names = []
    for app_file in app_files:
        # Extract index name from app_{index_name}.py
        file_name = app_file.stem  # Remove .py
        if file_name.startswith("app_"):
            index_name = file_name[4:]  # Remove "app_" prefix
            index_names.append(index_name)
    return index_names


def get_server_port(index_name: str) -> int:
    """Get the port number for a specific index server."""
    # Use hash of index name to get consistent port
    import hashlib
    hash_val = int(hashlib.md5(index_name.encode()).hexdigest(), 16)
    return BASE_PORT + (hash_val % 1000)


def generate_app_file(index_name: str, column_mapping: Dict[str, str],
                      weights: Optional[Dict[str, float]] = None, csv_path: str = "") -> str:
    # Convert to relative path if absolute
    if csv_path and Path(csv_path).is_absolute():
        csv_path = Path(csv_path).name
    """
    Generate a Superlinked app file for the given index configuration.

    Args:
        index_name: Name of the index
        column_mapping: Column to space type mapping
        weights: Optional weights
        csv_path: Path to CSV file

    Returns:
        Path to generated app file
    """
    # Build schema fields
    field_definitions = []
    for col_name, space_type in column_mapping.items():
        if space_type == "text_similarity":
            field_definitions.append(f"    {col_name}: sl.String")
        elif space_type == "recency":
            field_definitions.append(f"    {col_name}: sl.Timestamp")
        elif space_type == "number":
            field_definitions.append(f"    {col_name}: sl.Float")

    fields_code = "\n".join(field_definitions)

    # Build spaces and collect default weights
    space_definitions = []
    space_names = []
    weight_mapping = []
    default_weights = {}

    for i, (col_name, space_type) in enumerate(column_mapping.items()):
        space_name = f"{col_name}_space"
        space_names.append(space_name)
        # Calculate default weight for this column
        default_weight = weights.get(col_name, 1.0) if weights else (1.0 if space_type == "text_similarity" else 0.5)
        default_weights[col_name] = default_weight
        # Use parameterized weight with default value
        weight_mapping.append(f"            {space_name}: sl.Param(\"{col_name}_weight\", default={default_weight})")

        if space_type == "text_similarity":
            space_definitions.append(f"""
# {col_name.capitalize()} space for semantic similarity
{space_name} = sl.TextSimilaritySpace(
    text=sl.chunk(schema.{col_name}, chunk_size=100, chunk_overlap=20),
    model="{EMBEDDING_MODEL}"
)""")
        elif space_type == "recency":
            space_definitions.append(f"""
# {col_name.capitalize()} space for temporal relevance
{space_name} = sl.RecencySpace(
    timestamp=schema.{col_name},
    period_time_list=[sl.PeriodTime(timedelta(days={RECENCY_PERIOD_DAYS}))],
    negative_filter=-0.25
)""")
        elif space_type == "number":
            space_definitions.append(f"""
# {col_name.capitalize()} space for numerical optimization
{space_name} = sl.NumberSpace(
    number=schema.{col_name},
    min_value=0.0,
    max_value=1.0,
    mode=sl.Mode.MAXIMUM
)""")

    spaces_code = "\n".join(space_definitions)
    space_list = ", ".join(space_names)
    weights_code = ",\n".join(weight_mapping)

    # Find first text similarity space for query
    text_space = None
    for col_name, space_type in column_mapping.items():
        if space_type == "text_similarity":
            text_space = f"{col_name}_space"
            break

    # Generate app file content
    app_content = f'''"""
Auto-generated Superlinked app for index: {index_name}
"""
from superlinked import framework as sl
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class {index_name.capitalize()}Schema(sl.Schema):
    """Schema for {index_name} data."""
    id: sl.IdField
{fields_code}


# Create schema instance
schema = {index_name.capitalize()}Schema()

{spaces_code}

# Build index with all spaces
index = sl.Index([{space_list}])

# Create query with parameterized weights
query = (
    sl.Query(
        index,
        weights={{
{weights_code}
        }},
    )
    .find(schema)
    .similar({text_space}.text, sl.Param("search_query"))
    .select_all()
    .limit(sl.Param("limit"))
)

# Data sources
rest_source = sl.RestSource(schema)

# CSV data loader
data_loader_config = sl.DataLoaderConfig(
    "{csv_path}",
    sl.DataFormat.CSV,
)
csv_loader_source = sl.DataLoaderSource(schema, data_loader_config)

# Use Qdrant vector database
vector_database = sl.QdrantVectorDatabase("{QDRANT_URL}", "")

# Create REST executor
executor = sl.RestExecutor(
    sources=[rest_source, csv_loader_source],
    indices=[index],
    queries=[
        sl.RestQuery(
            sl.RestDescriptor("{index_name}_query"),
            query
        )
    ],
    vector_database=vector_database
)

# Register executor
sl.SuperlinkedRegistry.register(executor)
'''

    # Write to file in WORK_DIR
    app_file_path = WORK_DIR / f"app_{index_name}.py"
    with open(app_file_path, 'w') as f:
        f.write(app_content)

    return str(app_file_path)


def start_superlinked_server(index_name: str) -> Dict[str, Any]:
    """
    Start a Superlinked REST server for the given index on default port 8080.

    Args:
        index_name: Name of the index

    Returns:
        Dict with server info (port, pid)
    """
    port = 8080
    app_module = f"app_{index_name}"

    # Check if server already running on port 8080
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            # Kill existing server to restart with new config
            print(f"Stopping existing server on port {port}...")
            subprocess.run(["pkill", "-9", "-f", "superlinked.server"], capture_output=True)
            time.sleep(2)
    except requests.exceptions.RequestException:
        pass

    # Start new server (change to WORK_DIR first)
    env = os.environ.copy()
    env["APP_MODULE_PATH"] = app_module

    process = subprocess.Popen(
        [sys.executable, "-m", "superlinked.server"],
        env=env,
        cwd=str(WORK_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to start
    print(f"Starting server on port {port}...")
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ Server started successfully on port {port}")
                return {"port": port, "pid": process.pid, "status": "started"}
        except requests.exceptions.RequestException:
            time.sleep(1)

    return {"port": port, "pid": process.pid, "status": "starting"}


def ingest_data_via_rest(index_name: str, csv_path: str, port: int) -> int:
    """
    Ingest CSV data via REST API.

    Args:
        index_name: Name of the index
        csv_path: Path to CSV file
        port: Server port

    Returns:
        Number of documents ingested
    """
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')

    url = f"http://localhost:{port}/api/v1/ingest/{index_name}_schema"
    success_count = 0

    print(f"Ingesting {len(records)} documents...")
    for i, record in enumerate(records):
        try:
            response = requests.post(url, json=record, headers={"Content-Type": "application/json"}, timeout=10)
            response.raise_for_status()
            success_count += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(records):
                print(f"  Ingested {i+1}/{len(records)} documents")
        except requests.exceptions.RequestException as e:
            print(f"  Error ingesting record {i+1}: {e}")

    return success_count


def create_index_and_load(csv_path: str, column_mapping: Dict[str, str],
                         weights: Optional[Dict[str, float]] = None,
                         recreate: bool = False) -> Dict[str, Any]:
    """
    Create a Superlinked index and load CSV data using REST API.

    Args:
        csv_path: Path to CSV file
        column_mapping: Dict mapping column names to space types
        weights: Optional weights for each column
        recreate: If True, delete existing index and recreate it

    Returns:
        Dict with status and info
    """
    # Extract index name from CSV filename
    index_name = Path(csv_path).stem

    # Check if file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Check if index already exists (based on app file)
    app_file = WORK_DIR / f"app_{index_name}.py"
    if app_file.exists():
        if recreate:
            # Recreate: delete existing index and continue
            print(f"⚠️  Recreating index '{index_name}'...")
            print(f"  Stopping existing server...")
            subprocess.run(["pkill", "-9", "-f", "superlinked.server"], capture_output=True)
            time.sleep(2)

            print(f"  Deleting app file: {app_file}")
            app_file.unlink()

            # Delete Qdrant collection
            print(f"  Deleting Qdrant collection 'default'...")
            try:
                requests.delete("http://localhost:6333/collections/default", timeout=2)
            except:
                pass  # Collection might not exist

            print(f"✓ Old index deleted, creating new one...")
        else:
            # Don't recreate: return exists message
            return {
                "status": "exists",
                "index_name": index_name,
                "message": f"Index '{index_name}' already exists. Use query_index to search it, or set recreate=True to rebuild it."
            }

    # Generate app file
    print(f"Generating app file for '{index_name}'...")
    app_file = generate_app_file(index_name, column_mapping, weights, csv_path)
    print(f"✓ Generated: {app_file}")

    # Start Superlinked server
    server_info = start_superlinked_server(index_name)
    port = server_info["port"]

    # Read CSV for info
    df = pd.read_csv(csv_path)
    expected_count = len(df)

    # Trigger DataLoader to load CSV data
    schema_name = f"{index_name}_schema"
    print(f"Triggering DataLoader to load {expected_count} documents from CSV...")
    try:
        response = requests.post(f"http://localhost:{port}/data-loader/{schema_name}/run", timeout=5)
        response.raise_for_status()
        print(f"✓ DataLoader started successfully")

        # Wait for DataLoader to process all documents
        print(f"  Processing and indexing {expected_count} documents...")
        time.sleep(15)  # Allow time for embedding and indexing
        print(f"✓ Data loading complete")
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Could not trigger DataLoader: {e}")
        print(f"  You may need to manually trigger it via POST /data-loader/{schema_name}/run")

    return {
        "status": "success",
        "index_name": index_name,
        "document_count": expected_count,
        "columns": list(df.columns),
        "server_port": port,
        "server_pid": server_info.get("pid"),
        "message": f"Index '{index_name}' created and server started. DataLoader should have loaded {expected_count} documents."
    }


def query_index(index_name: str, query_text: str,
               limit: int = 5, weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Query an existing index via REST API.

    Args:
        index_name: Name of the index to query
        query_text: Natural language query
        limit: Maximum number of results
        weights: Optional weights to override defaults

    Returns:
        List of results with scores and fields
    """
    # Check if app file exists (indicates index was created)
    app_file = WORK_DIR / f"app_{index_name}.py"
    if not app_file.exists():
        raise ValueError(f"Index '{index_name}' not found. App file {app_file} does not exist.")

    # Use default port 8080
    port = 8080

    # Check if server is running, if not, start it
    server_running = False
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            server_running = True
    except requests.exceptions.RequestException:
        pass

    # If server not running, start it
    if not server_running:
        print(f"Server not running, starting it for index '{index_name}'...")
        server_info = start_superlinked_server(index_name)
        if server_info["status"] != "started":
            raise ValueError(f"Failed to start server on port {port}")

    # Prepare query payload
    payload = {
        "search_query": query_text,
        "limit": limit
    }

    # Add weight parameters only if explicitly provided
    # If not provided, Superlinked will use the default values from sl.Param(default=...)
    if weights:
        for key, value in weights.items():
            payload[f"{key}_weight"] = value

    # Query via REST API
    url = f"http://localhost:{port}/api/v1/search/{index_name}_query"

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        # Parse results
        entries = results.get('entries', [])
        parsed_results = []

        for entry in entries:
            parsed_results.append({
                "id": entry.get("id"),
                "score": entry.get("metadata", {}).get("score", 0.0),
                "fields": entry.get("fields", {})
            })

        return parsed_results

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error querying index: {str(e)}")

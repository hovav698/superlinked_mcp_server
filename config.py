"""
Simple configuration for Superlinked MCP Server.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Space type mappings
SPACE_TYPE_TO_SCHEMA_FIELD = {
    "text_similarity": "String",
    "recency": "Timestamp",
    "number": "Float",
    "category": "String"
}

# Consolidated weight mapping
SPACE_TYPE_WEIGHTS = {
    "text_similarity": 1.0,
    "recency": 0.3,
    "number": 0.3,
    "category": 0.5
}


# Server
SERVER_PORT = 8080
SERVER_STARTUP_TIMEOUT = 30
DATA_LOAD_WAIT = 15

# Working Directory
WORK_DIR = Path(__file__).parent.absolute()

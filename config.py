"""
Simple configuration for Superlinked MCP Server.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Vector Database
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Fixed Weights (simple defaults)
TEXT_WEIGHT = 1.0
RECENCY_WEIGHT = 0.3
NUMBER_WEIGHT = 0.3
CATEGORY_WEIGHT = 0.5

# Text Processing
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# Server
SERVER_PORT = 8080
SERVER_STARTUP_TIMEOUT = 30
DATA_LOAD_WAIT = 15

# Debugging
DEBUG_MODE = True  # Set to False for production
DEBUG_TIMEOUT = 600  # 10 minutes for debugging with breakpoints

# Working Directory
WORK_DIR = Path(__file__).parent.absolute()

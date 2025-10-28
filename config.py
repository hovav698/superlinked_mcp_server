"""
Simple configuration for Superlinked MCP Server.
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Claude Model - Change this to use a different model
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Space type mappings
SPACE_TYPE_TO_SCHEMA_FIELD = {
    "text_similarity": "String",
    "recency": "Timestamp",
    "number": "Float",
    "category": "String"
}

# Server
SERVER_PORT = 8080
SERVER_STARTUP_TIMEOUT = 30
DATA_LOAD_WAIT = 15

# Working Directory
WORK_DIR = Path(__file__).parent.absolute()

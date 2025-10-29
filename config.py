"""
Simple configuration for Superlinked MCP Server.
"""
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

# Claude Model - Change this to use a different model
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# Space type mappings
SPACE_TYPE_TO_SCHEMA_FIELD = {
    "text_similarity": "String",
    "recency": "Timestamp",
    "number": "Float",
    "category": "String"
}

# Working Directory
WORK_DIR = Path(__file__).parent.absolute()

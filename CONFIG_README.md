# Configuration Guide

## Overview

The Superlinked RAG system is now database-agnostic and fully configurable through environment variables. All configuration is centralized in `config.py` and can be overridden using a `.env` file or environment variables.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to configure your setup:
   ```bash
   # Example: Switch to a different vector database
   VECTOR_DB_PROVIDER=qdrant
   VECTOR_DB_URL=http://localhost:6333

   # Example: Use a different embedding model
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

3. The system will automatically use your configuration when you run the MCP server or utilities.

## Configuration Sections

### Directory Configuration

**SUPERLINKED_WORK_DIR**: Working directory for generated app files
- Default: Current directory
- Example: `/path/to/your/work/directory`

### Vector Database Configuration

**VECTOR_DB_PROVIDER**: Vector database provider
- Default: `qdrant`
- Supported: `qdrant` (more coming soon)
- Example: `VECTOR_DB_PROVIDER=qdrant`

**VECTOR_DB_URL**: Connection URL for the vector database
- Default: `http://localhost:6333`
- Example: `VECTOR_DB_URL=http://localhost:6333`

**VECTOR_DB_API_KEY**: API key for the vector database (if required)
- Default: Empty string
- Example: `VECTOR_DB_API_KEY=your-api-key-here`

**VECTOR_DB_DEFAULT_COLLECTION**: Default collection/index name
- Default: `default`
- Example: `VECTOR_DB_DEFAULT_COLLECTION=my_collection`

### Embedding Model Configuration

**EMBEDDING_MODEL**: Sentence transformer model for text embeddings
- Default: `sentence-transformers/all-mpnet-base-v2`
- Example: `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

**CHUNK_SIZE**: Token size for text chunking
- Default: `100`
- Example: `CHUNK_SIZE=150`

**CHUNK_OVERLAP**: Overlap between chunks
- Default: `20`
- Example: `CHUNK_OVERLAP=30`

### Space Configuration

**RECENCY_PERIOD_DAYS**: Time period for recency decay (days)
- Default: `300`
- Example: `RECENCY_PERIOD_DAYS=365`

**RECENCY_NEGATIVE_FILTER**: Negative filter value for old documents
- Default: `-0.25`
- Example: `RECENCY_NEGATIVE_FILTER=-0.5`

**NUMBER_MIN_VALUE**: Minimum value for number spaces
- Default: `0.0`
- Example: `NUMBER_MIN_VALUE=0.0`

**NUMBER_MAX_VALUE**: Maximum value for number spaces
- Default: `1.0`
- Example: `NUMBER_MAX_VALUE=10.0`

### Server Configuration

**SUPERLINKED_BASE_PORT**: Port for the Superlinked REST server
- Default: `8080`
- Example: `SUPERLINKED_BASE_PORT=8000`

**SERVER_STARTUP_TIMEOUT**: Timeout for server startup (seconds)
- Default: `30`
- Example: `SERVER_STARTUP_TIMEOUT=60`

**DATA_LOADING_WAIT_TIME**: Wait time for data loading (seconds)
- Default: `15`
- Example: `DATA_LOADING_WAIT_TIME=30`

### Default Weights Configuration

**DEFAULT_TEXT_WEIGHT**: Default weight for text similarity spaces
- Default: `1.0`
- Example: `DEFAULT_TEXT_WEIGHT=2.0`

**DEFAULT_RECENCY_WEIGHT**: Default weight for recency spaces
- Default: `0.5`
- Example: `DEFAULT_RECENCY_WEIGHT=0.3`

**DEFAULT_NUMBER_WEIGHT**: Default weight for number spaces
- Default: `0.5`
- Example: `DEFAULT_NUMBER_WEIGHT=0.7`

### Logging Configuration

**LOG_LEVEL**: Logging level
- Default: `INFO`
- Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- Example: `LOG_LEVEL=DEBUG`

## Database-Agnostic Architecture

The system is designed to support multiple vector databases. Currently, Qdrant is fully supported, and the architecture allows for easy addition of new providers.

### Adding a New Vector Database Provider

1. Update `config.py` to add your provider to `VECTOR_DB_CLASS_MAP`:
   ```python
   VECTOR_DB_CLASS_MAP = {
       "qdrant": "QdrantVectorDatabase",
       "pinecone": "PineconeVectorDatabase",  # New provider
       # ...
   }
   ```

2. Update `superlinked_utils.py` to handle the new provider in:
   - `get_vector_db_client()`: Add client initialization
   - `list_vector_db_collections()`: Add collection listing logic
   - Collection deletion logic in `create_index_and_load()`

3. Test with your new provider by setting:
   ```bash
   VECTOR_DB_PROVIDER=pinecone
   VECTOR_DB_URL=https://your-pinecone-url
   VECTOR_DB_API_KEY=your-api-key
   ```

## Configuration Helpers

The `config.py` module provides helper functions to access configuration:

```python
from config import get_vector_db_config, get_embedding_config, get_default_weights

# Get vector database configuration
db_config = get_vector_db_config()
# Returns: {"provider": "qdrant", "url": "...", "api_key": "...", ...}

# Get embedding configuration
embed_config = get_embedding_config()
# Returns: {"model": "...", "chunk_size": 100, "chunk_overlap": 20}

# Get default weights
weights = get_default_weights()
# Returns: {"text_similarity": 1.0, "recency": 0.5, "number": 0.5}
```

## Environment Variable Priority

Configuration values are loaded in this order (highest priority first):

1. **Environment variables**: Set in your shell or .env file
2. **Default values**: Defined in config.py

Example:
```bash
# In .env or shell
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# This will override the default model in config.py
```

## Validation

The configuration is automatically validated when `config.py` is imported. Invalid values will print a warning message but won't prevent the system from starting.

## Best Practices

1. **Use .env for local development**: Keep your local settings in `.env` (which should be gitignored)
2. **Use environment variables for production**: Set env vars directly in your deployment environment
3. **Document custom configurations**: If you change defaults, document why in your .env file
4. **Test configuration changes**: After changing settings, test with a small dataset first

## Troubleshooting

### Configuration not loading
- Ensure `.env` is in the same directory as `config.py`
- Check that `python-dotenv` is installed: `pip install python-dotenv`

### Vector database connection fails
- Verify `VECTOR_DB_URL` is correct and the database is running
- Check `VECTOR_DB_API_KEY` if authentication is required

### Model not found
- Ensure the `EMBEDDING_MODEL` name is correct
- Check that the model is available in Hugging Face or your model repository

### Performance issues
- Increase `CHUNK_SIZE` for fewer, larger chunks
- Adjust `DATA_LOADING_WAIT_TIME` if data loading times out
- Tune weight values to optimize result quality

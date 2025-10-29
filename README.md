# Superlinked MCP Server

A Model Context Protocol (MCP) server that provides semantic search and RAG capabilities using [Superlinked](https://superlinked.com)'s vector search framework. This server enables AI assistants to create and query vector indexes from structured data files with support for multiple search dimensions (text similarity, recency, categories, and numbers).

## Overview

This project creates an MCP server that wraps Superlinked's InMemoryExecutor, allowing any MCP-compatible client (like Claude Desktop) to perform sophisticated semantic search operations on structured data. The server provides tools to preview data files, create multi-dimensional vector indexes, and query them using natural language.

### How It Works

The server uses Superlinked's vector search capabilities to create in-memory indexes that combine multiple search dimensions:

1. **Text Similarity**: Semantic search using sentence transformers embeddings
2. **Recency**: Time-based ranking for timestamp fields
3. **Number**: Numeric value ranking (prices, ratings, etc.)
4. **Category**: Categorical matching and filtering

When you create an index, the server:
- Analyzes your data file structure
- Creates appropriate vector spaces based on your column mapping
- Applies custom weights to each dimension
- Ingests the data into an in-memory index
- Makes it available for semantic queries

## Features

- **Multi-format support**: CSV and JSON data files
- **Flexible schema**: Dynamic schema creation based on your data
- **Multi-dimensional search**: Combine text, time, numbers, and categories
- **Custom weighting**: Control the importance of each search dimension
- **In-memory execution**: Fast queries without external dependencies
- **MCP compatibility**: Works with any MCP client

## Installation

### Prerequisites

- Python 3.10+
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd rag_repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set required environment variable:
```bash
# Required for Streamlit demo
export ANTHROPIC_API_KEY=your_api_key_here

# Optional: Change Claude model (default: claude-haiku-4-5-20251001)
export CLAUDE_MODEL=claude-haiku-4-5-20251001
```

## MCP Server Tools

The server provides three main tools:

### 1. `preview_file`

Preview the structure and contents of a data file before indexing. Use this tool to understand your data structure and decide which spaces to create for indexing.

**Parameters:**
- `file_path` (string): Path to data file (CSV or JSON)
- `rows` (integer, optional): Number of sample rows to preview (default: 5)

**Returns:**
- Total row count
- Column names and data types
- Sample records

**Example use case:**
"Show me what's in sample_data/business_news.json"

### 2. `create_index`

Create a searchable vector index from structured data.

**Parameters:**
- `file_path` (string): Path to data file to index
- `column_mapping` (dict): Maps column names to space types
  - `text_similarity`: For text fields (semantic search)
  - `recency`: For timestamp/date fields (Unix timestamps)
  - `number`: For numeric fields
  - `category`: For categorical fields
- `weights` (dict): Maps column names to weight values (0.0 to 1.0)
  - Higher weight = more influence on ranking
  - Example: `{"description": 0.8, "timestamp": 0.3}`

**Returns:**
- Index name (derived from filename)
- Column configuration
- Ingestion statistics
- Applied weights

**Example use case:**
"Create an index from business_news.json with text search on the description field and recency on the date field. Put more weight on the description."

### 3. `query_index`

Search an index using natural language queries.

**Parameters:**
- `index_name` (string): Name of the index (filename without extension)
- `query_text` (string): Natural language search query
- `limit` (integer, optional): Maximum results to return (default: 5)

**Returns:**
- Array of results with:
  - `id`: Record identifier
  - `score`: Similarity score
  - `fields`: All indexed fields and their values

**Example use case:**
"Search the business_news index for articles about strikes"

## Integration with External MCP Clients

### Claude Desktop

To use this server with Claude Desktop:

1. Locate your Claude Desktop configuration file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the server configuration:
```json
{
  "mcpServers": {
    "superlinked-rag": {
      "command": "python",
      "args": ["/absolute/path/to/rag_repo/mcp_server.py"],
      "type": "stdio"
    }
  }
}
```

3. Restart Claude Desktop

4. The server tools will now be available to Claude. You can ask Claude to:
   - Preview your data files
   - Create vector indexes
   - Perform semantic searches


### Streamlit Demo App

This repository includes a Streamlit chatbot demo that uses the Claude Agent SDK as an MCP client. This demonstrates how to build a conversational interface with the MCP server.

### Running the Demo

1. Run the Streamlit app:
```bash
streamlit run streamlit_chatbot.py
```

2. Open your browser to the provided URL (typically `http://localhost:8501`)

### Demo Features

- **Real-time streaming**: See AI reasoning steps as they happen
- **Conversation memory**: Multi-turn conversations with context
- **Tool visualization**: Watch the AI use MCP tools
- **Interactive UI**: Clean interface with execution step display

### Example Conversation Flow

1. "I have this file: sample_data/business_news.json. Show me what's in it."
2. "Create an index for semantic search on the description field and time-based ranking on the date field."
3. "Who wanted to have a strike?"
4. "This news is old, put more weight on the date field."
5. "Any news related to gas? What does it say?"



## Resources

- [Superlinked Documentation](https://docs.superlinked.com)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)
- [FastMCP](https://github.com/jlowin/fastmcp)


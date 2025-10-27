# RAG Pipeline with Superlinked

A Retrieval-Augmented Generation (RAG) pipeline based on the Superlinked HR Knowledgebase implementation. This uses multi-space vector search combining text relevance (with sentence-transformers), recency, and usefulness scores with OpenAI for response generation.

## Implementation

This implementation is based on the [Superlinked HR Knowledgebase notebook](https://github.com/superlinked/superlinked/blob/main/notebook/rag_hr_knowledgebase.ipynb) and demonstrates:

- **TextSimilaritySpace** - Semantic search with sentence-transformers/all-mpnet-base-v2
- **Text Chunking** - 100-token chunks with 20-token overlap for better granularity
- **RecencySpace** - Temporal decay for prioritizing recent documents (300-day window)
- **NumberSpace** - Usefulness scoring to rank by quality/feedback
- **Weighted Query** - Dynamic weight adjustment for different spaces
- **RAG Integration** - Retrieved context fed to OpenAI LLM for response generation
- **Single-File Pipeline** - All functionality in one script, no pickle files

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your `.env` file contains your OpenAI API key (for LLM generation only):
```
OPENAI_API_KEY=your_api_key_here
```

## Data Format

The CSV file must have the following columns:

- `id` - Unique identifier
- `body` - Text content (paragraphs)
- `created_at` - Unix timestamp (e.g., 1672527600 for Dec 31, 2022)
- `usefulness` - Float between 0.0 and 1.0 (quality/feedback score)

Example:
```csv
id,body,created_at,usefulness
0,"Management is tasked with ensuring...",1672527600,0.85
1,"The company offers a comprehensive vacation policy...",1680307200,0.92
```

## Usage

The pipeline runs as a single command that creates the index and queries in one execution:

**Basic query:**
```bash
python rag_pipeline.py sample_data.csv "What is the vacation policy?"
```

**Query with more results:**
```bash
python rag_pipeline.py sample_data.csv "What is the vacation policy?" 5
```

**Query with custom weights:**
```bash
python rag_pipeline.py sample_data.csv "What is the vacation policy?" 5 1.0 0.8 0.6
```

Arguments:
1. CSV file path (required)
2. Query text (required)
3. Top K results (optional, default: 3)
4. Relevance weight (optional, default: 1.0)
5. Recency weight (optional, default: 0.5)
6. Usefulness weight (optional, default: 0.5)

The pipeline will:
1. Load the CSV data
2. Create three embedding spaces (relevance, recency, usefulness)
3. Build the multi-space index with sentence-transformers
4. Query using weighted multi-space embeddings
5. Retrieve top-k most similar documents
6. Display document text and usefulness scores
7. Generate a response using OpenAI based on the retrieved context

## How It Works

### Three Embedding Spaces

1. **TextSimilaritySpace (Relevance)**
   - Uses `sentence-transformers/all-mpnet-base-v2` model
   - Text is chunked into 100-token segments with 20-token overlap
   - Enables semantic similarity matching
   - Embeddings generated automatically by Superlinked

2. **RecencySpace (Temporal)**
   - Prioritizes documents within a 300-day window
   - Applies -0.25 penalty to older documents
   - Helps surface fresh, up-to-date information

3. **NumberSpace (Usefulness)**
   - Ranks documents by usefulness scores (0.0 to 1.0)
   - Uses MAXIMUM mode to prefer higher-rated content
   - Reflects quality or community feedback

### Weighted Query System

At query time, you can adjust the importance of each space:
- **Higher relevance weight** = prioritize semantic match
- **Higher recency weight** = prioritize recent documents
- **Higher usefulness weight** = prioritize high-quality documents

This allows you to:
- Balance between accuracy and freshness
- Prioritize newer policies over older ones
- Surface highly-rated content first

### RAG Pipeline

1. **Load CSV**: Read data with id, body, created_at, usefulness columns
2. **Create Index**: Build multi-space index with sentence-transformers embeddings
3. **Query**: Execute weighted multi-space query via app.query()
4. **Retrieve**: Get top-K results and convert to pandas DataFrame
5. **Generate**: Feed retrieved context to OpenAI LLM for answer generation

## Example Workflow

```bash
# 1. Query with default weights
python rag_pipeline.py sample_data.csv "What is the parental leave policy?"

# 2. Prioritize recent documents (higher recency weight)
python rag_pipeline.py sample_data.csv "What is the remote work policy?" 5 1.0 0.9 0.5

# 3. Prioritize usefulness scores (higher usefulness weight)
python rag_pipeline.py sample_data.csv "What is the vacation policy?" 5 1.0 0.3 0.8

# 4. Get more results
python rag_pipeline.py sample_data.csv "What are the benefits?" 8
```

## Files

- `rag_pipeline.py` - Single-file RAG pipeline (indexing + querying + generation)
- `sample_data.csv` - Sample HR policy data (16 paragraphs)
- `requirements.txt` - Python dependencies
- `.env` - OpenAI API key for LLM generation

## Key Features from the Notebook

- **Text Chunking**: Breaks documents into smaller segments for better granularity
- **Conflict Resolution**: Balance old accurate policies vs newer but potentially incorrect ones through weight tuning
- **Specialized Coverage**: Maintain access to older policies while incorporating newer content
- **Quality Control**: Filter low-usefulness documents via scoring
- **Flexible Weighting**: Manually adjust parameters based on your use case
- **Sentence Transformers**: Uses open-source embeddings (no API costs for embeddings)
- **OpenAI Integration**: Uses OpenAI only for final response generation

## Notes

- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (open-source, runs locally)
- **LLM Generation**: OpenAI gpt-3.5-turbo (requires API key)
- **Single Execution**: Index is created and queried in memory during each run (no persistence files)
- **Based on Notebook**: Query approach follows the official [Superlinked HR Knowledgebase notebook](https://github.com/superlinked/superlinked/blob/main/notebook/rag_hr_knowledgebase.ipynb)
- Timestamps are Unix timestamps (seconds since epoch)
- Usefulness scores should reflect quality, user feedback, or validation metrics
- Text chunking with overlap ensures better context preservation

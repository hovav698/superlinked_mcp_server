"""
Script to query the Superlinked Redis index via REST API.
"""
import sys
import os
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


def query_with_llm(index_name: str, query_text: str, top_k: int = 3,
                   relevance_weight: float = 1.0, recency_weight: float = 0.5,
                   usefulness_weight: float = 0.5, server_url: str = "http://localhost:8080"):
    """
    Query the Redis index via REST API and generate response with OpenAI.

    Args:
        index_name: Name of the index (from CSV filename)
        query_text: The search query
        top_k: Number of results to retrieve
        relevance_weight: Weight for semantic relevance (default: 1.0)
        recency_weight: Weight for recency (default: 0.5)
        usefulness_weight: Weight for usefulness score (default: 0.5)
        server_url: URL of the Superlinked REST server
    """
    # Load environment variables
    load_dotenv()

    print(f"Querying index: '{index_name}'")
    print(f"Query: '{query_text}'")
    print(f"Weights - Relevance: {relevance_weight}, Recency: {recency_weight}, Usefulness: {usefulness_weight}\n")

    # Construct API endpoint
    query_endpoint = f"{index_name}_query"
    api_url = f"{server_url}/api/v1/search/{query_endpoint}"

    # Prepare query payload
    payload = {
        "search_query": query_text,
        "limit": top_k,
        "relevance_weight": relevance_weight,
        "recency_weight": recency_weight,
        "usefulness_weight": usefulness_weight
    }

    # Make POST request to query API
    print(f"Calling REST API: {api_url}")
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Superlinked server at {server_url}")
        print("Make sure create_index.py is running to start the REST server.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error querying index: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        sys.exit(1)

    # Parse response
    results = response.json()
    entries = results.get('entries', [])

    if not entries or len(entries) == 0:
        print("No results found.")
        return

    # Display results
    print(f"Found {len(entries)} relevant documents:")
    print("-" * 80)

    retrieved_texts = []
    for idx, entry in enumerate(entries):
        doc_id = entry.get('id', 'N/A')
        fields = entry.get('fields', {})
        metadata = entry.get('metadata', {})

        body = fields.get('body', '')
        usefulness = fields.get('usefulness', 'N/A')
        score = metadata.get('score', 'N/A')

        retrieved_texts.append(body)

        # Truncate body for display
        body_preview = body[:200] + "..." if len(body) > 200 else body

        print(f"{idx + 1}. [ID: {doc_id}] (Score: {score:.4f}, Usefulness: {usefulness})")
        print(f"   {body_preview}\n")

    # Generate response with OpenAI
    print("=" * 80)
    print("Generating response with OpenAI...")
    print("=" * 80)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare context from retrieved documents
    context = "\n\n".join([
        f"Document {idx + 1}:\n{text}"
        for idx, text in enumerate(retrieved_texts)
    ])

    # Create prompt for OpenAI
    prompt = f"""Based on the following context from our knowledge base, please answer the question.

Context:
{context}

Question: {query_text}

Answer:"""

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful HR assistant. Answer questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    print("\nAnswer:")
    print("-" * 80)
    print(answer)
    print("-" * 80)


def main():
    """Main function to query the RAG pipeline."""
    if len(sys.argv) < 3:
        print("Usage: python query.py <csv_filename_stem> <query> [top_k] [relevance_weight] [recency_weight] [usefulness_weight] [server_url]")
        print("\nExample:")
        print('  python query.py sample_data "What is the vacation policy?"')
        print('  python query.py sample_data "What is the vacation policy?" 5')
        print('  python query.py sample_data "What is the vacation policy?" 5 1.0 0.8 0.5')
        print('  python query.py sample_data "What is the vacation policy?" 5 1.0 0.8 0.5 "http://localhost:8000"')
        sys.exit(1)

    index_name = sys.argv[1]
    query_text = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    relevance_weight = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    recency_weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    usefulness_weight = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    server_url = sys.argv[7] if len(sys.argv) > 7 else "http://localhost:8080"

    query_with_llm(
        index_name,
        query_text,
        top_k,
        relevance_weight,
        recency_weight,
        usefulness_weight,
        server_url
    )


if __name__ == "__main__":
    main()

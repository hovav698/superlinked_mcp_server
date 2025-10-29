"""
Claude Agent SDK client wrapper for RAG chatbot.
Pure Claude SDK functions without Streamlit-specific logic.
"""
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from pathlib import Path
from config import CLAUDE_MODEL


def get_client_options():
    """
    Create and return MCP client options.

    Returns:
        ClaudeAgentOptions configured with MCP server and system prompt
    """
    mcp_server_path = Path(__file__).parent / "mcp_server.py"

    system_prompt = """You are a Superlinked RAG assistant that helps users work with their data files.
    When working with data files, only use the tools from the 'superlinked-rag' MCP server."""

    options = ClaudeAgentOptions(
        model=CLAUDE_MODEL,
        system_prompt=system_prompt,
        allowed_tools=[
            "mcp__superlinked-rag__preview_file",
            "mcp__superlinked-rag__create_index",
            "mcp__superlinked-rag__query_index"
        ],  # For security reasons, only allow Superlinked RAG MCP tools
        mcp_servers={
            "superlinked-rag": {
                "command": "python",
                "args": [str(mcp_server_path)],
                "type": "stdio"
            }
        }
    )
    return options


def create_client():
    """
    Create a new Claude SDK client.

    Returns:
        ClaudeSDKClient instance (not yet connected)
    """
    options = get_client_options()
    return ClaudeSDKClient(options=options)


async def connect_client(client):
    """
    Connect a Claude SDK client.

    Args:
        client: ClaudeSDKClient instance to connect
    """
    await client.connect()


async def disconnect_client(client):
    """
    Disconnect a Claude SDK client.

    Args:
        client: ClaudeSDKClient instance to disconnect
    """
    await client.disconnect()


async def stream_query(client, user_input: str):
    """
    Stream query using Claude SDK client.
    Yields raw messages from the client.

    Args:
        client: ClaudeSDKClient instance
        user_input: User's query text

    Yields:
        Messages from client.receive_response()
    """
    await client.query(user_input)

    async for message in client.receive_response():
        yield message

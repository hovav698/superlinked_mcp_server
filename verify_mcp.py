#!/usr/bin/env python3
"""Quick verification that MCP server can start."""
import sys
import subprocess

# Test that the server can import and describe its tools
result = subprocess.run(
    ["/Users/hovav/miniconda3/envs/rag_test/bin/python", "-c", """
from mcp_server import mcp
import inspect

# Get all tool functions
tools = []
for name, func in inspect.getmembers(mcp):
    if hasattr(func, '__mcp_tool__') or (hasattr(func, '_tool_metadata')):
        tools.append((name, func))

if len(tools) == 0:
    # Try accessing _tools directly
    if hasattr(mcp, '_tools'):
        tools = [(name, tool) for name, tool in mcp._tools.items()]

print(f"✓ MCP server loaded successfully")
print(f"✓ Found {len(tools)} tool(s)")
for name, _ in tools:
    print(f"  - {name}")
"""],
    capture_output=True,
    text=True,
    env={"SUPERLINKED_WORK_DIR": "/Users/hovav/projects/rag_repo"}
)

print(result.stdout)
if result.stderr:
    print("stderr:", result.stderr)

sys.exit(result.returncode)

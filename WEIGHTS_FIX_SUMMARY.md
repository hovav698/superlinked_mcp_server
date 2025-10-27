# Weights Parameter Parsing Fix

## Problem

When using the MCP server from Claude Desktop, the `weights` parameter (type `Optional[Dict[str, float]]`) was being received as a JSON string instead of a dictionary:

```
1 validation error for call[create_index]
weights
  Input should be a valid dictionary [type=dict_type, input_value='{"body": 1.5, ...}', input_type=str]
```

This was causing Pydantic validation to fail at the FastMCP framework level, before the function body was even reached.

## Root Cause

FastMCP uses Pydantic to validate function parameters. When Claude Desktop passes the optional `weights` parameter as a JSON string, Pydantic validation rejects it because the type annotation was `Optional[Dict[str, float]]`, which doesn't accept strings.

This appears to be an MCP client behavior where optional dictionary parameters are serialized as JSON strings when passing over the stdio transport, while required dictionaries (like `column_mapping`) are passed correctly as objects.

## Solution

The fix requires two changes:

### 1. Update Type Annotations

Changed the `weights` parameter type annotation to accept both dictionaries and strings using `Union`:

```python
# Before:
weights: Optional[Dict[str, float]] = None

# After:
weights: Optional[Union[Dict[str, float], str]] = None
```

This allows Pydantic validation to accept string inputs from the MCP client.

### 2. Add JSON String Parsing Logic

Added JSON parsing logic at the start of both functions to convert string inputs to dictionaries:

**In `create_index()` (mcp_server.py:152-160)**

```python
# Handle weights being passed as JSON string from MCP
if isinstance(weights, str):
    try:
        weights = json.loads(weights)
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid weights format. Expected dictionary, got malformed JSON string: {weights}",
            "json_error": str(e)
        })
```

**In `query_indexed_data()` (mcp_server.py:275-283)**

```python
# Handle weights being passed as JSON string from MCP
if isinstance(weights, str):
    try:
        weights = json.loads(weights)
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid weights format. Expected dictionary, got malformed JSON string: {weights}",
            "json_error": str(e)
        })
```

## Testing

Created `test_weights_parsing.py` to verify the fix handles all cases:

1. **Dictionary input** - Passes through unchanged (normal Python usage)
2. **JSON string input** - Correctly parsed to dictionary (MCP usage)
3. **Malformed JSON** - Caught with helpful error message
4. **None value** - Passes through unchanged (optional parameter)

All tests pass successfully.

## Usage

The fix is transparent to users. Both calling methods now work:

### From Python code:
```python
weights = {"body": 2.0, "created_at": 0.3, "usefulness": 0.3}
create_index(csv_path, column_mapping, weights, recreate=True)
```

### From Claude Desktop MCP (JSON string):
```json
{
  "csv_path": "/path/to/data.csv",
  "column_mapping": {"body": "text_similarity", "created_at": "recency"},
  "weights": "{\"body\": 2.0, \"created_at\": 0.3, \"usefulness\": 0.3}",
  "recreate": true
}
```

Both will work correctly after this fix.

## Files Modified

1. `/Users/hovav/projects/rag_repo/mcp_server.py`
   - Added `Union` to imports (line 8)
   - Changed `create_index()` type annotation to `Optional[Union[Dict[str, float], str]]` (line 106)
   - Added JSON parsing to `create_index()` function (lines 152-160)
   - Changed `query_indexed_data()` type annotation to `Optional[Union[Dict[str, float], str]]` (line 244)
   - Added JSON parsing to `query_indexed_data()` function (lines 275-283)

2. `/Users/hovav/projects/rag_repo/test_weights_parsing.py` (new file)
   - Test suite to verify JSON parsing logic

## Related Issues

This issue only affects optional dictionary parameters passed through MCP. Required dictionary parameters like `column_mapping` work correctly without this workaround.

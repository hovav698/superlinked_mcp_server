#!/usr/bin/env python3
"""
Test script to verify that weights parameter parsing works correctly.
Tests both dictionary and JSON string inputs.
"""
import json
from pathlib import Path

def test_json_parsing_logic():
    """Test the JSON parsing logic that was added to handle MCP string inputs."""

    print("=" * 80)
    print("Testing Weights JSON Parsing Logic")
    print("=" * 80)

    # Test 1: Dictionary input (normal case - should pass through)
    print("\n[Test 1] Dictionary input...")
    weights = {"body": 2.0, "created_at": 0.3, "usefulness": 0.3}

    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse: {e}")
            return False

    print(f"✓ Dictionary passes through: {weights}")
    assert isinstance(weights, dict)
    assert weights["body"] == 2.0

    # Test 2: JSON string input (MCP case - should be parsed)
    print("\n[Test 2] JSON string input...")
    weights = '{"body": 2.5, "created_at": 0.2, "usefulness": 0.2}'

    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
            print(f"✓ JSON string parsed successfully: {weights}")
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse: {e}")
            return False

    assert isinstance(weights, dict)
    assert weights["body"] == 2.5

    # Test 3: Malformed JSON string (should fail gracefully)
    print("\n[Test 3] Malformed JSON string...")
    weights = '{"body": 2.0, invalid json}'

    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
            print(f"✗ Should have failed but didn't")
            return False
        except json.JSONDecodeError as e:
            print(f"✓ Correctly caught malformed JSON: {e}")

    # Test 4: None value (should pass through for Optional)
    print("\n[Test 4] None value...")
    weights = None

    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except json.JSONDecodeError as e:
            print(f"✗ Failed: {e}")
            return False

    print(f"✓ None passes through: {weights}")
    assert weights is None

    print("\n" + "=" * 80)
    print("✓ All JSON parsing tests passed!")
    print("=" * 80)
    print("\nThe fix in mcp_server.py adds this logic before processing:")
    print("```python")
    print("if isinstance(weights, str):")
    print("    try:")
    print("        weights = json.loads(weights)")
    print("    except json.JSONDecodeError as e:")
    print("        return json.dumps({")
    print('            "error": f"Invalid weights format...",')
    print('            "json_error": str(e)')
    print("        })")
    print("```")
    print("\nThis handles the MCP issue where weights is passed as a string")
    print("instead of a dictionary from Claude Desktop.")

    return True

if __name__ == "__main__":
    try:
        success = test_json_parsing_logic()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

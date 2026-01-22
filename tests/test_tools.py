
import sys
import json
# Add src to path
sys.path.append("/mnt/d/Research Experiments/nexus/src")

from multimodal.tools import get_default_executor

def test_tools():
    print("🔧 Testing Tool Executor...")
    
    executor = get_default_executor()
    
    # 1. Test Schema Generation
    schema = executor.get_schema()
    print("\n[Schema Output]")
    print(schema)
    
    parsed = json.loads(schema)
    assert len(parsed) >= 2, "Should have at least calculator and search"
    print("✅ Schema generated successfully.")
    
    # 2. Test Execution: Calculator
    print("\n[Testing Calculator]")
    res = executor.execute("calculator", {"expression": "25 * 4"})
    print(f"25 * 4 = {res}")
    assert res == "100", f"Expected 100, got {res}"
    print("✅ Calculator verified.")
    
    # 3. Test Execution: Web Search (Mock)
    print("\n[Testing Search]")
    res = executor.execute("search_web", {"query": "meaning of life"})
    print(f"Search Result: {res}")
    assert "Mock Search Result" in res
    print("✅ Search verified.")
    
    # 4. Test Error Handling
    print("\n[Testing Error Handling]")
    res = executor.execute("calculator", {"expression": "import os"})
    print(f"Malicious Input Result: {res}")
    assert "Invalid characters" in res
    print("✅ Safety check verified.")

if __name__ == "__main__":
    test_tools()

import pytest
import json
from src.multimodal.tools import Tool, ToolExecutor, calculator, search_web, get_default_executor

def test_tool_dataclass():
    tool = Tool(name="test", description="desc", func=lambda x: x, parameters={})
    assert tool.name == "test"
    assert tool.description == "desc"

def test_tool_executor_registration():
    executor = ToolExecutor()
    tool = Tool(name="echo", description="echoes", func=lambda x: x, parameters={})
    executor.register(tool)
    assert "echo" in executor.tools
    
    schema = executor.get_schema()
    assert "echo" in schema
    assert "echoes" in schema

def test_tool_executor_execute_success():
    executor = ToolExecutor()
    executor.register(Tool(name="add", description="add", func=lambda x, y: x + y, parameters={}))
    
    result = executor.execute("add", {"x": 1, "y": 2})
    assert result == "3"

def test_tool_executor_not_found():
    executor = ToolExecutor()
    result = executor.execute("missing", {})
    assert "not found" in result

def test_tool_executor_error():
    executor = ToolExecutor()
    def fail(): raise ValueError("Fail")
    executor.register(Tool(name="fail", description="fail", func=fail, parameters={}))
    
    result = executor.execute("fail", {})
    assert "Error executing fail" in result

def test_calculator():
    assert calculator("2 + 2") == "4"
    assert calculator("10 * (2 + 3)") == "50"
    assert "Invalid characters" in calculator("import os")
    assert "Error" in calculator("1 / 0")

def test_search_web():
    assert "Mock Search Result" in search_web("nexus model")

def test_default_executor():
    executor = get_default_executor()
    assert "calculator" in executor.tools
    assert "search_web" in executor.tools
    
    res = executor.execute("calculator", {"expression": "1+1"})
    assert res == "2"

import pytest
import json
from src.utils.resume_repetitive_generation import (
    gen_log_extraction, gen_json_lookup, gen_directory_lookup, gen_table_lookup
)

def test_gen_log_extraction():
    query, context, result = gen_log_extraction()
    assert "log" in query.lower()
    assert "E-" in context or "No error codes found." in result
    if result != "No error codes found.":
        # Should be a JSON list
        data = json.loads(result)
        assert isinstance(data, list)

def test_gen_json_lookup():
    query, context, result = gen_json_lookup()
    assert "value" in query.lower() or "key" in query.lower()
    data = json.loads(context)
    # result can be bool, int, float, str
    # Check if result (converted to string if needed) is in values
    values_str = [str(v) for v in data.values()]
    assert str(result) in values_str

def test_gen_directory_lookup():
    query, context, result = gen_directory_lookup()
    assert "extension" in query.lower()
    assert "Employee Directory" in context
    assert result in context

def test_gen_table_lookup():
    query, context, result = gen_table_lookup()
    assert "department" in query.lower()
    assert "|" in context
    assert result in ["Engineering", "Sales", "Marketing", "HR", "Finance"]

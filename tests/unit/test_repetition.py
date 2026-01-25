import pytest
from src.utils.repetition import PromptRepetitionEngine

def test_repetition_baseline():
    engine = PromptRepetitionEngine()
    query = "What is gravity?"
    assert engine.apply_repetition(query, style="baseline") == query
    assert engine.apply_repetition(query, factor=1) == query

def test_repetition_2x():
    engine = PromptRepetitionEngine()
    query = "Repeat this."
    expected = f"{query} {query}"
    assert engine.apply_repetition(query, style="2x") == expected
    assert engine.apply_repetition(query, factor=2) == expected

def test_repetition_3x():
    engine = PromptRepetitionEngine()
    query = "Triple."
    expected = f"{query} Let me repeat that: {query} Let me repeat that one more time: {query}"
    assert engine.apply_repetition(query, style="3x") == expected
    assert engine.apply_repetition(query, factor=3) == expected

def test_repetition_verbose():
    engine = PromptRepetitionEngine()
    query = "Verbose."
    expected = f"{query} Let me repeat that: {query}"
    assert engine.apply_repetition(query, style="verbose") == expected

def test_repetition_with_context():
    engine = PromptRepetitionEngine()
    query = "Q"
    context = "C"
    expected = "C\nQ C\nQ"
    assert engine.apply_repetition(query, context=context, style="2x") == expected

def test_repetition_default_fallback():
    engine = PromptRepetitionEngine()
    query = "Q"
    assert engine.apply_repetition(query, style="unknown") == query

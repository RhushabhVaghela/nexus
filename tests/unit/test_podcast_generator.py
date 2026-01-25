import pytest
import json
from unittest.mock import MagicMock
from src.podcast.generator import Turn, PodcastScript, generate_podcast_script, handle_user_interrupt, _parse_script_json

def test_turn_dataclass():
    turn = Turn(speaker="Host A", text="Hello", vibe="excited")
    assert turn.speaker == "Host A"
    assert turn.text == "Hello"
    assert turn.vibe == "excited"
    
    d = turn.__dict__
    assert d["speaker"] == "Host A"
    assert d["text"] == "Hello"
    assert d["vibe"] == "excited"

def test_podcast_script_serialization():
    turns = [
        Turn(speaker="Host A", text="Welcome", vibe="neutral"),
        Turn(speaker="Host B", text="Hi everyone", vibe="excited")
    ]
    script = PodcastScript(turns=turns)
    data = script.to_dict()
    
    assert "turns" in data
    assert len(data["turns"]) == 2
    assert data["turns"][0]["speaker"] == "Host A"
    
    new_script = PodcastScript.from_dict(data)
    assert len(new_script.turns) == 2
    assert new_script.turns[1].speaker == "Host B"
    assert new_script.turns[1].vibe == "excited"

def test_parse_script_json_success():
    raw_json = json.dumps({
        "turns": [
            {"speaker": "Host A", "text": "Testing", "vibe": "neutral"}
        ]
    })
    script = _parse_script_json(raw_json)
    assert len(script.turns) == 1
    assert script.turns[0].text == "Testing"

def test_parse_script_json_failure():
    with pytest.raises(ValueError, match="Failed to parse podcast script JSON"):
        _parse_script_json("invalid json")

def test_generate_podcast_script():
    mock_llm = MagicMock(return_value=json.dumps({
        "turns": [
            {"speaker": "Host A", "text": "Analysis", "vibe": "thoughtful"},
            {"speaker": "Host B", "text": "Example", "vibe": "excited"}
        ]
    }))
    
    docs = ["Doc 1 content", "Doc 2 content"]
    script = generate_podcast_script(docs, topic_hint="Math", llm=mock_llm)
    
    assert len(script.turns) == 2
    assert mock_llm.called
    args, kwargs = mock_llm.call_args
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert "Math" in messages[1]["content"]

def test_handle_user_interrupt():
    mock_llm = MagicMock(return_value=json.dumps({
        "turns": [
            {"speaker": "Host A", "text": "Good question", "vibe": "supportive"}
        ]
    }))
    
    base_turns = [Turn(speaker="Host A", text="Context", vibe="neutral")]
    base_script = PodcastScript(turns=base_turns)
    
    follow_up = handle_user_interrupt(base_script, "What about X?", llm=mock_llm)
    
    assert len(follow_up.turns) == 1
    assert follow_up.turns[0].text == "Good question"
    
    args, kwargs = mock_llm.call_args
    messages = args[0]
    assert "What about X?" in messages[1]["content"]
    assert "Context" in messages[1]["content"]

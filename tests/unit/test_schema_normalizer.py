import pytest
from src.utils.schema_normalizer import SchemaNormalizer

def test_normalize_musiccaps():
    sample = {
        "id": "123",
        "ytid": "abc.mp3",
        "caption": "A beautiful piano piece."
    }
    normalized = SchemaNormalizer.normalize(sample, "google_MusicCaps")
    
    assert normalized["id"] == "123"
    assert normalized["messages"][0]["content"] == "Describe this music."
    assert normalized["messages"][1]["content"] == "A beautiful piano piece."
    assert normalized["modalities"]["audio"][0]["path"] == "abc.mp3"

def test_normalize_pure_dove():
    sample = {
        "id": "456",
        "instruction": "Explain gravity.",
        "response": "Gravity is a force..."
    }
    normalized = SchemaNormalizer.normalize(sample, "LDJnr_Pure-Dove")
    
    assert normalized["id"] == "456"
    assert normalized["messages"][0]["content"] == "Explain gravity."
    assert normalized["messages"][1]["content"] == "Gravity is a force..."
    assert "modalities" in normalized
    assert not normalized["modalities"] # text only

def test_normalize_journeydb():
    sample = {
        "hash": "789",
        "image_path": "path/to/img.png",
        "caption": "A mountain landscape."
    }
    normalized = SchemaNormalizer.normalize(sample, "LucasFang_JourneyDB-GoT")
    
    assert normalized["id"] == "789"
    assert normalized["messages"][0]["content"] == "What is in this image?"
    assert normalized["messages"][1]["content"] == "A mountain landscape."
    assert normalized["modalities"]["image"][0]["path"] == "path/to/img.png"

def test_normalize_emm1():
    sample = {
        "id": "emm1",
        "caption": "Generic description."
    }
    normalized = SchemaNormalizer.normalize(sample, "E-MM1-100M")
    
    assert normalized["messages"][0]["content"] == "Describe this."
    assert normalized["messages"][1]["content"] == "Generic description."

def test_normalize_unknown_dataset_with_image():
    sample = {
        "id": "unknown",
        "image_path": "custom.jpg",
        "instruction": "Custom prompt",
        "output": "Custom response"
    }
    normalized = SchemaNormalizer.normalize(sample, "UnknownDataset")
    
    assert normalized["messages"][0]["content"] == "Custom prompt"
    assert normalized["messages"][1]["content"] == "Custom response"
    assert normalized["modalities"]["image"][0]["path"] == "custom.jpg"

def test_normalize_fallback_defaults():
    sample = {}
    normalized = SchemaNormalizer.normalize(sample, "Empty")
    
    assert normalized["id"] == "unknown"
    assert normalized["messages"][0]["content"] == "Describe the content."
    assert normalized["messages"][1]["content"] == ""
    assert not normalized["modalities"]

def test_all_modalities():
    # Audio
    sample_audio = {"audio_path": "a.wav"}
    assert "audio" in SchemaNormalizer.normalize(sample_audio, "any")["modalities"]
    
    # Video
    sample_video = {"video_path": "v.mp4"}
    assert "video" in SchemaNormalizer.normalize(sample_video, "any")["modalities"]
    
    # Image (already tested but for completeness)
    sample_image = {"image_path": "i.png"}
    assert "image" in SchemaNormalizer.normalize(sample_image, "any")["modalities"]

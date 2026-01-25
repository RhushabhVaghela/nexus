import pytest
import json
from unittest.mock import patch, MagicMock
from src.utils.prefetch_assets import extract_urls_from_json, download_asset

def test_extract_urls():
    data = {
        "key1": "http://example.com/1.png",
        "key2": ["https://example.com/2.jpg", {"key3": "not a url"}],
        "key4": "just text"
    }
    urls = extract_urls_from_json(data)
    assert "http://example.com/1.png" in urls
    assert "https://example.com/2.jpg" in urls
    assert len(urls) == 2

@patch("requests.get")
def test_download_asset(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"image_data"
    mock_get.return_value = mock_resp
    
    with patch("src.utils.prefetch_assets.PUBLIC_DIR", tmp_path):
        download_asset("http://example.com/logo.png")
        
        dest = tmp_path / "logo.png"
        assert dest.exists()
        assert dest.read_bytes() == b"image_data"

@patch("requests.get")
def test_download_asset_skip_existing(mock_get, tmp_path):
    dest = tmp_path / "already.png"
    dest.write_bytes(b"old_data")
    
    with patch("src.utils.prefetch_assets.PUBLIC_DIR", tmp_path):
        download_asset("http://example.com/already.png")
        assert not mock_get.called
        assert dest.read_bytes() == b"old_data"

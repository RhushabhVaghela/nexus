import pytest
from pathlib import Path
from src.utils.asset_manager import AssetManager
import shutil

class TestAssetManager:
    
    @pytest.fixture
    def asset_mgr(self, tmp_path):
        data_root = tmp_path / "data"
        remotion_public = tmp_path / "public"
        data_root.mkdir()
        remotion_public.mkdir()
        return AssetManager(data_root=str(data_root), remotion_public=str(remotion_public))

    def test_find_local_asset_found(self, asset_mgr):
        # Create a mock image asset
        mock_asset = Path(asset_mgr.data_root) / "neuron.png"
        mock_asset.write_text("fake image data")
        
        result = asset_mgr.find_local_asset("neuron")
        assert result == "neuron.png"
        assert (Path(asset_mgr.remotion_public) / "neuron.png").exists()

    def test_find_local_audio_asset(self, asset_mgr):
        # Create a mock audio asset
        mock_audio = Path(asset_mgr.data_root) / "background_music.mp3"
        mock_audio.write_text("fake audio data")
        
        result = asset_mgr.find_local_asset("music")
        assert result == "background_music.mp3"
        assert (Path(asset_mgr.remotion_public) / "background_music.mp3").exists()

    def test_find_local_asset_not_found(self, asset_mgr):
        result = asset_mgr.find_local_asset("nonexistent")
        assert result is None

    def test_resolve_asset_fallback(self, asset_mgr):
        # Should return web placeholder if local fails
        result = asset_mgr.resolve_asset("mitochondria")
        assert "unsplash.com" in result or "remotion-dev" in result

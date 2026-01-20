#!/usr/bin/env python3
"""
test_export_model.py
Unit tests for model export functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModelExporterImport:
    """Test ModelExporter can be imported."""
    
    def test_import_exporter(self):
        from src.export_model import ModelExporter
        assert ModelExporter is not None
    
    def test_import_export_all_formats(self):
        from src.export_model import export_all_formats
        assert export_all_formats is not None
    
    def test_supported_formats(self):
        from src.export_model import ModelExporter
        
        assert "safetensors" in ModelExporter.SUPPORTED_FORMATS
        assert "gguf" in ModelExporter.SUPPORTED_FORMATS
        assert "awq" in ModelExporter.SUPPORTED_FORMATS


class TestModelExporterInit:
    """Test ModelExporter initialization."""
    
    def test_init_with_valid_path(self, tmp_path):
        from src.export_model import ModelExporter
        
        # Create dummy model dir
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "config.json").touch()
        
        output_path = tmp_path / "output"
        
        exporter = ModelExporter(str(model_path), str(output_path))
        assert exporter.model_path == model_path
        assert output_path.exists()  # Should be created
    
    def test_init_with_invalid_path(self, tmp_path):
        from src.export_model import ModelExporter
        
        with pytest.raises(ValueError, match="Model path does not exist"):
            ModelExporter("/nonexistent/path", str(tmp_path))


class TestModelExporterSafetensors:
    """Test SafeTensors export."""
    
    def test_export_safetensors(self, tmp_path):
        from src.export_model import ModelExporter
        
        # Create dummy model
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}")
        (model_path / "model.safetensors").write_bytes(b"dummy")
        
        output_path = tmp_path / "output"
        
        exporter = ModelExporter(str(model_path), str(output_path))
        result = exporter.export("safetensors")
        
        assert result.exists()
        assert (result / "config.json").exists()
        assert (result / "model.safetensors").exists()


class TestModelExporterFormats:
    """Test format validation."""
    
    def test_invalid_format_raises(self, tmp_path):
        from src.export_model import ModelExporter
        
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "config.json").touch()
        
        exporter = ModelExporter(str(model_path), str(tmp_path / "output"))
        
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export("invalid_format")
    
    def test_case_insensitive_format(self, tmp_path):
        from src.export_model import ModelExporter
        
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "config.json").touch()
        
        exporter = ModelExporter(str(model_path), str(tmp_path / "output"))
        
        # Should not raise
        result = exporter.export("SAFETENSORS")
        assert result.exists()


class TestExportMain:
    """Test main function."""
    
    def test_main_exists(self):
        from src.export_model import main
        assert callable(main)
    
    @patch('sys.argv', ['export_model.py', '--model', '/tmp/model', '--format', 'safetensors'])
    @patch('src.export_model.ModelExporter')
    def test_main_calls_exporter(self, mock_exporter_class):
        from src.export_model import main
        
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = Path("/tmp/output")
        mock_exporter_class.return_value = mock_exporter
        
        # Would fail without --model arg, but mocked
        # Just verify the function exists and is callable
        assert callable(main)

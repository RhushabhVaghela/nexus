import os
import torch
import pytest
import shutil
from src.nexus_final.export import NexusExporter

@pytest.fixture
def mock_nexus_components(tmp_path):
    # Student
    student_dir = tmp_path / "student"
    student_dir.mkdir()
    (student_dir / "pytorch_model.bin").write_text("dummy_weights")
    
    # Router
    router_file = tmp_path / "router.pt"
    torch.save({"weights": [1, 2, 3]}, router_file)
    
    # Index
    index_file = tmp_path / "index.faiss"
    index_file.write_text("faiss_binary_data")
    
    return str(student_dir), str(router_file), str(index_file)

def test_export_assembly(tmp_path, mock_nexus_components):
    student_path, router_path, index_path = mock_nexus_components
    release_dir = tmp_path / "nexus-release"
    
    exporter = NexusExporter(output_dir=str(release_dir))
    exporter.export(student_path, router_path, index_path)
    
    # Verify structure
    assert (release_dir / "student_core" / "pytorch_model.bin").exists()
    assert (release_dir / "student_core" / "config.json").exists()
    assert (release_dir / "router.pt").exists()
    assert (release_dir / "knowledge_index.faiss").exists()
    assert (release_dir / "README.md").exists()
    
    # Verify content
    readme_content = (release_dir / "README.md").read_text()
    assert "# Nexus Model (v1.0)" in readme_content

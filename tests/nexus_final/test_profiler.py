import pytest
import torch
import numpy as np
import os
import shutil
from src.nexus_final.profiler import StreamingPCAProfiler

@pytest.fixture
def profiler_setup(tmp_path):
    output_dir = tmp_path / "test_profiles"
    layer_names = ["model.layers.0", "model.layers.1"]
    profiler = StreamingPCAProfiler(
        model_id="facebook/opt-125m", # Using a tiny model for fast testing
        layer_names=layer_names,
        output_dir=str(output_dir)
    )
    return profiler, layer_names, output_dir

def test_pca_rank_selection(profiler_setup):
    profiler, _, _ = profiler_setup
    # Mock explained variance ratio: 70%, 20%, 9%, 1%
    variance_ratio = np.array([0.7, 0.2, 0.09, 0.01])
    
    # Target 99% variance
    # Using slightly lower threshold to avoid float precision issues in mock
    rank = profiler._calculate_intrinsic_dimension(variance_ratio, threshold=0.989)
    assert rank == 3 # 0.7 + 0.2 + 0.09 = 0.99

    # Target 95% variance
    rank_95 = profiler._calculate_intrinsic_dimension(variance_ratio, threshold=0.95)
    assert rank_95 == 3 # 0.7 + 0.2 = 0.9 (too low), + 0.09 = 0.99 (matches)

def test_dynamic_batch_sizing(profiler_setup):
    profiler, _, _ = profiler_setup
    
    # High VRAM (e.g., 14GB free)
    bs_high = profiler.compute_optimal_batch_size(14000)
    assert bs_high == 16 or bs_high == 8 # Depending on exact ACT_PER_SAMPLE_MB
    
    # Medium VRAM (e.g., 10GB free)
    bs_med = profiler.compute_optimal_batch_size(10000)
    assert bs_med == 4 # (10000 - 6500 - 1500) / 450 = 4.4 -> 4
    
    # Low VRAM (e.g., 6GB free)
    bs_low = profiler.compute_optimal_batch_size(7000)
    assert bs_low == 1
    
    # Extremely Low
    bs_crit = profiler.compute_optimal_batch_size(2000)
    assert bs_crit == 1

def test_profiling_output_structure(profiler_setup):
    profiler, layer_names, output_dir = profiler_setup
    
    # Mock data
    dummy_data = ["This is a test sentence."] * 5
    
    # We skip the actual heavy model loading/profiling in unit tests 
    # but verify the results saving logic
    for name in layer_names:
        profiler.pcas[name].n_samples_seen_ = 100
        profiler.pcas[name].components_ = np.random.rand(profiler.n_components, 768)
        profiler.pcas[name].mean_ = np.random.rand(768)
        profiler.pcas[name].singular_values_ = np.random.rand(profiler.n_components)
        profiler.pcas[name].explained_variance_ratio_ = np.linspace(0.1, 0.001, profiler.n_components)
        profiler.critical_scores[name] = 0.5
    
    profiler.save_results()
    profiler.save_profile_summary()
    
    assert os.path.exists(output_dir / "profile_summary.json")
    for name in layer_names:
        safe_name = name.replace(".", "_")
        assert os.path.exists(output_dir / f"{safe_name}_pca.npz")

def test_buffer_processing(profiler_setup):
    profiler, layer_names, _ = profiler_setup
    layer = layer_names[0]
    
    # Fill buffer
    data = [np.random.rand(512, 768) for _ in range(3)]
    profiler.activation_buffers[layer] = data
    
    # Force process
    profiler._process_buffer(layer, force=True)
    
    # Check if fit was called (partial_fit updates n_samples_seen_)
    assert profiler.pcas[layer].n_samples_seen_ > 0
    assert len(profiler.activation_buffers[layer]) == 0

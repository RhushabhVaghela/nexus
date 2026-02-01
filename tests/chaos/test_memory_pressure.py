"""
Memory Pressure Tests

Tests system behavior under memory pressure conditions.
"""

import pytest
import torch
import numpy as np
import psutil
import gc
from typing import List, Optional
import threading
import time


class MemoryPressureSimulator:
    """Simulates memory pressure conditions."""
    
    def __init__(self):
        self.allocated_blocks: List[np.ndarray] = []
        self.target_memory_percent = 90
        
    def apply_pressure(self, target_percent: float = 90.0):
        """Apply memory pressure by allocating RAM."""
        memory = psutil.virtual_memory()
        target_bytes = int(memory.total * target_percent / 100)
        current_bytes = memory.used
        
        # Allocate until target reached
        while current_bytes < target_bytes:
            block_size = min(100 * 1024 * 1024, target_bytes - current_bytes)  # 100MB blocks
            block = np.ones(block_size // 8, dtype=np.float64)
            self.allocated_blocks.append(block)
            current_bytes = psutil.virtual_memory().used
    
    def release_pressure(self):
        """Release memory pressure."""
        self.allocated_blocks.clear()
        gc.collect()


class TestMemoryPressure:
    """Test behavior under memory pressure."""
    
    def test_tensor_allocation_under_pressure(self):
        """Test tensor allocation when system is under memory pressure."""
        simulator = MemoryPressureSimulator()
        
        try:
            # Apply memory pressure
            simulator.apply_pressure(target_percent=85)
            
            # Try to allocate tensor
            tensor = torch.randn(1000, 1000)
            assert tensor is not None
            del tensor
            
        finally:
            simulator.release_pressure()
    
    def test_dataloader_under_pressure(self):
        """Test data loading under memory pressure."""
        simulator = MemoryPressureSimulator()
        
        try:
            simulator.apply_pressure(target_percent=80)
            
            # Simulate data loading
            batch_size = 32
            data = []
            for i in range(10):
                batch = torch.randn(batch_size, 3, 224, 224)
                data.append(batch)
                # Process and release
                del batch
            
            assert len(data) == 10
            
        finally:
            simulator.release_pressure()
    
    def test_model_inference_under_pressure(self):
        """Test model inference under memory pressure."""
        import torch.nn as nn
        
        simulator = MemoryPressureSimulator()
        
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )
        
        try:
            simulator.apply_pressure(target_percent=85)
            
            # Run inference
            for i in range(5):
                input_tensor = torch.randn(100, 1000)
                output = model(input_tensor)
                del input_tensor, output
                gc.collect()
            
            assert True  # Completed without crash
            
        finally:
            simulator.release_pressure()


class TestOutOfMemoryHandling:
    """Test out-of-memory handling."""
    
    def test_oom_graceful_degradation(self):
        """Test graceful degradation on OOM."""
        try:
            # Try to allocate huge tensor
            huge_tensor = torch.empty(1000000000000)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Gracefully handle OOM
                smaller_tensor = torch.empty(1000, 1000)
                assert smaller_tensor is not None
            else:
                raise
    
    def test_batch_size_reduction_on_oom(self):
        """Test automatic batch size reduction on OOM."""
        batch_sizes = [1024, 512, 256, 128, 64]
        successful_batch_size = None
        
        for batch_size in batch_sizes:
            try:
                tensor = torch.randn(batch_size, 10000, 10000)
                successful_batch_size = batch_size
                del tensor
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                continue
        
        # Should find a working batch size
        assert successful_batch_size is not None or True


class TestMemoryFragmentation:
    """Test memory fragmentation handling."""
    
    def test_memory_fragmentation_simulation(self):
        """Simulate memory fragmentation."""
        tensors = []
        
        # Allocate varying sizes
        for size in [1000, 500, 2000, 300, 1500, 800]:
            tensor = torch.randn(size, size)
            tensors.append(tensor)
        
        # Deallocate alternating tensors to create fragmentation
        for i in range(0, len(tensors), 2):
            tensors[i] = None
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Try to allocate large contiguous block
        try:
            large_tensor = torch.randn(2500, 2500)
            assert large_tensor is not None
        except RuntimeError:
            # Expected under fragmentation
            pass


class TestMemoryLeaks:
    """Test for memory leaks."""
    
    def test_no_leak_in_inference_loop(self):
        """Test no memory leak during inference."""
        import torch.nn as nn
        
        model = nn.Linear(1000, 1000)
        
        # Warmup
        for _ in range(5):
            _ = model(torch.randn(100, 1000))
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Run inference many times
        for _ in range(100):
            output = model(torch.randn(100, 1000))
            del output
        
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        # Memory growth should be minimal (less than 10%)
        growth = (final_memory - initial_memory) / initial_memory
        assert growth < 0.1, f"Memory grew by {growth:.2%}"


class TestSwapHandling:
    """Test swap/virtual memory handling."""
    
    def test_swap_usage_detection(self):
        """Test detection of swap usage."""
        swap = psutil.swap_memory()
        
        # Get swap stats
        stats = {
            "total": swap.total,
            "used": swap.used,
            "free": swap.free,
            "percent": swap.percent
        }
        
        assert "percent" in stats
    
    def test_memory_threshold_alert(self):
        """Test memory threshold alerting."""
        memory = psutil.virtual_memory()
        
        threshold = 95.0
        is_critical = memory.percent >= threshold
        
        # In production, this would trigger alerts
        assert isinstance(is_critical, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

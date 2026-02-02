"""
Tests for Storage Tier Manager Module

Comprehensive test suite covering:
- Hot/warm/cold tier management
- Layer promotion and demotion
- Access pattern detection
- Tier statistics
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import time
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nexus.models.sli.storage_tier_manager import (
    StorageTierManager,
    StorageTierConfig,
    StorageTier,
    TieredEntry,
    TierStats,
    create_tier_manager,
)


class TestStorageTierConfig:
    """Test cases for StorageTierConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StorageTierConfig()
        
        assert config.hot_max_memory_gb == 4.0
        assert config.warm_max_size_gb == 50.0
        assert config.cold_max_size_gb == 200.0
        assert config.enable_auto_tiering == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StorageTierConfig(
            hot_max_memory_gb=8.0,
            warm_max_size_gb=100.0,
            hot_promotion_threshold=5
        )
        
        assert config.hot_max_memory_gb == 8.0
        assert config.warm_max_size_gb == 100.0
        assert config.hot_promotion_threshold == 5


class TestTierStats:
    """Test cases for TierStats."""

    def test_initial_stats(self):
        """Test initial statistics."""
        stats = TierStats()
        
        assert stats.entries == 0
        assert stats.total_size_bytes == 0
        assert stats.hit_ratio == 0.0

    def test_record_access(self):
        """Test access recording."""
        stats = TierStats()
        
        stats.record_access(hit=True, load_time_ms=10.0)
        assert stats.hits == 1
        assert stats.accesses == 1
        
        stats.record_access(hit=False)
        assert stats.misses == 1
        assert stats.accesses == 2

    def test_hit_ratio(self):
        """Test hit ratio calculation."""
        stats = TierStats()
        
        stats.record_access(hit=True)
        stats.record_access(hit=True)
        stats.record_access(hit=False)
        
        assert stats.hit_ratio == 2/3


class TestTieredEntry:
    """Test cases for TieredEntry."""

    def test_entry_creation(self):
        """Test creating a tiered entry."""
        entry = TieredEntry(
            layer_id="test_layer",
            model_id="test_model",
            layer_index=0,
            current_tier=StorageTier.WARM
        )
        
        assert entry.layer_id == "test_layer"
        assert entry.current_tier == StorageTier.WARM
        assert entry.access_count == 0

    def test_update_access(self):
        """Test access update."""
        entry = TieredEntry(
            layer_id="test_layer",
            model_id="test_model",
            layer_index=0,
            current_tier=StorageTier.WARM
        )
        
        entry.update_access()
        assert entry.access_count == 1
        assert entry.last_accessed > entry.first_accessed


class TestStorageTierManager:
    """Test cases for StorageTierManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create tier manager."""
        config = StorageTierConfig(
            hot_max_memory_gb=1.0,
            warm_max_size_gb=5.0,
            enable_auto_tiering=False,  # Disable for testing
            warm_tier_path=temp_dir,
        )
        return StorageTierManager(config)

    @pytest.fixture
    def test_layer(self):
        """Create test layer."""
        return nn.Linear(256, 256)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config is not None
        assert len(manager._entries) == 0

    def test_store_layer_warm(self, manager, test_layer):
        """Test storing layer in warm tier."""
        entry = manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        
        assert isinstance(entry, TieredEntry)
        assert entry.current_tier == StorageTier.WARM
        assert entry.warm_path is not None

    def test_store_layer_hot(self, manager, test_layer):
        """Test storing layer in hot tier."""
        entry = manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.HOT
        )
        
        assert entry.current_tier == StorageTier.HOT

    def test_get_layer(self, manager, test_layer):
        """Test getting a layer."""
        manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        
        loaded = manager.get_layer("test_layer")
        assert loaded is not None
        assert isinstance(loaded, nn.Module)

    def test_get_nonexistent_layer(self, manager):
        """Test getting non-existent layer."""
        loaded = manager.get_layer("nonexistent")
        assert loaded is None

    def test_auto_promotion(self, manager, test_layer):
        """Test automatic tier promotion."""
        manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        
        # Access multiple times to trigger promotion
        for _ in range(manager.config.hot_promotion_threshold + 1):
            manager.get_layer("test_layer")
        
        # Check promotion
        tier = manager.get_entry_tier("test_layer")
        assert tier == StorageTier.HOT

    def test_get_stats(self, manager, test_layer):
        """Test getting statistics."""
        manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        
        manager.get_layer("test_layer")
        
        stats = manager.get_stats()
        assert 'hot' in stats
        assert 'warm' in stats
        assert 'cold' in stats
        assert 'total_entries' in stats

    def test_get_tier_stats(self, manager, test_layer):
        """Test getting specific tier stats."""
        manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        
        stats = manager.get_stats(StorageTier.WARM)
        assert stats['entries'] == 1

    def test_delete_layer(self, manager, test_layer):
        """Test deleting a layer."""
        manager.store_layer(
            test_layer,
            "test_layer",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        
        success = manager.delete_layer("test_layer")
        assert success == True
        
        # Verify deletion
        assert manager.get_layer("test_layer") is None

    def test_delete_nonexistent(self, manager):
        """Test deleting non-existent layer."""
        success = manager.delete_layer("nonexistent")
        assert success == False

    def test_clear_tier(self, manager, test_layer):
        """Test clearing a tier."""
        manager.store_layer(
            test_layer,
            "layer1",
            model_id="test_model",
            layer_index=0,
            preferred_tier=StorageTier.WARM
        )
        manager.store_layer(
            test_layer,
            "layer2",
            model_id="test_model",
            layer_index=1,
            preferred_tier=StorageTier.WARM
        )
        
        manager.clear_tier(StorageTier.WARM)
        
        assert manager.get_layer("layer1") is None
        assert manager.get_layer("layer2") is None


class TestCreateTierManager:
    """Test factory function."""

    def test_create_default(self):
        """Test creating with default settings."""
        manager = create_tier_manager()
        assert isinstance(manager, StorageTierManager)
        assert manager.config.hot_max_memory_gb == 4.0

    def test_create_custom(self):
        """Test creating with custom settings."""
        manager = create_tier_manager(hot_memory_gb=8.0)
        assert manager.config.hot_max_memory_gb == 8.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
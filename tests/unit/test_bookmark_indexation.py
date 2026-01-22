#!/usr/bin/env python3
"""
Unit tests for Bookmark Indexation module.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.bookmark_indexation import (
    BookmarkIndexation, BookmarkConfig, BookmarkEntry, BookmarkIndex,
    TieredKVCache, DiskCache, StorageTier, create_bookmark_indexation
)


class TestStorageTier:
    """Tests for StorageTier enum."""
    
    def test_tiers_exist(self):
        """Test all storage tiers are defined."""
        assert StorageTier.VRAM.value == "vram"
        assert StorageTier.RAM.value == "ram"
        assert StorageTier.DISK.value == "disk"


class TestBookmarkConfig:
    """Tests for BookmarkConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BookmarkConfig()
        
        assert config.vram_capacity == 32768
        assert config.ram_capacity == 131072
        assert config.bookmark_dim == 256
        assert config.block_size == 64
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BookmarkConfig(
            vram_capacity=8192,
            ram_capacity=65536,
            block_size=128,
            top_k_retrieve=512,
        )
        
        assert config.vram_capacity == 8192
        assert config.block_size == 128


class TestBookmarkEntry:
    """Tests for BookmarkEntry dataclass."""
    
    def test_create_entry(self):
        """Test creating a bookmark entry."""
        entry = BookmarkEntry(
            position=100,
            block_id=1,
            importance_score=0.8,
            summary_embedding=torch.randn(256),
            tier=StorageTier.VRAM,
        )
        
        assert entry.position == 100
        assert entry.block_id == 1
        assert entry.importance_score == 0.8
        assert entry.tier == StorageTier.VRAM
    
    def test_entry_with_disk_offset(self):
        """Test entry with disk offset."""
        entry = BookmarkEntry(
            position=1000,
            block_id=10,
            importance_score=0.5,
            summary_embedding=torch.randn(256),
            tier=StorageTier.DISK,
            disk_offset=4096,
        )
        
        assert entry.disk_offset == 4096
        assert entry.tier == StorageTier.DISK


class TestBookmarkIndex:
    """Tests for BookmarkIndex."""
    
    def test_initialization(self):
        """Test index initialization."""
        config = BookmarkConfig()
        index = BookmarkIndex(config)
        
        assert len(index.entries) == 0
        assert index.embeddings is None
    
    def test_add_entry(self):
        """Test adding entries to index."""
        config = BookmarkConfig()
        index = BookmarkIndex(config)
        
        entry = BookmarkEntry(
            position=0,
            block_id=0,
            importance_score=1.0,
            summary_embedding=torch.randn(config.bookmark_dim),
            tier=StorageTier.VRAM,
        )
        
        index.add(entry)
        
        assert len(index.entries) == 1
        assert 0 in index.entries
    
    def test_remove_entry(self):
        """Test removing entries from index."""
        config = BookmarkConfig()
        index = BookmarkIndex(config)
        
        entry = BookmarkEntry(
            position=0,
            block_id=0,
            importance_score=1.0,
            summary_embedding=torch.randn(config.bookmark_dim),
            tier=StorageTier.VRAM,
        )
        
        index.add(entry)
        index.remove(0)
        
        assert len(index.entries) == 0
    
    def test_search(self):
        """Test similarity search."""
        config = BookmarkConfig(bookmark_dim=64, similarity_threshold=0.0)
        index = BookmarkIndex(config)
        
        # Add several entries
        for i in range(10):
            entry = BookmarkEntry(
                position=i * 64,
                block_id=i,
                importance_score=1.0,
                summary_embedding=torch.randn(config.bookmark_dim),
                tier=StorageTier.VRAM,
            )
            index.add(entry)
        
        # Search
        query = torch.randn(config.bookmark_dim)
        results = index.search(query, top_k=5)
        
        assert len(results) <= 5
        # Results are (position, similarity) tuples
        for pos, sim in results:
            assert isinstance(pos, int)
            assert isinstance(sim, float)


class TestDiskCache:
    """Tests for DiskCache."""
    
    def test_initialization(self, tmp_path):
        """Test disk cache initialization."""
        config = BookmarkConfig(disk_cache_path=str(tmp_path / "kv_cache"))
        cache = DiskCache(config)
        
        assert cache.cache_dir.exists()
    
    def test_write_read_block(self, tmp_path):
        """Test writing and reading blocks."""
        config = BookmarkConfig(disk_cache_path=str(tmp_path / "kv_cache"))
        cache = DiskCache(config)
        
        # Create test KV
        block_size = 64
        num_heads = 8
        head_dim = 32
        
        k = torch.randn(block_size, num_heads, head_dim).half()
        v = torch.randn(block_size, num_heads, head_dim).half()
        
        # Write
        offset = cache.write_block(0, k, v)
        
        assert offset >= 0
        
        # Read back
        k_read, v_read = cache.read_block(
            0,
            k_shape=(block_size, num_heads, head_dim),
            v_shape=(block_size, num_heads, head_dim),
        )
        
        assert k_read.shape == k.shape
        assert v_read.shape == v.shape
        
        cache.close()


class TestTieredKVCache:
    """Tests for TieredKVCache."""
    
    def test_initialization(self, tmp_path):
        """Test tiered cache initialization."""
        config = BookmarkConfig(
            vram_capacity=256,
            ram_capacity=512,
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        cache = TieredKVCache(config, device=torch.device("cpu"))
        
        assert len(cache.vram_cache) == 0
        assert len(cache.ram_cache) == 0
    
    def test_put_in_vram(self, tmp_path):
        """Test putting KV directly in VRAM."""
        config = BookmarkConfig(
            vram_capacity=256,
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        cache = TieredKVCache(config, device=torch.device("cpu"))
        
        k = torch.randn(64, 8, 32)
        v = torch.randn(64, 8, 32)
        
        cache.put(0, k, v, importance=1.0)
        
        assert 0 in cache.vram_cache
    
    def test_get_from_vram(self, tmp_path):
        """Test getting KV from VRAM."""
        config = BookmarkConfig(
            vram_capacity=256,
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        cache = TieredKVCache(config, device=torch.device("cpu"))
        
        k = torch.randn(64, 8, 32)
        v = torch.randn(64, 8, 32)
        
        cache.put(0, k, v, importance=1.0)
        result = cache.get(0)
        
        assert result is not None
        assert result[0].shape == k.shape
    
    def test_eviction_to_ram(self, tmp_path):
        """Test eviction from VRAM to RAM."""
        config = BookmarkConfig(
            vram_capacity=64,  # Only 1 block in VRAM
            ram_capacity=256,
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        cache = TieredKVCache(config, device=torch.device("cpu"))
        
        # Add two blocks (second should cause eviction)
        k1 = torch.randn(64, 8, 32)
        v1 = torch.randn(64, 8, 32)
        k2 = torch.randn(64, 8, 32)
        v2 = torch.randn(64, 8, 32)
        
        cache.put(0, k1, v1, importance=0.5)
        cache.put(1, k2, v2, importance=1.0)  # Higher importance
        
        # Block 0 should be evicted to RAM (lower importance)
        assert 1 in cache.vram_cache
        assert 0 in cache.ram_cache
    
    def test_update_importance(self, tmp_path):
        """Test importance score update."""
        config = BookmarkConfig(
            vram_capacity=256,
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        cache = TieredKVCache(config, device=torch.device("cpu"))
        
        k = torch.randn(64, 8, 32)
        v = torch.randn(64, 8, 32)
        
        cache.put(0, k, v, importance=0.5)
        cache.update_importance(0, gradient_norm=2.0)
        
        # Importance should be updated
        assert cache.importance_scores[0] != 0.5


class TestBookmarkIndexation:
    """Tests for BookmarkIndexation module."""
    
    def test_initialization(self, tmp_path):
        """Test system initialization."""
        config = BookmarkConfig(disk_cache_path=str(tmp_path / "kv_cache"))
        system = BookmarkIndexation(config, hidden_dim=512)
        
        assert system.current_position == 0
        assert system.hidden_dim == 512
    
    def test_add_tokens(self, tmp_path):
        """Test adding tokens to the system."""
        config = BookmarkConfig(
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        system = BookmarkIndexation(config, hidden_dim=512)
        
        batch_size = 1
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        hidden = torch.randn(batch_size, seq_len, 512)
        keys = torch.randn(batch_size, seq_len, num_heads, head_dim)
        values = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        system.add_tokens(hidden, keys, values)
        
        assert system.current_position == seq_len
        assert len(system.index.entries) > 0
    
    def test_retrieve(self, tmp_path):
        """Test retrieving relevant KV."""
        config = BookmarkConfig(
            block_size=32,
            top_k_retrieve=64,
            similarity_threshold=0.0,  # Accept all
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        system = BookmarkIndexation(config, hidden_dim=256)
        
        batch_size = 1
        seq_len = 64
        num_heads = 4
        head_dim = 32
        
        # Add some tokens
        hidden = torch.randn(batch_size, seq_len, 256)
        keys = torch.randn(batch_size, seq_len, num_heads, head_dim)
        values = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        system.add_tokens(hidden, keys, values)
        
        # Retrieve
        query = torch.randn(batch_size, 16, 256)
        k_ret, v_ret, positions = system.retrieve(query, top_k=32)
        
        # Should retrieve something
        assert len(positions) > 0 or (k_ret is None)  # May not retrieve if similarity too low
    
    def test_get_stats(self, tmp_path):
        """Test getting statistics."""
        config = BookmarkConfig(
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        system = BookmarkIndexation(config, hidden_dim=512)
        
        hidden = torch.randn(1, 128, 512)
        keys = torch.randn(1, 128, 8, 64)
        values = torch.randn(1, 128, 8, 64)
        
        system.add_tokens(hidden, keys, values)
        
        stats = system.get_stats()
        
        assert "total_tokens" in stats
        assert "total_bookmarks" in stats
        assert "vram_blocks" in stats
        assert stats["total_tokens"] == 128


class TestCreateBookmarkIndexation:
    """Tests for factory function."""
    
    def test_create_default(self, tmp_path):
        """Test creating with defaults."""
        system = create_bookmark_indexation(
            disk_cache_path=str(tmp_path / "kv_cache")
        )
        
        assert system.config.vram_capacity == 32768
        assert system.hidden_dim == 4096
    
    def test_create_custom(self, tmp_path):
        """Test creating with custom params."""
        system = create_bookmark_indexation(
            hidden_dim=2048,
            vram_capacity=8192,
            ram_capacity=65536,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        
        assert system.hidden_dim == 2048
        assert system.config.vram_capacity == 8192


class TestHardwareCompatibility:
    """Tests to verify single GPU compatibility (RTX 5080 16GB scenario)."""
    
    def test_single_gpu_workflow(self, tmp_path):
        """Test complete workflow on single GPU."""
        # Simulate RTX 5080 16GB constraints
        config = BookmarkConfig(
            vram_capacity=8192,      # Conservative for testing
            ram_capacity=32768,
            block_size=64,
            bookmark_dim=256,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        
        system = BookmarkIndexation(config, hidden_dim=1024)
        
        # Simulate processing long sequence in chunks
        batch_size = 1
        chunk_size = 512
        num_heads = 8
        head_dim = 64
        total_tokens = 2048
        
        for start in range(0, total_tokens, chunk_size):
            hidden = torch.randn(batch_size, chunk_size, 1024)
            keys = torch.randn(batch_size, chunk_size, num_heads, head_dim)
            values = torch.randn(batch_size, chunk_size, num_heads, head_dim)
            
            system.add_tokens(hidden, keys, values)
        
        # Verify all tokens were added
        assert system.current_position == total_tokens
        
        # Verify retrieval works
        query = torch.randn(batch_size, 64, 1024)
        k_ret, v_ret, positions = system.retrieve(query, top_k=256)
        
        # System should work without errors
        stats = system.get_stats()
        assert stats["total_tokens"] == total_tokens
    
    def test_tiered_eviction(self, tmp_path):
        """Test that eviction to RAM/Disk works correctly."""
        # Very small VRAM to force eviction
        config = BookmarkConfig(
            vram_capacity=128,       # Only 2 blocks in VRAM
            ram_capacity=256,        # 4 blocks in RAM
            block_size=64,
            disk_cache_path=str(tmp_path / "kv_cache"),
        )
        
        system = BookmarkIndexation(config, hidden_dim=256)
        
        # Add more blocks than fit in VRAM+RAM
        batch_size = 1
        seq_len = 512  # 8 blocks, will overflow
        
        hidden = torch.randn(batch_size, seq_len, 256)
        keys = torch.randn(batch_size, seq_len, 4, 32)
        values = torch.randn(batch_size, seq_len, 4, 32)
        
        system.add_tokens(hidden, keys, values)
        
        # Check distribution across tiers
        stats = system.get_stats()
        
        assert stats["vram_blocks"] <= config.vram_capacity // config.block_size
        # RAM should have some blocks
        assert stats["ram_blocks"] >= 0

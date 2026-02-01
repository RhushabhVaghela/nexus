"""
Comprehensive unit tests for Nested Update Scheduler.

Tests cover:
- NestedUpdateScheduler initialization
- Update frequency groups (fast/medium/slow)
- should_update() logic
- Dynamic rebalancing
- Group assignment by layer type
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

# Import the module under test
from src.nexus.models.sli.nested_scheduler import (
    NestedUpdateScheduler,
    NestedUpdateConfig,
    UpdateGroup,
    UpdateStats,
    NestedSchedulerError,
    get_nested_scheduler,
    create_attention_focused_scheduler,
)
from src.nexus.models.sli.exceptions import SLIError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """Create default nested update config."""
    return NestedUpdateConfig()


@pytest.fixture
def custom_config():
    """Create custom nested update config."""
    return NestedUpdateConfig(
        fast_interval=1,
        medium_interval=5,
        slow_interval=25,
        fast_layers={0, 1},
        medium_layers={2, 3, 4},
        slow_layers={5, 6},
        warmup_steps=50,
        dynamic_rebalancing=True,
        rebalance_interval=100
    )


@pytest.fixture
def scheduler_default():
    """Create scheduler with default config."""
    return NestedUpdateScheduler()


@pytest.fixture
def scheduler_12_layers():
    """Create scheduler for 12-layer model."""
    config = NestedUpdateConfig(
        fast_layers={0, 1, 2},
        medium_layers={3, 4, 5, 6, 7, 8},
        slow_layers={9, 10, 11}
    )
    return NestedUpdateScheduler(config, num_layers=12)


@pytest.fixture
def scheduler_with_rebalancing():
    """Create scheduler with dynamic rebalancing enabled."""
    config = NestedUpdateConfig(
        fast_layers={0, 1},
        medium_layers={2, 3},
        slow_layers={4, 5},
        dynamic_rebalancing=True,
        rebalance_interval=10,
        warmup_steps=5
    )
    return NestedUpdateScheduler(config, num_layers=6)


# ============================================================================
# Test NestedUpdateConfig
# ============================================================================

class TestNestedUpdateConfig:
    """Test suite for NestedUpdateConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NestedUpdateConfig()
        
        assert config.fast_interval == 1
        assert config.medium_interval == 10
        assert config.slow_interval == 100
        assert config.fast_layers == set()
        assert config.medium_layers == set()
        assert config.slow_layers == set()
        assert config.warmup_steps == 100
        assert config.dynamic_rebalancing is False
        assert config.rebalance_interval == 1000

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = NestedUpdateConfig(
            fast_interval=2,
            medium_interval=20,
            slow_interval=200,
            fast_layers={0, 1},
            medium_layers={2, 3, 4},
            slow_layers={5, 6},
            warmup_steps=200,
            dynamic_rebalancing=True,
            rebalance_interval=500
        )
        
        assert config.fast_interval == 2
        assert config.medium_interval == 20
        assert config.slow_interval == 200
        assert config.fast_layers == {0, 1}
        assert config.medium_layers == {2, 3, 4}
        assert config.slow_layers == {5, 6}
        assert config.warmup_steps == 200
        assert config.dynamic_rebalancing is True
        assert config.rebalance_interval == 500

    def test_config_invalid_fast_interval(self):
        """Test that invalid fast_interval raises ValueError."""
        with pytest.raises(ValueError, match="fast_interval must be >= 1"):
            NestedUpdateConfig(fast_interval=0)
        
        with pytest.raises(ValueError, match="fast_interval must be >= 1"):
            NestedUpdateConfig(fast_interval=-1)

    def test_config_invalid_medium_interval(self):
        """Test that invalid medium_interval raises ValueError."""
        with pytest.raises(ValueError, match="medium_interval must be >= 1"):
            NestedUpdateConfig(medium_interval=0)

    def test_config_invalid_slow_interval(self):
        """Test that invalid slow_interval raises ValueError."""
        with pytest.raises(ValueError, match="slow_interval must be >= 1"):
            NestedUpdateConfig(slow_interval=0)

    def test_config_interval_order_warning(self, caplog):
        """Test warning when intervals are not ordered correctly."""
        import logging
        with caplog.at_level(logging.WARNING):
            NestedUpdateConfig(fast_interval=10, medium_interval=5)
        
        assert "Intervals should be ordered" in caplog.text

    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = NestedUpdateConfig(
            fast_interval=1,
            medium_interval=5,
            fast_layers={0, 1}
        )
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['fast_interval'] == 1
        assert config_dict['medium_interval'] == 5
        assert config_dict['fast_layers'] == [0, 1]
        assert config_dict['dynamic_rebalancing'] is False

    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        data = {
            'fast_interval': 2,
            'medium_interval': 15,
            'slow_interval': 150,
            'fast_layers': [0, 1, 2],
            'medium_layers': [3, 4, 5],
            'slow_layers': [6, 7],
            'warmup_steps': 150,
            'dynamic_rebalancing': True,
            'rebalance_interval': 750
        }
        
        config = NestedUpdateConfig.from_dict(data)
        
        assert config.fast_interval == 2
        assert config.medium_interval == 15
        assert config.slow_interval == 150
        assert config.fast_layers == {0, 1, 2}
        assert config.medium_layers == {3, 4, 5}
        assert config.slow_layers == {6, 7}
        assert config.warmup_steps == 150
        assert config.dynamic_rebalancing is True
        assert config.rebalance_interval == 750

    def test_config_from_dict_defaults(self):
        """Test configuration from dict with missing values uses defaults."""
        data = {'fast_interval': 5}
        
        config = NestedUpdateConfig.from_dict(data)
        
        assert config.fast_interval == 5
        assert config.medium_interval == 10  # Default
        assert config.slow_interval == 100  # Default
        assert config.warmup_steps == 100  # Default


# ============================================================================
# Test UpdateStats
# ============================================================================

class TestUpdateStats:
    """Test suite for UpdateStats dataclass."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = UpdateStats(layer_idx=0, group=UpdateGroup.FAST)
        
        assert stats.layer_idx == 0
        assert stats.group == UpdateGroup.FAST
        assert stats.updates == 0
        assert stats.skipped == 0
        assert stats.last_update_step == 0
        assert stats.avg_gradient_norm == 0.0

    def test_stats_to_dict(self):
        """Test statistics serialization to dict."""
        stats = UpdateStats(
            layer_idx=5,
            group=UpdateGroup.MEDIUM,
            updates=100,
            skipped=50,
            last_update_step=150,
            avg_gradient_norm=0.5
        )
        stats_dict = stats.to_dict()
        
        assert isinstance(stats_dict, dict)
        assert stats_dict['layer_idx'] == 5
        assert stats_dict['group'] == 'medium'
        assert stats_dict['updates'] == 100
        assert stats_dict['skipped'] == 50
        assert stats_dict['avg_gradient_norm'] == 0.5


# ============================================================================
# Test UpdateGroup Enum
# ============================================================================

class TestUpdateGroup:
    """Test suite for UpdateGroup enum."""

    def test_group_values(self):
        """Test group enum values."""
        assert UpdateGroup.FAST.value == 'fast'
        assert UpdateGroup.MEDIUM.value == 'medium'
        assert UpdateGroup.SLOW.value == 'slow'
        assert UpdateGroup.FROZEN.value == 'frozen'

    def test_group_comparison(self):
        """Test group comparison."""
        assert UpdateGroup.FAST != UpdateGroup.MEDIUM
        assert UpdateGroup.SLOW == UpdateGroup.SLOW


# ============================================================================
# Test NestedUpdateScheduler Initialization
# ============================================================================

class TestNestedUpdateSchedulerInitialization:
    """Test suite for scheduler initialization."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        scheduler = NestedUpdateScheduler()
        
        assert scheduler.config is not None
        assert scheduler._step == 0
        assert isinstance(scheduler._layer_groups, dict)
        assert isinstance(scheduler._update_stats, dict)
        assert isinstance(scheduler._gradient_history, dict)
        assert isinstance(scheduler._lock, threading.RLock)

    def test_initialization_with_num_layers(self):
        """Test initialization with num_layers for auto-assignment."""
        scheduler = NestedUpdateScheduler(num_layers=10)
        
        # Should have auto-assigned layers
        assert len(scheduler._layer_groups) == 10

    def test_initialization_with_config(self, custom_config):
        """Test initialization with custom config."""
        scheduler = NestedUpdateScheduler(custom_config, num_layers=7)
        
        assert scheduler.config == custom_config
        # Should have assigned layers from config
        assert 0 in scheduler._layer_groups
        assert 1 in scheduler._layer_groups
        assert 5 in scheduler._layer_groups

    def test_auto_assign_layers(self):
        """Test auto-assignment of layers to groups."""
        scheduler = NestedUpdateScheduler(num_layers=10)
        
        # Check distribution (20% fast, 60% medium, 20% slow)
        fast_count = sum(1 for g in scheduler._layer_groups.values() if g == UpdateGroup.FAST)
        medium_count = sum(1 for g in scheduler._layer_groups.values() if g == UpdateGroup.MEDIUM)
        slow_count = sum(1 for g in scheduler._layer_groups.values() if g == UpdateGroup.SLOW)
        
        assert fast_count == 2  # 20% of 10
        assert medium_count == 6  # 60% of 10
        assert slow_count == 2  # 20% of 10

    def test_auto_assign_layers_uneven(self):
        """Test auto-assignment with uneven distribution."""
        scheduler = NestedUpdateScheduler(num_layers=5)
        
        # Should still assign all layers
        assert len(scheduler._layer_groups) == 5


# ============================================================================
# Test should_update Logic
# ============================================================================

class TestShouldUpdate:
    """Test suite for should_update logic."""

    def test_should_update_fast_group(self):
        """Test should_update for fast group (every step)."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config)
        
        # Fast group should update every step
        for step in range(10):
            assert scheduler.should_update(0, step) is True

    def test_should_update_medium_group(self):
        """Test should_update for medium group (every 10 steps)."""
        config = NestedUpdateConfig(medium_layers={0}, medium_interval=10)
        scheduler = NestedUpdateScheduler(config)
        
        # Medium group should update every 10 steps
        assert scheduler.should_update(0, step=0) is True
        assert scheduler.should_update(0, step=5) is False
        assert scheduler.should_update(0, step=10) is True
        assert scheduler.should_update(0, step=20) is True

    def test_should_update_slow_group(self):
        """Test should_update for slow group (every 100 steps)."""
        config = NestedUpdateConfig(slow_layers={0}, slow_interval=100)
        scheduler = NestedUpdateScheduler(config)
        
        # Slow group should update every 100 steps
        assert scheduler.should_update(0, step=0) is True
        assert scheduler.should_update(0, step=50) is False
        assert scheduler.should_update(0, step=100) is True
        assert scheduler.should_update(0, step=200) is True

    def test_should_update_frozen_group(self):
        """Test should_update for frozen group (never)."""
        scheduler = NestedUpdateScheduler()
        scheduler.set_group(0, UpdateGroup.FROZEN)
        
        # Frozen layer should never update
        for step in range(10):
            assert scheduler.should_update(0, step) is False

    def test_should_update_during_warmup(self):
        """Test that all layers update during warmup."""
        config = NestedUpdateConfig(
            slow_layers={0},
            slow_interval=100,
            warmup_steps=50
        )
        scheduler = NestedUpdateScheduler(config)
        
        # During warmup, even slow layers should update
        for step in range(50):
            assert scheduler.should_update(0, step) is True

    def test_should_update_after_warmup(self):
        """Test update pattern after warmup."""
        config = NestedUpdateConfig(
            slow_layers={0},
            slow_interval=100,
            warmup_steps=50
        )
        scheduler = NestedUpdateScheduler(config)
        
        # After warmup, slow interval applies
        assert scheduler.should_update(0, step=100) is True
        assert scheduler.should_update(0, step=150) is False
        assert scheduler.should_update(0, step=200) is True

    def test_should_update_unassigned_layer(self):
        """Test should_update for unassigned layer."""
        scheduler = NestedUpdateScheduler(num_layers=5)
        
        # Clear assignment for layer 0
        del scheduler._layer_groups[0]
        
        # Should return True with warning
        assert scheduler.should_update(0, step=0) is True

    def test_should_update_updates_stats(self):
        """Test that should_update updates statistics."""
        config = NestedUpdateConfig(fast_layers={0}, medium_layers={1})
        scheduler = NestedUpdateScheduler(config)
        
        # Fast layer should always update
        scheduler.should_update(0, step=100)
        stats = scheduler.get_stats(0)
        assert stats['updates'] >= 1
        
        # Medium layer updates every 10 steps
        scheduler.should_update(1, step=5)  # Should skip
        stats = scheduler.get_stats(1)
        assert stats['skipped'] >= 1

    def test_should_update_uses_internal_step(self):
        """Test that should_update uses internal step counter when not provided."""
        config = NestedUpdateConfig(medium_layers={0}, medium_interval=10)
        scheduler = NestedUpdateScheduler(config)
        
        scheduler._step = 10
        result = scheduler.should_update(0)
        assert result is True


# ============================================================================
# Test Step and Scheduling
# ============================================================================

class TestStepAndScheduling:
    """Test suite for step method and scheduling."""

    def test_step_increments_counter(self):
        """Test that step increments internal counter."""
        scheduler = NestedUpdateScheduler()
        
        assert scheduler._step == 0
        scheduler.step()
        assert scheduler._step == 1
        scheduler.step()
        assert scheduler._step == 2

    def test_get_update_layers(self):
        """Test getting list of layers to update."""
        config = NestedUpdateConfig(
            fast_layers={0, 1},
            medium_layers={2, 3},
            slow_layers={4}
        )
        scheduler = NestedUpdateScheduler(config)
        
        # At step 0, all should update
        layers = scheduler.get_update_layers(step=0)
        assert 0 in layers
        assert 1 in layers
        assert 2 in layers
        
        # At step 5, only fast should update
        layers = scheduler.get_update_layers(step=105)  # After warmup
        assert 0 in layers  # Fast
        assert 4 not in layers  # Slow

    def test_step_thread_safety(self):
        """Test that step is thread-safe."""
        scheduler = NestedUpdateScheduler()
        errors = []
        
        def step_worker():
            try:
                for _ in range(100):
                    scheduler.step()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=step_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert scheduler._step == 400


# ============================================================================
# Test Dynamic Rebalancing
# ============================================================================

class TestDynamicRebalancing:
    """Test suite for dynamic rebalancing."""

    def test_rebalancing_triggered(self, scheduler_with_rebalancing):
        """Test that rebalancing is triggered at correct interval."""
        scheduler = scheduler_with_rebalancing
        
        # Record gradient norms to create activity data
        for i in range(10):
            scheduler.record_gradient(0, 1.0)
            scheduler.record_gradient(1, 0.5)
            scheduler.record_gradient(2, 0.3)
            scheduler.record_gradient(3, 0.1)
            scheduler.record_gradient(4, 0.05)
            scheduler.record_gradient(5, 0.01)
        
        # Step past rebalance interval (10) + warmup (5)
        for _ in range(16):
            scheduler.step()
        
        # Rebalancing should have occurred
        # Most active layer (0) should be in FAST group
        assert scheduler.get_group(0) == UpdateGroup.FAST

    def test_rebalancing_not_during_warmup(self):
        """Test that rebalancing doesn't occur during warmup."""
        config = NestedUpdateConfig(
            dynamic_rebalancing=True,
            rebalance_interval=5,
            warmup_steps=100
        )
        scheduler = NestedUpdateScheduler(config, num_layers=4)
        
        # Step multiple times during warmup
        for _ in range(50):
            scheduler.step()
        
        # Rebalancing should not have occurred
        # Groups should remain as initially assigned

    def test_rebalancing_changes_groups(self):
        """Test that rebalancing can change layer groups."""
        config = NestedUpdateConfig(
            fast_layers={0},
            medium_layers={1, 2, 3, 4},
            slow_layers={5},
            dynamic_rebalancing=True,
            rebalance_interval=10,
            warmup_steps=5
        )
        scheduler = NestedUpdateScheduler(config, num_layers=6)
        
        # Record high activity for slow layer
        for _ in range(20):
            scheduler.record_gradient(5, 10.0)  # Very high activity
            scheduler.record_gradient(0, 0.1)   # Low activity
        
        # Step past rebalance interval
        for _ in range(15):
            scheduler.step()
        
        # Groups may have changed based on activity

    def test_record_gradient(self):
        """Test recording gradient norms."""
        scheduler = NestedUpdateScheduler(num_layers=3)
        
        scheduler.record_gradient(0, 1.5)
        scheduler.record_gradient(0, 2.0)
        scheduler.record_gradient(0, 1.0)
        
        # Check that history is maintained
        assert len(scheduler._gradient_history[0]) == 3
        
        # Check that stats are updated
        stats = scheduler.get_stats(0)
        assert stats['avg_gradient_norm'] == 1.5  # Average of [1.5, 2.0, 1.0]

    def test_record_gradient_max_history(self):
        """Test that gradient history is limited."""
        scheduler = NestedUpdateScheduler(num_layers=1)
        
        # Record more gradients than max history
        for i in range(150):
            scheduler.record_gradient(0, float(i))
        
        # History should be limited
        assert len(scheduler._gradient_history[0]) <= 100


# ============================================================================
# Test Group Management
# ============================================================================

class TestGroupManagement:
    """Test suite for group management."""

    def test_get_group(self):
        """Test getting group for a layer."""
        config = NestedUpdateConfig(fast_layers={0}, medium_layers={1})
        scheduler = NestedUpdateScheduler(config)
        
        assert scheduler.get_group(0) == UpdateGroup.FAST
        assert scheduler.get_group(1) == UpdateGroup.MEDIUM

    def test_get_group_unassigned(self):
        """Test getting group for unassigned layer."""
        scheduler = NestedUpdateScheduler()
        
        # Unassigned layer should return MEDIUM as default
        assert scheduler.get_group(0) == UpdateGroup.MEDIUM

    def test_set_group(self):
        """Test setting group for a layer."""
        scheduler = NestedUpdateScheduler()
        
        scheduler.set_group(0, UpdateGroup.FAST)
        assert scheduler.get_group(0) == UpdateGroup.FAST
        
        scheduler.set_group(0, UpdateGroup.SLOW)
        assert scheduler.get_group(0) == UpdateGroup.SLOW

    def test_set_group_creates_stats(self):
        """Test that set_group creates stats if not exists."""
        scheduler = NestedUpdateScheduler()
        
        scheduler.set_group(5, UpdateGroup.FAST)
        
        assert 5 in scheduler._update_stats
        assert scheduler._update_stats[5].group == UpdateGroup.FAST

    def test_set_group_updates_stats(self):
        """Test that set_group updates existing stats."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config)
        
        original_stats = scheduler._update_stats[0]
        scheduler.set_group(0, UpdateGroup.SLOW)
        
        assert scheduler._update_stats[0].group == UpdateGroup.SLOW
        assert scheduler._update_stats[0] is original_stats  # Same object

    def test_freeze_layer(self):
        """Test freezing a layer."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config)
        
        scheduler.freeze_layer(0)
        
        assert scheduler.get_group(0) == UpdateGroup.FROZEN
        assert scheduler.should_update(0, step=1000) is False

    def test_unfreeze_layer(self):
        """Test unfreezing a layer."""
        scheduler = NestedUpdateScheduler()
        
        scheduler.freeze_layer(0)
        assert scheduler.get_group(0) == UpdateGroup.FROZEN
        
        scheduler.unfreeze_layer(0, group=UpdateGroup.MEDIUM)
        assert scheduler.get_group(0) == UpdateGroup.MEDIUM

    def test_unfreeze_layer_default_group(self):
        """Test unfreezing a layer with default group."""
        scheduler = NestedUpdateScheduler()
        
        scheduler.freeze_layer(0)
        scheduler.unfreeze_layer(0)
        
        assert scheduler.get_group(0) == UpdateGroup.MEDIUM


# ============================================================================
# Test Statistics and Reporting
# ============================================================================

class TestStatisticsAndReporting:
    """Test suite for statistics and reporting."""

    def test_get_stats_single_layer(self):
        """Test getting stats for single layer."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config)
        
        # Simulate some updates
        scheduler.should_update(0, step=0)
        scheduler.should_update(0, step=1)
        scheduler.should_update(0, step=2)
        
        stats = scheduler.get_stats(0)
        
        assert isinstance(stats, dict)
        assert stats['layer_idx'] == 0
        assert stats['group'] == 'fast'
        assert stats['updates'] >= 3

    def test_get_stats_all_layers(self):
        """Test getting stats for all layers."""
        config = NestedUpdateConfig(fast_layers={0}, medium_layers={1})
        scheduler = NestedUpdateScheduler(config)
        
        # Simulate updates
        scheduler.should_update(0, step=0)
        scheduler.should_update(1, step=0)
        
        stats = scheduler.get_stats()
        
        assert isinstance(stats, dict)
        assert 'step' in stats
        assert 'total_layers' in stats
        assert 'layer_stats' in stats
        assert stats['total_layers'] == 2

    def test_get_stats_nonexistent_layer(self):
        """Test getting stats for non-existent layer."""
        scheduler = NestedUpdateScheduler()
        
        stats = scheduler.get_stats(999)
        
        assert stats == {}

    def test_get_compute_savings(self):
        """Test computing computational savings."""
        config = NestedUpdateConfig(
            fast_layers={0, 1},
            medium_layers={2, 3, 4, 5},
            slow_layers={6, 7}
        )
        scheduler = NestedUpdateScheduler(config, num_layers=8)
        
        savings = scheduler.get_compute_savings()
        
        assert isinstance(savings, float)
        assert 0.0 <= savings <= 1.0

    def test_get_compute_savings_no_layers(self):
        """Test compute savings with no layers."""
        scheduler = NestedUpdateScheduler()
        
        savings = scheduler.get_compute_savings()
        
        assert savings == 0.0

    def test_export_schedule(self):
        """Test exporting update schedule."""
        config = NestedUpdateConfig(
            fast_layers={0},
            medium_layers={1},
            slow_layers={2}
        )
        scheduler = NestedUpdateScheduler(config)
        
        schedule = scheduler.export_schedule(total_steps=20)
        
        assert isinstance(schedule, dict)
        assert len(schedule) == 20
        
        # Step 0 should have all layers
        assert len(schedule[0]) == 3
        
        # Step 105 (after warmup) should have only fast
        schedule_after_warmup = scheduler.export_schedule(total_steps=110)
        assert 0 in schedule_after_warmup[105]  # Fast updates every step


# ============================================================================
# Test Reset
# ============================================================================

class TestReset:
    """Test suite for reset functionality."""

    def test_reset(self):
        """Test resetting scheduler state."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config)
        
        # Simulate some activity
        for step in range(10):
            scheduler.should_update(0, step)
            scheduler.record_gradient(0, 1.0)
        
        scheduler.step()
        scheduler.step()
        
        # Reset
        scheduler.reset()
        
        # Check state is reset
        assert scheduler._step == 0
        assert len(scheduler._gradient_history) == 0
        
        stats = scheduler.get_stats(0)
        assert stats['updates'] == 0
        assert stats['skipped'] == 0


# ============================================================================
# Test Callbacks
# ============================================================================

class TestCallbacks:
    """Test suite for callbacks."""

    def test_register_pre_update_callback(self):
        """Test registering pre-update callback."""
        scheduler = NestedUpdateScheduler()
        
        callback_called = []
        
        def callback():
            callback_called.append(True)
        
        scheduler.register_pre_update_callback(callback)
        
        assert callback in scheduler._pre_update_callbacks

    def test_register_post_update_callback(self):
        """Test registering post-update callback."""
        scheduler = NestedUpdateScheduler()
        
        callback_called = []
        
        def callback():
            callback_called.append(True)
        
        scheduler.register_post_update_callback(callback)
        
        assert callback in scheduler._post_update_callbacks


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_get_nested_scheduler(self):
        """Test get_nested_scheduler convenience function."""
        scheduler = get_nested_scheduler(
            num_layers=12,
            fast_ratio=0.2,
            medium_ratio=0.6,
            slow_ratio=0.2,
            warmup_steps=50
        )
        
        assert isinstance(scheduler, NestedUpdateScheduler)
        assert scheduler.num_layers == 12
        assert scheduler.config.warmup_steps == 50
        
        # Check group distribution
        fast_count = sum(1 for g in scheduler._layer_groups.values() if g == UpdateGroup.FAST)
        slow_count = sum(1 for g in scheduler._layer_groups.values() if g == UpdateGroup.SLOW)
        
        assert fast_count == 2  # 20% of 12
        assert slow_count == 2  # 20% of 12

    def test_get_nested_scheduler_default_ratios(self):
        """Test get_nested_scheduler with default ratios."""
        scheduler = get_nested_scheduler(num_layers=10)
        
        assert scheduler.num_layers == 10

    def test_create_attention_focused_scheduler(self):
        """Test create_attention_focused_scheduler convenience function."""
        scheduler = create_attention_focused_scheduler(
            num_layers=12,
            attention_layer_indices=[0, 2, 4, 6, 8, 10]
        )
        
        assert isinstance(scheduler, NestedUpdateScheduler)
        
        # Attention layers should be in fast group
        for idx in [0, 2, 4, 6, 8, 10]:
            assert scheduler.get_group(idx) == UpdateGroup.FAST
        
        # Other layers should be in slow group
        for idx in [1, 3, 5, 7, 9, 11]:
            assert scheduler.get_group(idx) == UpdateGroup.SLOW


# ============================================================================
# Test Thread Safety
# ============================================================================

class TestThreadSafety:
    """Test suite for thread safety."""

    def test_concurrent_should_update(self):
        """Test concurrent should_update calls."""
        config = NestedUpdateConfig(fast_layers={0, 1, 2}, medium_layers={3, 4, 5})
        scheduler = NestedUpdateScheduler(config)
        
        errors = []
        
        def worker():
            try:
                for step in range(50):
                    for layer in range(6):
                        scheduler.should_update(layer, step)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_stats_access(self):
        """Test concurrent stats access."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config)
        
        errors = []
        
        def worker():
            try:
                for _ in range(50):
                    scheduler.should_update(0, 0)
                    scheduler.get_stats(0)
                    scheduler.get_stats()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_layer(self):
        """Test scheduler with single layer."""
        config = NestedUpdateConfig(fast_layers={0})
        scheduler = NestedUpdateScheduler(config, num_layers=1)
        
        assert scheduler.num_layers == 1
        assert scheduler.get_group(0) == UpdateGroup.FAST

    def test_two_layers(self):
        """Test scheduler with only two layers."""
        scheduler = NestedUpdateScheduler(num_layers=2)
        
        assert scheduler.num_layers == 2
        assert len(scheduler._layer_groups) == 2

    def test_very_large_num_layers(self):
        """Test scheduler with very large number of layers."""
        scheduler = NestedUpdateScheduler(num_layers=100)
        
        assert scheduler.num_layers == 100
        assert len(scheduler._layer_groups) == 100

    def test_interval_of_one(self):
        """Test with all intervals set to 1."""
        config = NestedUpdateConfig(
            fast_interval=1,
            medium_interval=1,
            slow_interval=1,
            fast_layers={0},
            medium_layers={1},
            slow_layers={2}
        )
        scheduler = NestedUpdateScheduler(config)
        
        # All layers should update every step
        for step in range(5):
            assert scheduler.should_update(0, step)
            assert scheduler.should_update(1, step)
            assert scheduler.should_update(2, step)

    def test_overlapping_layer_assignments(self):
        """Test with overlapping layer assignments (last wins)."""
        config = NestedUpdateConfig(
            fast_layers={0, 1},
            medium_layers={1, 2},  # Layer 1 in both
            slow_layers={2, 3}     # Layer 2 in both
        )
        scheduler = NestedUpdateScheduler(config)
        
        # Last assignment wins
        assert scheduler.get_group(0) == UpdateGroup.FAST
        assert scheduler.get_group(1) == UpdateGroup.MEDIUM
        assert scheduler.get_group(2) == UpdateGroup.SLOW
        assert scheduler.get_group(3) == UpdateGroup.SLOW

    def test_zero_warmup(self):
        """Test with zero warmup steps."""
        config = NestedUpdateConfig(
            slow_layers={0},
            slow_interval=10,
            warmup_steps=0
        )
        scheduler = NestedUpdateScheduler(config)
        
        # Should respect interval from step 0
        assert scheduler.should_update(0, step=0) is True
        assert scheduler.should_update(0, step=5) is False
        assert scheduler.should_update(0, step=10) is True

    def test_very_long_warmup(self):
        """Test with very long warmup."""
        config = NestedUpdateConfig(
            slow_layers={0},
            warmup_steps=10000
        )
        scheduler = NestedUpdateScheduler(config)
        
        # All steps should update during warmup
        for step in [0, 100, 1000, 9999]:
            assert scheduler.should_update(0, step) is True

    def test_all_layers_same_group(self):
        """Test when all layers are in same group."""
        config = NestedUpdateConfig(fast_layers=set(range(10)))
        scheduler = NestedUpdateScheduler(config, num_layers=10)
        
        for layer in range(10):
            assert scheduler.get_group(layer) == UpdateGroup.FAST

    def test_empty_groups(self):
        """Test with all groups empty (auto-assignment)."""
        scheduler = NestedUpdateScheduler(num_layers=10)
        
        # All layers should be auto-assigned
        assert len(scheduler._layer_groups) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

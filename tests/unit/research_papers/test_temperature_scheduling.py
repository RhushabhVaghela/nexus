"""
Comprehensive Unit Tests for Temperature Scheduling (Paper 2601.15394).

Tests cover:
- Linear decay
- Cosine decay
- Exponential decay
- Step-based updates
- Backward compatibility (fixed temp)
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.training_methods import (
    TemperatureSchedule,
    TrainingMethodConfig,
    TrainingMethod,
    get_training_config,
    get_distillation_config_with_schedule
)


class TestConstantSchedule:
    """Test constant temperature schedule."""
    
    def test_constant_temperature(self):
        """Test constant temperature schedule returns fixed value."""
        temp = TemperatureSchedule.constant(temperature=2.0)
        assert temp == 2.0
    
    def test_constant_temperature_different_values(self):
        """Test constant schedule with various temperature values."""
        for temp_val in [1.0, 2.0, 5.0, 10.0]:
            result = TemperatureSchedule.constant(temperature=temp_val)
            assert result == temp_val


class TestLinearSchedule:
    """Test linear temperature decay schedule."""
    
    def test_linear_start(self):
        """Test linear schedule at start."""
        temp = TemperatureSchedule.linear(0, 1000, 5.0, 1.0)
        assert temp == 5.0
    
    def test_linear_end(self):
        """Test linear schedule at end."""
        temp = TemperatureSchedule.linear(1000, 1000, 5.0, 1.0)
        assert temp == 1.0
    
    def test_linear_middle(self):
        """Test linear schedule at middle."""
        temp = TemperatureSchedule.linear(500, 1000, 5.0, 1.0)
        assert temp == 3.0
    
    def test_linear_beyond_end(self):
        """Test linear schedule beyond end stays at final."""
        temp = TemperatureSchedule.linear(1500, 1000, 5.0, 1.0)
        assert temp == 1.0


class TestCosineSchedule:
    """Test cosine temperature decay schedule."""
    
    def test_cosine_start(self):
        """Test cosine schedule at start."""
        temp = TemperatureSchedule.cosine(0, 1000, 5.0, 1.0)
        assert abs(temp - 5.0) < 0.001
    
    def test_cosine_end(self):
        """Test cosine schedule at end."""
        temp = TemperatureSchedule.cosine(1000, 1000, 5.0, 1.0)
        assert abs(temp - 1.0) < 0.001
    
    def test_cosine_middle(self):
        """Test cosine schedule at middle."""
        temp = TemperatureSchedule.cosine(500, 1000, 5.0, 1.0)
        assert temp > 1.0
        assert temp < 5.0


class TestExponentialSchedule:
    """Test exponential temperature decay schedule."""
    
    def test_exponential_start(self):
        """Test exponential schedule at start."""
        temp = TemperatureSchedule.exponential(0, 1000, 5.0, 1.0, 0.95)
        assert abs(temp - 5.0) < 0.001
    
    def test_exponential_end(self):
        """Test exponential schedule at end."""
        temp = TemperatureSchedule.exponential(1000, 1000, 5.0, 1.0, 0.95)
        assert abs(temp - 1.0) < 0.001


class TestGetSchedule:
    """Test schedule retrieval by name."""
    
    def test_get_constant_schedule(self):
        """Test getting constant schedule by name."""
        schedule = TemperatureSchedule.get_schedule("constant")
        assert schedule == TemperatureSchedule.constant
    
    def test_get_linear_schedule(self):
        """Test getting linear schedule by name."""
        schedule = TemperatureSchedule.get_schedule("linear")
        assert schedule == TemperatureSchedule.linear
    
    def test_get_unknown_schedule_defaults_to_constant(self):
        """Test that unknown schedule name defaults to constant."""
        schedule = TemperatureSchedule.get_schedule("unknown")
        assert schedule == TemperatureSchedule.constant


class TestTrainingMethodConfig:
    """Test suite for TrainingMethodConfig with temperature scheduling."""
    
    def test_backward_compatibility_fixed_temp(self):
        """Test that fixed temperature still works (backward compatible)."""
        config = TrainingMethodConfig(
            method=TrainingMethod.DISTILLATION,
            description="Test",
            use_distillation=True,
            temperature=2.0,
            use_temperature_schedule=False
        )
        
        temp = config.get_temperature(current_step=0, total_steps=100)
        assert temp == 2.0
    
    def test_temperature_schedule_linear(self):
        """Test linear temperature schedule integration."""
        config = TrainingMethodConfig(
            method=TrainingMethod.DISTILLATION,
            description="Test",
            use_distillation=True,
            use_temperature_schedule=True,
            temperature_schedule="linear",
            initial_temperature=5.0,
            final_temperature=1.0
        )
        
        temp_start = config.get_temperature(0, 1000)
        assert temp_start == 5.0
        
        temp_end = config.get_temperature(1000, 1000)
        assert temp_end == 1.0


class TestGetTrainingConfig:
    """Test suite for get_training_config with temperature scheduling."""
    
    def test_distillation_without_schedule(self):
        """Test getting distillation config without scheduling."""
        config = get_training_config(TrainingMethod.DISTILLATION)
        
        assert config.method == TrainingMethod.DISTILLATION
        assert config.use_distillation == True
        assert config.use_temperature_schedule == False
    
    def test_distillation_with_linear_schedule(self):
        """Test getting distillation config with linear schedule."""
        config = get_training_config(
            TrainingMethod.DISTILLATION,
            use_temperature_schedule=True,
            temperature_schedule="linear",
            initial_temperature=5.0,
            final_temperature=1.0
        )
        
        assert config.use_temperature_schedule == True
        assert config.temperature_schedule == "linear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for temperature scheduling (Paper 2601.15394).
"""

import pytest
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training_methods import (
    TemperatureSchedule, 
    TrainingMethodConfig, 
    TrainingMethod,
    get_training_config,
    get_distillation_config_with_schedule
)


class TestTemperatureSchedule:
    """Test suite for TemperatureSchedule."""
    
    def test_constant_schedule(self):
        """Test constant temperature schedule."""
        temp = TemperatureSchedule.constant(temperature=2.0)
        assert temp == 2.0
    
    def test_linear_schedule(self):
        """Test linear temperature decay."""
        # Start
        temp_start = TemperatureSchedule.linear(0, 1000, 5.0, 1.0)
        assert temp_start == 5.0
        
        # End
        temp_end = TemperatureSchedule.linear(1000, 1000, 5.0, 1.0)
        assert temp_end == 1.0
        
        # Middle
        temp_mid = TemperatureSchedule.linear(500, 1000, 5.0, 1.0)
        assert temp_mid == 3.0
        
        # Beyond end
        temp_beyond = TemperatureSchedule.linear(1500, 1000, 5.0, 1.0)
        assert temp_beyond == 1.0
    
    def test_cosine_schedule(self):
        """Test cosine temperature decay."""
        # Start
        temp_start = TemperatureSchedule.cosine(0, 1000, 5.0, 1.0)
        assert abs(temp_start - 5.0) < 0.001
        
        # End
        temp_end = TemperatureSchedule.cosine(1000, 1000, 5.0, 1.0)
        assert abs(temp_end - 1.0) < 0.001
        
        # Middle should be closer to final than linear
        temp_mid = TemperatureSchedule.cosine(500, 1000, 5.0, 1.0)
        assert temp_mid > 1.0
        assert temp_mid < 5.0
    
    def test_exponential_schedule(self):
        """Test exponential temperature decay."""
        # Start
        temp_start = TemperatureSchedule.exponential(0, 1000, 5.0, 1.0, 0.95)
        assert abs(temp_start - 5.0) < 0.001
        
        # End
        temp_end = TemperatureSchedule.exponential(1000, 1000, 5.0, 1.0, 0.95)
        assert abs(temp_end - 1.0) < 0.001
    
    def test_get_schedule(self):
        """Test getting schedule by name."""
        assert TemperatureSchedule.get_schedule("constant") == TemperatureSchedule.constant
        assert TemperatureSchedule.get_schedule("linear") == TemperatureSchedule.linear
        assert TemperatureSchedule.get_schedule("cosine") == TemperatureSchedule.cosine
        assert TemperatureSchedule.get_schedule("exponential") == TemperatureSchedule.exponential
        
        # Unknown schedule defaults to constant
        assert TemperatureSchedule.get_schedule("unknown") == TemperatureSchedule.constant


class TestTrainingMethodConfig:
    """Test suite for TrainingMethodConfig with temperature scheduling."""
    
    def test_backward_compatibility(self):
        """Test that fixed temperature still works (backward compatible)."""
        config = TrainingMethodConfig(
            method=TrainingMethod.DISTILLATION,
            description="Test",
            use_distillation=True,
            temperature=2.0,
            use_temperature_schedule=False
        )
        
        # Should return fixed temperature
        temp = config.get_temperature(current_step=0, total_steps=100)
        assert temp == 2.0
        
        temp = config.get_temperature(current_step=50, total_steps=100)
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
        
        # Start
        temp_start = config.get_temperature(0, 1000)
        assert temp_start == 5.0
        
        # End
        temp_end = config.get_temperature(1000, 1000)
        assert temp_end == 1.0
    
    def test_temperature_schedule_cosine(self):
        """Test cosine temperature schedule integration."""
        config = TrainingMethodConfig(
            method=TrainingMethod.DISTILLATION,
            description="Test",
            use_distillation=True,
            use_temperature_schedule=True,
            temperature_schedule="cosine",
            initial_temperature=5.0,
            final_temperature=1.0
        )
        
        # Start
        temp_start = config.get_temperature(0, 1000)
        assert abs(temp_start - 5.0) < 0.001
        
        # End
        temp_end = config.get_temperature(1000, 1000)
        assert abs(temp_end - 1.0) < 0.001
    
    def test_to_dict_includes_schedule(self):
        """Test that serialization includes schedule settings."""
        config = TrainingMethodConfig(
            method=TrainingMethod.DISTILLATION,
            description="Test",
            use_distillation=True,
            use_temperature_schedule=True,
            temperature_schedule="cosine",
            initial_temperature=5.0,
            final_temperature=1.0
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["use_temperature_schedule"] == True
        assert config_dict["temperature_schedule"] == "cosine"
        assert config_dict["initial_temperature"] == 5.0
        assert config_dict["final_temperature"] == 1.0


class TestGetTrainingConfig:
    """Test suite for get_training_config with temperature scheduling."""
    
    def test_distillation_without_schedule(self):
        """Test getting distillation config without scheduling."""
        config = get_training_config(TrainingMethod.DISTILLATION)
        
        assert config.method == TrainingMethod.DISTILLATION
        assert config.use_distillation == True
        assert config.use_temperature_schedule == False
        assert config.temperature == 2.0  # Legacy default
    
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
        assert config.initial_temperature == 5.0
        assert config.final_temperature == 1.0
    
    def test_distillation_with_cosine_schedule(self):
        """Test getting distillation config with cosine schedule."""
        config = get_distillation_config_with_schedule(
            schedule_type="cosine",
            initial_temp=5.0,
            final_temp=1.0
        )
        
        assert config.use_temperature_schedule == True
        assert config.temperature_schedule == "cosine"
        assert config.initial_temperature == 5.0
        assert config.final_temperature == 1.0
        assert config.use_distillation == True
    
    def test_schedule_on_non_distillation_method(self):
        """Test that scheduling only applies to distillation."""
        config = get_training_config(
            TrainingMethod.SFT,
            use_temperature_schedule=True
        )
        
        # Scheduling should only apply to distillation
        assert config.use_distillation == False


class TestScheduleValues:
    """Test specific schedule value calculations."""
    
    def test_linear_values_over_training(self):
        """Test linear schedule values at various points."""
        config = get_distillation_config_with_schedule("linear", 5.0, 1.0)
        
        total_steps = 1000
        expected_values = [
            (0, 5.0),
            (250, 4.0),
            (500, 3.0),
            (750, 2.0),
            (1000, 1.0),
        ]
        
        for step, expected in expected_values:
            temp = config.get_temperature(step, total_steps)
            assert abs(temp - expected) < 0.001, f"Step {step}: expected {expected}, got {temp}"
    
    def test_cosine_monotonic_decrease(self):
        """Test that cosine schedule decreases monotonically."""
        config = get_distillation_config_with_schedule("cosine", 5.0, 1.0)
        
        total_steps = 1000
        prev_temp = config.get_temperature(0, total_steps)
        
        for step in range(100, 1001, 100):
            temp = config.get_temperature(step, total_steps)
            assert temp <= prev_temp, f"Temperature should decrease: step {step}"
            prev_temp = temp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Integration Tests for Training with Temperature Scheduling (Paper 2601.15394).

Tests cover:
- Full training loop with scheduling
- Verify temperature changes over time
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.training_methods import (
    TemperatureSchedule,
    TrainingMethodConfig,
    TrainingMethod,
    get_training_config,
    get_distillation_config_with_schedule,
    create_distillation_callback
)


class TestTrainingLoopWithScheduling:
    """Test training loop integration with temperature scheduling."""
    
    def test_linear_schedule_over_training(self):
        """Test linear schedule over simulated training."""
        config = get_distillation_config_with_schedule("linear", 5.0, 1.0)
        
        total_steps = 1000
        temperatures = []
        
        for step in range(0, total_steps + 1, 100):
            temp = config.get_temperature(step, total_steps)
            temperatures.append((step, temp))
        
        # Verify temperatures decrease
        for i in range(1, len(temperatures)):
            assert temperatures[i][1] <= temperatures[i-1][1]
        
        # Verify start and end
        assert temperatures[0][1] == 5.0
        assert temperatures[-1][1] == 1.0
    
    def test_cosine_schedule_over_training(self):
        """Test cosine schedule over simulated training."""
        config = get_distillation_config_with_schedule("cosine", 5.0, 1.0)
        
        total_steps = 1000
        temperatures = []
        
        for step in range(0, total_steps + 1, 100):
            temp = config.get_temperature(step, total_steps)
            temperatures.append((step, temp))
        
        # Verify temperatures decrease monotonically
        for i in range(1, len(temperatures)):
            assert temperatures[i][1] <= temperatures[i-1][1]
        
        # Verify start and end
        assert abs(temperatures[0][1] - 5.0) < 0.001
        assert abs(temperatures[-1][1] - 1.0) < 0.001
    
    def test_exponential_schedule_over_training(self):
        """Test exponential schedule over simulated training."""
        config = get_distillation_config_with_schedule("exponential", 5.0, 1.0)
        
        total_steps = 1000
        temperatures = []
        
        for step in range(0, total_steps + 1, 100):
            temp = config.get_temperature(step, total_steps)
            temperatures.append((step, temp))
        
        # Verify temperatures decrease
        for i in range(1, len(temperatures)):
            assert temperatures[i][1] <= temperatures[i-1][1]
        
        # Verify start and end
        assert abs(temperatures[0][1] - 5.0) < 0.001
        assert abs(temperatures[-1][1] - 1.0) < 0.001
    
    def test_distillation_callback_creation(self):
        """Test creation of distillation callback."""
        config = get_distillation_config_with_schedule("linear", 5.0, 1.0)
        
        callback = create_distillation_callback(config)
        
        # Test callback at different steps
        temp_0 = callback(0, 1000)
        temp_500 = callback(500, 1000)
        temp_1000 = callback(1000, 1000)
        
        assert temp_0 == 5.0
        assert temp_500 == 3.0
        assert temp_1000 == 1.0


class TestTemperatureScheduleVariations:
    """Test various temperature schedule configurations."""
    
    def test_different_temperature_ranges(self):
        """Test schedules with different temperature ranges."""
        ranges = [
            (10.0, 1.0),
            (5.0, 0.5),
            (3.0, 1.0),
            (2.0, 0.8)
        ]
        
        for initial, final in ranges:
            config = get_distillation_config_with_schedule("linear", initial, final)
            
            temp_start = config.get_temperature(0, 1000)
            temp_end = config.get_temperature(1000, 1000)
            
            assert temp_start == initial
            assert temp_end == final
    
    def test_backward_compatibility_no_schedule(self):
        """Test backward compatibility without scheduling."""
        config = get_training_config(TrainingMethod.DISTILLATION)
        
        # Should use fixed temperature
        temp_0 = config.get_temperature(0, 1000)
        temp_500 = config.get_temperature(500, 1000)
        temp_1000 = config.get_temperature(1000, 1000)
        
        # All should be the same (fixed temperature)
        assert temp_0 == temp_500 == temp_1000
        assert temp_0 == 2.0  # Default temperature


class TestScheduleComparisons:
    """Compare different schedule types."""
    
    def test_linear_vs_cosine_at_midpoint(self):
        """Compare linear vs cosine schedules at midpoint."""
        linear_config = get_distillation_config_with_schedule("linear", 5.0, 1.0)
        cosine_config = get_distillation_config_with_schedule("cosine", 5.0, 1.0)
        
        total_steps = 1000
        step = 500
        
        linear_temp = linear_config.get_temperature(step, total_steps)
        cosine_temp = cosine_config.get_temperature(step, total_steps)
        
        # Linear should be at 3.0, cosine should be higher
        assert linear_temp == 3.0
        assert cosine_temp > linear_temp
    
    def test_all_schedules_end_at_same_temperature(self):
        """Verify all schedules end at the same temperature."""
        schedules = ["linear", "cosine", "exponential"]
        total_steps = 1000
        
        for schedule_type in schedules:
            config = get_distillation_config_with_schedule(schedule_type, 5.0, 1.0)
            final_temp = config.get_temperature(total_steps, total_steps)
            assert abs(final_temp - 1.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

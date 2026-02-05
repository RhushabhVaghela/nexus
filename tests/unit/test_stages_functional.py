import pytest
import torch
from unittest.mock import MagicMock, patch
from src.stages.base import BaseStage
from src.stages.stage_reasoning import ReasoningStage
from src.stages.stage_podcast import PodcastStage
from src.stages.stage_omni import OmniTrainingStage

class MockConfig:
    def __init__(self):
        self.model_name_or_path = "mock-model"
        self.dataset_name = "mock-dataset"
        self.output_dir = "mock-output"
        self.per_device_train_batch_size = 1
        self.learning_rate = 1e-5
        self.num_train_epochs = 1
        self.max_steps = 10
        self.push_to_hub = False
        self.report_to = "none"
        self.capability_name = "mock_capability"

def test_base_stage_initialization():
    config = MockConfig()
    class ConcreteStage(BaseStage):
        def prepare(self): pass
        def train(self): pass
    stage = ConcreteStage(config)
    assert stage.config == config

def test_reasoning_stage_setup():
    config = MockConfig()
    config.capability_name = "reasoning"
    with patch("stages.stage_reasoning.ReasoningStage"):
        stage = ReasoningStage(config)
        assert stage.config.capability_name == "reasoning"
        # Verify it has standard methods
        assert hasattr(stage, "run")

def test_podcast_stage_logic():
    config = MockConfig()
    config.capability_name = "podcast"
    with patch("stages.stage_podcast.PodcastStage"):
        stage = PodcastStage(config)
        assert stage.config.capability_name == "podcast"

def test_omni_stage_initialization():
    config = MockConfig()
    config.capability_name = "omni"
    stage = OmniTrainingStage(config)
    assert stage.config.capability_name == "omni"
    assert hasattr(stage, "run")

@pytest.mark.parametrize("stage_class", [ReasoningStage, PodcastStage, OmniTrainingStage])
def test_stage_name_consistency(stage_class):
    config = MockConfig()
    config.capability_name = "cons_test"
    # Mocking dependencies for classes that might trigger them on init
    with patch("stages.stage_reasoning.ReasoningStage"), \
         patch("stages.stage_podcast.PodcastStage"):
        stage = stage_class(config)
        assert isinstance(stage.config.capability_name, str)
        assert len(stage.config.capability_name) > 0

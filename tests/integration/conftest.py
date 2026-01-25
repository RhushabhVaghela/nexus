import pytest
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

@pytest.fixture(scope="session")
def encoders_config():
    path = PROJECT_ROOT / "configs" / "encoders.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def decoders_config():
    path = PROJECT_ROOT / "configs" / "decoders.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def datasets_config():
    path = PROJECT_ROOT / "configs" / "datasets.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def outputs_config():
    path = PROJECT_ROOT / "configs" / "outputs.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

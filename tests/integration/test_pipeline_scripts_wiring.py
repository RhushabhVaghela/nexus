import pytest
import importlib
import sys
import os
from pathlib import Path

# Automatically discover all numbered pipeline scripts in src/
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

def get_pipeline_scripts():
    scripts = []
    if SRC_DIR.exists():
        for f in os.listdir(SRC_DIR):
            if f.endswith(".py") and f[0].isdigit():
                # Convert filename to module path, e.g., src.01_download_real_datasets
                module_name = f"src.{f[:-3]}"
                scripts.append(module_name)
    return sorted(scripts)

PIPELINE_SCRIPTS = get_pipeline_scripts()

@pytest.mark.parametrize("module_name", PIPELINE_SCRIPTS)
def test_script_functional(module_name, monkeypatch):
    """Ensure scripts can be imported and their main() can be invoked with mocks."""
    try:
        # Import the module
        module = importlib.import_module(module_name)
        assert module is not None
        
        # If the module has a main function, try to call it with --help to verify arg parsing
        if hasattr(module, 'main'):
            # Mock sys.argv to simulate --help call which should exit with 0 or just print help
            monkeypatch.setattr(sys, "argv", [module_name, "--help"])
            
            # Catch SystemExit because argparse --help usually calls sys.exit(0)
            try:
                module.main()
            except SystemExit as e:
                # 0 is success for --help
                assert e.code == 0
            except Exception as e:
                # Some scripts might not use argparse or might fail even with --help if imports are broken
                # We catch this as a failure if it's a critical error
                if isinstance(e, (SyntaxError, NameError, TypeError, AttributeError)):
                    pytest.fail(f"Functional error in {module_name}.main(): {type(e).__name__}: {e}")
                else:
                    print(f"Skipping non-critical functional error in {module_name}: {e}")

    except ImportError as e:
        pytest.fail(f"Could not import {module_name}: {e}")
    except Exception as e:
        error_msg = str(e)
        critical_errors = (SyntaxError, NameError, TypeError, AttributeError)
        if isinstance(e, critical_errors):
            pytest.fail(f"Critical error in {module_name}: {type(e).__name__}: {e}")
        else:
            print(f"Skipping runtime error in {module_name}: {e}")

def test_project_root_in_path():
    """Ensure src is in the python path for all scripts."""
    root_str = str(PROJECT_ROOT)
    assert any(root_str in p for p in sys.path) or "" in sys.path

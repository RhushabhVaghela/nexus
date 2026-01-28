import unittest
import os
import sys
import subprocess

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "scripts/nexus_pipeline.py")
STATE_FILE = os.path.join(BASE_DIR, ".pipeline_state.json")

class TestPipelineConfig(unittest.TestCase):
    def setUp(self):
        # Backup existing state to avoid 'current state: done' skipping
        self.state_backup = None
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                self.state_backup = f.read()
            os.remove(STATE_FILE)

    def tearDown(self):
        # Restore state
        if self.state_backup:
            with open(STATE_FILE, 'w') as f:
                f.write(self.state_backup)
        elif os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    def test_parameter_propagation_dry_run(self):
        """
        Integration test: Verify that custom parameters are correctly passed 
        through the orchestrator to child scripts in dry-run mode.
        """
        # Define some custom "beast mode" parameters
        custom_epochs = "3"
        custom_lr = "7e-6"
        custom_router_epochs = "12"
        custom_router_lr = "3e-4"
        custom_embed = "sentence-transformers/all-mpnet-base-v2"
        
        cmd = [
            sys.executable, PIPELINE_SCRIPT,
            "--dry-run",
            "--epochs", custom_epochs,
            "--lr", custom_lr,
            "--router_epochs", custom_router_epochs,
            "--router_lr", custom_router_lr,
            "--embedding_model", custom_embed
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.path.join(BASE_DIR, 'src')}:{env.get('PYTHONPATH', '')}"
        
        print(f"\n[Test] Running pipeline dry-run with: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR, env=env)
        
        output = result.stdout
        
        if result.returncode != 0:
            print(f"[Test Error] Return code: {result.returncode}")
            print(f"[Test Error] Stderr: {result.stderr}")
            
        # Verify propagation in logs (run_command prints the cmd it WOULD execute)
        try:
            self.assertIn(f"--epochs {custom_epochs}", output)
            self.assertIn(f"--lr {custom_lr}", output)
            self.assertIn(f"--epochs {custom_router_epochs}", output)
            self.assertIn(f"--lr {custom_router_lr}", output)
            self.assertIn(f"--embedding_model '{custom_embed}'", output)
        except AssertionError as e:
            print(f"[Test Error] Output was: \n{output}")
            print(f"[Test Error] Stderr was: \n{result.stderr}")
            raise e
        
        print("[Test] ✅ Parameter propagation verified.")

if __name__ == "__main__":
    unittest.main()

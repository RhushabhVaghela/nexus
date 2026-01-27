import unittest
import os
import sys
import subprocess
from src.nexus_core.config import NexusConfig

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestNIWTPipelineIntegration(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV
        self.csv_path = "integration_test_models.csv"
        with open(self.csv_path, "w") as f:
            f.write("Model Name,Parameters,Category,Best Feature\n")
            f.write("MockModel,1B,Test,TestFeature\n")
            
        pass

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_pipeline_orchestration_script_exists(self):
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/run_niwt_pipeline.py'))
        self.assertTrue(os.path.exists(script_path), "Orchestration script not found")

if __name__ == '__main__':
    unittest.main()

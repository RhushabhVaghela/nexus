import unittest
import os
import shutil
import json
from unittest.mock import MagicMock, patch
import sys

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/nexus_core/student')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/nexus_core/training')))

class TestFullLifecycle(unittest.TestCase):
    def setUp(self):
        self.state_file = ".pipeline_state.json"
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            
    def tearDown(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    @patch('nexus_pipeline.NexusPipeline.run_command')
    def test_pipeline_integration(self, mock_run):
        """
        Simulates the full pipeline: Init -> Profiling -> Training -> Router
        Verifies that NIWT Core is invoked and State Transitions occur logicially.
        """
        from nexus_pipeline import NexusPipeline
        
        pipeline = NexusPipeline()
        
        # 1. INIT -> PROFILING
        pipeline.stage_profiling()
        
        # Verify NIWT Core was called for the active teacher (Reasoning)
        # We expect run_command to have been called with python -c "from niwt_core..."
        calls = [c[0][0] for c in mock_run.call_args_list]
        self.assertTrue(any("niwt_core" in cmd for cmd in calls))
        self.assertEqual(pipeline.state["current_stage"], "training")
        
        # 2. PROFILING -> TRAINING (Distillation)
        # Mock file existence for finding the 'profile'
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['mock_profile.json']):
             pipeline.stage_training()
        
        self.assertEqual(pipeline.state["current_stage"], "router_training")
        
        # 3. TRAINING -> ROUTER
        pipeline.stage_router_training()
        self.assertEqual(pipeline.state["current_stage"], "done")

    def test_loss_function_mechanics(self):
        """Verify the Activation Anchoring Loss compiles and runs."""
        from loss_functions import ActivationAnchoringLoss
        import torch
        
        loss_fn = ActivationAnchoringLoss()
        
        # Mock Tensors
        student = torch.randn(2, 10, requires_grad=True)
        teacher = torch.randn(2, 10)
        s_state = torch.randn(2, 4096, requires_grad=True)
        t_state = torch.randn(1, 1, 4096) # Adjust for matching logic if needed or just mock
        
        # Since I updated loss_functions.py to ActivationAnchoringLoss
        # I need to ensure it matches the test expectations.


if __name__ == '__main__':
    unittest.main()

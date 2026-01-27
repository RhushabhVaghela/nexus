import unittest
import torch
import numpy as np
import shutil
import tempfile
import os
from src.nexus_final.profiler import StreamingPCAProfiler

class TestNIWTProfiler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_intrinsic_dimension_calculation(self):
        profiler = StreamingPCAProfiler("mock", [], self.test_dir)
        
        # Case 1: Variance decays quickly (Low Rank)
        # [0.8, 0.1, 0.05, 0.01, ...] -> 0.9 @ idx 1 (2 components)
        var_ratio = np.array([0.8, 0.15, 0.04, 0.01])
        dim = profiler._calculate_intrinsic_dimension(var_ratio, threshold=0.95)
        self.assertEqual(dim, 2) # 0.8+0.15 = 0.95
        
        # Case 2: Variance spread out (High Rank)
        var_ratio_flat = np.array([0.1]*10) # Sum=1.0
        dim_flat = profiler._calculate_intrinsic_dimension(var_ratio_flat, threshold=0.95)
        self.assertEqual(dim_flat, 10)

    def test_perturbation_analysis_logic(self):
        # We need to mock model/tokenizer/dataset to test this without downloading weights
        pass 

if __name__ == '__main__':
    unittest.main()

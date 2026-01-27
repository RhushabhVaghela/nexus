import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from nexus_final.optimization_suite import ThermalWatchdog, GradNormMonitor, SynergyMonitor, compute_optimal_batch_size

class TestOptimizationSuite(unittest.TestCase):
    
    def test_batch_sizing(self):
        self.assertEqual(compute_optimal_batch_size(16000), 16) # 16GB -> 16
        self.assertEqual(compute_optimal_batch_size(9000), 8)   # 9GB -> 8
        self.assertEqual(compute_optimal_batch_size(4000), 4)   # 4GB -> 4

    def test_synergy_monitor(self):
        monitor = SynergyMonitor()
        
        # Case 1: Student Correct, Teacher Wrong -> Synergy
        monitor.record("42", "The answer is 42", False)
        self.assertEqual(monitor.synergy_count, 1)
        
        # Case 2: Student Correct, Teacher Correct -> No Synergy
        monitor.record("42", "42", True)
        self.assertEqual(monitor.synergy_count, 1)
        
        # Case 3: Student Wrong -> No Synergy
        monitor.record("42", "Wrong", False)
        self.assertEqual(monitor.synergy_count, 1)
        
        report = monitor.get_report()
        self.assertEqual(report['count'], 1)

    def test_grad_norm_monitor(self):
        monitor = GradNormMonitor()
        
        # Healthy ratio (Task=1.0, Anchor=1.0 -> 1.0)
        self.assertTrue(monitor.check_health(1.0, 1.0))
        
        # Masking (Task=0.001, Anchor=1.0 -> 0.001)
        self.assertFalse(monitor.check_health(0.001, 1.0))

if __name__ == '__main__':
    unittest.main()

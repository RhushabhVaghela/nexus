"""
Benchmarks Package
Comprehensive benchmark suite for Nexus multimodal model.

Includes:
- performance_benchmark.py: Core performance benchmarks
- optimization_benchmark.py: Optimization technique benchmarks
- training_benchmark.py: Training performance benchmarks
- memory_benchmark.py: Memory usage benchmarks
"""

__version__ = "1.0.0"
__author__ = "Nexus Team"

from .performance_benchmark import *
from .optimization_benchmark import *
from .training_benchmark import *
from .memory_benchmark import *

__all__ = [
    "TestTokenThroughput",
    "TestLatencyBenchmarks",
    "TestMemoryUsage",
    "TestEndToEndInference",
    "TestStreamingPerformance",
    "TestConcurrentRequests",
    "TestLayerPipelining",
    "TestLayerSkipping",
    "TestSemiAutoregressiveGeneration",
    "TestCompressionOptimization",
    "TestTrainingThroughput",
    "TestGradientAccumulation",
    "TestCheckpointPerformance",
    "TestMixedPrecision",
    "TestDataLoaderPerformance",
    "TestPeakMemoryUsage",
    "TestActivationMemory",
    "TestGradientCheckpointing",
    "TestMemoryOptimization",
    "TestGPUMemorySpecific",
]

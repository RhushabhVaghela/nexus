"""
Nexus Benchmarks — evaluation suites for measuring student model quality.

Provides configurable benchmark runners, fullstack evaluation,
repetition-detection benchmarks, and a RULER long-context benchmark suite.
"""

from .fullstack_eval import FullstackEval
from .lovable_benchmark import LovableBenchmark
from .expanded_eval_suite import ExpandedEvalSuite, BENCHMARK_REGISTRY

# ---------------------------------------------------------------------------
# Lazy imports for benchmark modules
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # benchmark_runner.py — configurable benchmark runner
    "BenchmarkConfig": (".benchmark_runner", "BenchmarkConfig"),
    "BenchmarkRunner": (".benchmark_runner", "BenchmarkRunner"),
    # benchmark_repetition.py — repetition-specific benchmarks
    "RepetitionBenchmark": (".benchmark_repetition", "RepetitionBenchmark"),
    # ruler_benchmark.py — RULER benchmark suite
    "RULERConfig": (".ruler_benchmark", "RULERConfig"),
    "RULERResult": (".ruler_benchmark", "RULERResult"),
    "RULERBenchmark": (".ruler_benchmark", "RULERBenchmark"),
    "run_ruler_benchmark": (".ruler_benchmark", "run_ruler_benchmark"),
    # ruler_tasks.py — RULER task implementations
    "TaskCategory": (".ruler_tasks", "TaskCategory"),
    "TaskSample": (".ruler_tasks", "TaskSample"),
    "RULERTask": (".ruler_tasks", "RULERTask"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FullstackEval",
    "LovableBenchmark",
    "ExpandedEvalSuite",
    "BENCHMARK_REGISTRY",
] + list(_LAZY_IMPORTS.keys())

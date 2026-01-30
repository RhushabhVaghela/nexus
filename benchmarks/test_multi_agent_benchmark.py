#!/usr/bin/env python3
"""
Multi-Agent Orchestration Benchmark Suite

Comprehensive benchmarks for multi-agent system performance:
- Agent initialization time
- Planning agent response time
- Backend agent code generation latency
- Frontend agent code generation latency
- Full workflow execution time
- Concurrent agent execution
- Context passing overhead
- Retry mechanism performance

Usage:
    python benchmarks/test_multi_agent_benchmark.py --all
    python benchmarks/test_multi_agent_benchmark.py --category initialization
    python benchmarks/test_multi_agent_benchmark.py --num-agents 5 10 20
    python benchmarks/test_multi_agent_benchmark.py --memory-profiling
    python benchmarks/test_multi_agent_benchmark.py --output results/multi_agent_benchmark.json

Environment Variables:
    BENCHMARK_ITERATIONS: Number of iterations (default: 100)
    BENCHMARK_WARMUP: Number of warmup iterations (default: 10)
    OPENAI_API_KEY: API key for LLM benchmarks (optional)
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
import statistics
import tracemalloc
import gc
import asyncio
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import numpy as np

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    iterations: int
    total_time: float
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    memory_delta_mb: Optional[float] = None
    throughput: Optional[float] = None  # ops/sec
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    timestamp: str
    python_version: str
    platform: str
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "categories": list(set(r.category for r in self.results))
            }
        }
    
    def save(self, path: str):
        """Save benchmark results to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"📊 Benchmark results saved to {path}")


class BenchmarkRunner:
    """Runner for executing benchmarks with timing and memory profiling."""
    
    def __init__(self, iterations: int = 100, warmup: int = 10):
        self.iterations = iterations
        self.warmup = warmup
        self.results: List[BenchmarkResult] = []
    
    def run(
        self,
        name: str,
        category: str,
        func: Callable,
        *args,
        memory_profile: bool = False,
        calculate_throughput: bool = False,
        throughput_items: int = 1,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark function and collect timing statistics.
        
        Args:
            name: Benchmark name
            category: Benchmark category
            func: Function to benchmark
            args: Positional arguments for func
            memory_profile: Whether to profile memory usage
            calculate_throughput: Whether to calculate throughput
            throughput_items: Number of items processed per iteration
            metadata: Additional metadata to store
            kwargs: Keyword arguments for func
        """
        times = []
        memory_delta = None
        
        # Warmup iterations
        for _ in range(self.warmup):
            result = func(*args, **kwargs)
        
        # Actual benchmark iterations
        if memory_profile:
            gc.collect()
            tracemalloc.start()
            start_mem = tracemalloc.get_traced_memory()[0]
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        if memory_profile:
            end_mem = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_delta = (end_mem - start_mem) / (1024 * 1024)  # MB
        
        # Calculate statistics
        times.sort()
        mean_time = statistics.mean(times)
        throughput = (throughput_items / mean_time) if calculate_throughput else None
        
        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.iterations,
            total_time=sum(times),
            mean_time=mean_time,
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            p95_time=times[int(len(times) * 0.95)],
            p99_time=times[int(len(times) * 0.99)],
            memory_delta_mb=memory_delta,
            throughput=throughput,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: BenchmarkResult):
        """Print a benchmark result in a formatted way."""
        print(f"\n{'='*60}")
        print(f"📊 {result.name}")
        print(f"{'='*60}")
        print(f"Category: {result.category}")
        print(f"Iterations: {result.iterations}")
        print(f"Total Time: {result.total_time:.4f}s")
        print(f"Mean: {result.mean_time*1000:.4f}ms")
        print(f"Median: {result.median_time*1000:.4f}ms")
        print(f"Std Dev: {result.std_dev*1000:.4f}ms")
        print(f"Min: {result.min_time*1000:.4f}ms")
        print(f"Max: {result.max_time*1000:.4f}ms")
        print(f"P95: {result.p95_time*1000:.4f}ms")
        print(f"P99: {result.p99_time*1000:.4f}ms")
        if result.memory_delta_mb is not None:
            print(f"Memory Delta: {result.memory_delta_mb:.4f} MB")
        if result.throughput is not None:
            print(f"Throughput: {result.throughput:.2f} ops/sec")


class MockAgent:
    """Mock agent for benchmarking."""
    
    def __init__(self, name: str, agent_type: str, delay_ms: float = 10.0):
        self.name = name
        self.agent_type = agent_type
        self.delay_ms = delay_ms
        self.context = {}
        self.message_history = []
        self.retry_count = 0
        self.max_retries = 3
    
    def initialize(self):
        """Initialize the agent."""
        time.sleep(self.delay_ms / 1000.0)  # Simulate initialization
        self.context["initialized"] = True
        return True
    
    def process(self, input_data: Dict) -> Dict:
        """Process input and return output."""
        time.sleep(self.delay_ms / 1000.0)  # Simulate processing
        
        # Simulate different agent behaviors
        if self.agent_type == "planning":
            return {"plan": self._generate_plan(input_data)}
        elif self.agent_type == "backend":
            return {"code": self._generate_backend_code(input_data)}
        elif self.agent_type == "frontend":
            return {"code": self._generate_frontend_code(input_data)}
        elif self.agent_type == "review":
            return {"review": self._generate_review(input_data)}
        else:
            return {"output": f"Processed by {self.name}"}
    
    def _generate_plan(self, input_data: Dict) -> List[Dict]:
        """Generate a plan for the task."""
        task = input_data.get("task", "")
        complexity = input_data.get("complexity", "medium")
        
        if complexity == "simple":
            return [{"step": 1, "action": "execute", "agent": "backend"}]
        elif complexity == "medium":
            return [
                {"step": 1, "action": "design", "agent": "planning"},
                {"step": 2, "action": "implement_backend", "agent": "backend"},
                {"step": 3, "action": "implement_frontend", "agent": "frontend"},
            ]
        else:  # complex
            return [
                {"step": 1, "action": "analyze", "agent": "planning"},
                {"step": 2, "action": "design_architecture", "agent": "planning"},
                {"step": 3, "action": "implement_backend", "agent": "backend"},
                {"step": 4, "action": "implement_frontend", "agent": "frontend"},
                {"step": 5, "action": "review", "agent": "review"},
            ]
    
    def _generate_backend_code(self, input_data: Dict) -> str:
        """Generate backend code."""
        task = input_data.get("task", "")
        # Simulate code generation delay based on complexity
        code_lines = len(task) // 10
        time.sleep(code_lines * 0.001)  # 1ms per 10 chars
        return f"# Backend code for: {task}\ndef main():\n    pass"
    
    def _generate_frontend_code(self, input_data: Dict) -> str:
        """Generate frontend code."""
        task = input_data.get("task", "")
        code_lines = len(task) // 10
        time.sleep(code_lines * 0.001)
        return f"// Frontend code for: {task}\nfunction Component() {{\n    return null;\n}}"
    
    def _generate_review(self, input_data: Dict) -> Dict:
        """Generate code review."""
        return {"issues": [], "score": 0.95}
    
    def pass_context(self, context: Dict):
        """Receive context from another agent."""
        self.context.update(context)
        return True


class MockMultiAgentSystem:
    """Mock multi-agent orchestration system."""
    
    def __init__(self):
        self.agents: Dict[str, MockAgent] = {}
        self.workflow_history = []
        self.context_store = {}
    
    def add_agent(self, agent: MockAgent):
        """Add an agent to the system."""
        self.agents[agent.name] = agent
    
    def initialize_all(self):
        """Initialize all agents."""
        for agent in self.agents.values():
            agent.initialize()
    
    def execute_workflow(self, workflow: List[Dict], initial_input: Dict) -> Dict:
        """Execute a workflow across multiple agents."""
        context = initial_input.copy()
        results = []
        
        for step in workflow:
            agent_name = step.get("agent")
            action = step.get("action")
            
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                agent.pass_context(context)
                result = agent.process({"task": action, **context})
                context.update(result)
                results.append(result)
        
        return {"results": results, "final_context": context}
    
    def execute_concurrent(self, tasks: List[Tuple[str, Dict]]) -> List[Dict]:
        """Execute tasks concurrently across agents."""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = []
            for agent_name, input_data in tasks:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    future = executor.submit(agent.process, input_data)
                    futures.append((agent_name, future))
            
            for agent_name, future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append({"agent": agent_name, "result": result})
                except Exception as e:
                    results.append({"agent": agent_name, "error": str(e)})
        
        return results


class AgentInitializationBenchmarks:
    """Benchmarks for agent initialization."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_single_agent_init(self):
        """Benchmark initialization of a single agent."""
        print("\n" + "="*60)
        print("🚀 SINGLE AGENT INITIALIZATION")
        print("="*60)
        
        agent_types = [
            ("planning", 50),
            ("backend", 30),
            ("frontend", 30),
            ("review", 20),
        ]
        
        for agent_type, delay_ms in agent_types:
            def init_agent():
                agent = MockAgent(f"test_{agent_type}", agent_type, delay_ms)
                return agent.initialize()
            
            result = self.runner.run(
                f"init_single_{agent_type}",
                "initialization",
                init_agent,
                metadata={"agent_type": agent_type, "expected_delay_ms": delay_ms}
            )
            self.runner.print_result(result)
    
    def benchmark_multi_agent_init(self, agent_counts: List[int]):
        """Benchmark initialization of multiple agents."""
        print("\n" + "="*60)
        print("👥 MULTI-AGENT INITIALIZATION")
        print("="*60)
        
        for num_agents in agent_counts:
            def init_multi():
                system = MockMultiAgentSystem()
                for i in range(num_agents):
                    agent = MockAgent(f"agent_{i}", "generic", delay_ms=10)
                    system.add_agent(agent)
                system.initialize_all()
                return True
            
            result = self.runner.run(
                f"init_multi_{num_agents}_agents",
                "initialization",
                init_multi,
                metadata={"num_agents": num_agents}
            )
            self.runner.print_result(result)


class AgentResponseBenchmarks:
    """Benchmarks for agent response times."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_planning_response(self):
        """Benchmark planning agent response time."""
        print("\n" + "="*60)
        print("📋 PLANNING AGENT RESPONSE")
        print("="*60)
        
        complexities = [
            ("simple", "Create a simple function"),
            ("medium", "Create a REST API with CRUD operations"),
            ("complex", "Create a microservices architecture with authentication, database, caching, and real-time notifications"),
        ]
        
        for complexity, task in complexities:
            planning_agent = MockAgent("planner", "planning", delay_ms=100)
            planning_agent.initialize()
            
            def get_plan():
                return planning_agent.process({"task": task, "complexity": complexity})
            
            result = self.runner.run(
                f"planning_{complexity}",
                "response",
                get_plan,
                metadata={"complexity": complexity, "task_length": len(task)}
            )
            self.runner.print_result(result)
    
    def benchmark_backend_generation(self):
        """Benchmark backend agent code generation."""
        print("\n" + "="*60)
        print("⚙️  BACKEND CODE GENERATION")
        print("="*60)
        
        tasks = [
            ("small", "Create a hello world API"),
            ("medium", "Create a user management API with authentication"),
            ("large", "Create a full e-commerce backend with payments, inventory, and order management"),
        ]
        
        for size, task in tasks:
            backend_agent = MockAgent("backend", "backend", delay_ms=80)
            backend_agent.initialize()
            
            def generate_backend():
                return backend_agent.process({"task": task})
            
            result = self.runner.run(
                f"backend_generation_{size}",
                "generation",
                generate_backend,
                metadata={"size": size, "task_length": len(task)}
            )
            self.runner.print_result(result)
    
    def benchmark_frontend_generation(self):
        """Benchmark frontend agent code generation."""
        print("\n" + "="*60)
        print("🎨 FRONTEND CODE GENERATION")
        print("="*60)
        
        tasks = [
            ("small", "Create a button component"),
            ("medium", "Create a login form with validation"),
            ("large", "Create a dashboard with charts, tables, and real-time updates"),
        ]
        
        for size, task in tasks:
            frontend_agent = MockAgent("frontend", "frontend", delay_ms=90)
            frontend_agent.initialize()
            
            def generate_frontend():
                return frontend_agent.process({"task": task})
            
            result = self.runner.run(
                f"frontend_generation_{size}",
                "generation",
                generate_frontend,
                metadata={"size": size, "task_length": len(task)}
            )
            self.runner.print_result(result)


class WorkflowBenchmarks:
    """Benchmarks for workflow execution."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_full_workflow(self):
        """Benchmark full workflow execution time."""
        print("\n" + "="*60)
        print("🔄 FULL WORKFLOW EXECUTION")
        print("="*60)
        
        workflows = [
            ("simple", [
                {"agent": "planning", "action": "plan"},
                {"agent": "backend", "action": "implement"},
            ]),
            ("standard", [
                {"agent": "planning", "action": "design"},
                {"agent": "backend", "action": "implement_api"},
                {"agent": "frontend", "action": "implement_ui"},
            ]),
            ("complex", [
                {"agent": "planning", "action": "analyze"},
                {"agent": "planning", "action": "design"},
                {"agent": "backend", "action": "implement_core"},
                {"agent": "backend", "action": "implement_auth"},
                {"agent": "frontend", "action": "implement_ui"},
                {"agent": "review", "action": "review"},
            ]),
        ]
        
        for workflow_name, workflow_steps in workflows:
            system = MockMultiAgentSystem()
            
            # Add required agents
            for step in workflow_steps:
                agent_type = step["agent"]
                if agent_type not in [a.name for a in system.agents.values()]:
                    system.add_agent(MockAgent(agent_type, agent_type, delay_ms=50))
            
            system.initialize_all()
            
            def execute_workflow():
                return system.execute_workflow(workflow_steps, {"task": "test workflow"})
            
            result = self.runner.run(
                f"workflow_{workflow_name}_{len(workflow_steps)}steps",
                "workflow",
                execute_workflow,
                metadata={
                    "workflow_name": workflow_name,
                    "num_steps": len(workflow_steps),
                    "agents_involved": len(set(s["agent"] for s in workflow_steps))
                }
            )
            self.runner.print_result(result)
    
    def benchmark_concurrent_execution(self):
        """Benchmark concurrent agent execution."""
        print("\n" + "="*60)
        print("⚡ CONCURRENT AGENT EXECUTION")
        print("="*60)
        
        concurrency_levels = [2, 4, 8, 16]
        
        for num_concurrent in concurrency_levels:
            system = MockMultiAgentSystem()
            
            # Add agents
            for i in range(num_concurrent):
                system.add_agent(MockAgent(f"agent_{i}", "generic", delay_ms=20))
            
            system.initialize_all()
            
            # Create concurrent tasks
            tasks = [(f"agent_{i}", {"task": f"task_{i}"}) for i in range(num_concurrent)]
            
            def execute_concurrent():
                return system.execute_concurrent(tasks)
            
            result = self.runner.run(
                f"concurrent_{num_concurrent}_agents",
                "concurrent",
                execute_concurrent,
                calculate_throughput=True,
                throughput_items=num_concurrent,
                metadata={"num_concurrent": num_concurrent}
            )
            self.runner.print_result(result)


class ContextPassingBenchmarks:
    """Benchmarks for context passing overhead."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_context_passing(self):
        """Benchmark context passing between agents."""
        print("\n" + "="*60)
        print("📤 CONTEXT PASSING OVERHEAD")
        print("="*60)
        
        context_sizes = [
            ("small", {"key": "value"}),
            ("medium", {f"key_{i}": f"value_{i}" * 100 for i in range(100)}),
            ("large", {f"key_{i}": f"value_{i}" * 1000 for i in range(1000)}),
        ]
        
        agent1 = MockAgent("agent1", "generic", delay_ms=0)
        agent2 = MockAgent("agent2", "generic", delay_ms=0)
        
        for size_name, context in context_sizes:
            def pass_context():
                agent2.pass_context(context)
                return True
            
            context_size_kb = len(json.dumps(context)) / 1024
            
            result = self.runner.run(
                f"context_pass_{size_name}",
                "context",
                pass_context,
                metadata={
                    "context_size_kb": context_size_kb,
                    "num_keys": len(context)
                }
            )
            self.runner.print_result(result)
    
    def benchmark_context_buildup(self):
        """Benchmark context buildup over multiple agent handoffs."""
        print("\n" + "="*60)
        print("📈 CONTEXT BUILDUP OVER HANDOFFS")
        print("="*60)
        
        handoff_counts = [2, 5, 10, 20]
        
        for num_handoffs in handoff_counts:
            agents = [MockAgent(f"agent_{i}", "generic", delay_ms=5) for i in range(num_handoffs)]
            context = {"initial": "data"}
            
            def build_context():
                ctx = context.copy()
                for agent in agents:
                    agent.pass_context(ctx)
                    result = agent.process({"task": "add_data"})
                    ctx.update(result)
                return ctx
            
            result = self.runner.run(
                f"context_buildup_{num_handoffs}_handoffs",
                "context",
                build_context,
                metadata={"num_handoffs": num_handoffs}
            )
            self.runner.print_result(result)


class RetryMechanismBenchmarks:
    """Benchmarks for retry mechanism performance."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_retry_overhead(self):
        """Benchmark retry mechanism overhead."""
        print("\n" + "="*60)
        print("🔄 RETRY MECHANISM OVERHEAD")
        print("="*60)
        
        retry_scenarios = [
            ("no_retry", 0, 0),
            ("success_first_try", 0, 0.1),
            ("retry_once", 1, 0.1),
            ("retry_twice", 2, 0.1),
            ("retry_max", 3, 0.1),
        ]
        
        for scenario_name, num_retries, failure_rate in retry_scenarios:
            agent = MockAgent("retry_agent", "generic", delay_ms=20)
            agent.max_retries = 3
            attempt_count = [0]
            
            def execute_with_retry():
                attempt_count[0] = 0
                for attempt in range(agent.max_retries + 1):
                    attempt_count[0] += 1
                    try:
                        # Simulate potential failure
                        if attempt < num_retries and failure_rate > 0:
                            raise Exception("Simulated failure")
                        return agent.process({"task": "test"})
                    except Exception:
                        if attempt < agent.max_retries:
                            time.sleep(0.01)  # Retry delay
                            continue
                        raise
            
            result = self.runner.run(
                f"retry_{scenario_name}",
                "retry",
                execute_with_retry,
                metadata={
                    "scenario": scenario_name,
                    "configured_retries": num_retries,
                    "failure_rate": failure_rate
                }
            )
            self.runner.print_result(result)
    
    def benchmark_retry_backoff_strategies(self):
        """Benchmark different retry backoff strategies."""
        print("\n" + "="*60)
        print("⏱️  RETRY BACKOFF STRATEGIES")
        print("="*60)
        
        backoff_strategies = [
            ("fixed", lambda attempt: 0.01),
            ("linear", lambda attempt: 0.01 * (attempt + 1)),
            ("exponential", lambda attempt: 0.01 * (2 ** attempt)),
        ]
        
        for strategy_name, backoff_fn in backoff_strategies:
            agent = MockAgent("backoff_agent", "generic", delay_ms=10)
            
            def execute_with_backoff():
                for attempt in range(3):
                    try:
                        if attempt < 2:  # Fail first 2 attempts
                            raise Exception("Simulated failure")
                        return agent.process({"task": "test"})
                    except Exception:
                        time.sleep(backoff_fn(attempt))
            
            result = self.runner.run(
                f"backoff_{strategy_name}",
                "retry",
                execute_with_backoff,
                metadata={"backoff_strategy": strategy_name}
            )
            self.runner.print_result(result)


class MemoryBenchmarks:
    """Benchmarks for memory usage during multi-agent operations."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def benchmark_agent_memory(self):
        """Benchmark memory usage per agent."""
        print("\n" + "="*60)
        print("💾 AGENT MEMORY USAGE")
        print("="*60)
        
        agent_counts = [1, 5, 10, 20, 50]
        
        for num_agents in agent_counts:
            def create_agents():
                system = MockMultiAgentSystem()
                for i in range(num_agents):
                    agent = MockAgent(f"agent_{i}", "generic", delay_ms=0)
                    # Add some memory usage
                    agent.context = {f"key_{j}": f"value_{j}" * 100 for j in range(100)}
                    system.add_agent(agent)
                return system
            
            result = self.runner.run(
                f"memory_{num_agents}_agents",
                "memory",
                create_agents,
                memory_profile=True,
                metadata={"num_agents": num_agents}
            )
            self.runner.print_result(result)
    
    def benchmark_workflow_memory(self):
        """Benchmark memory usage during workflow execution."""
        print("\n" + "="*60)
        print("🔄 WORKFLOW MEMORY USAGE")
        print("="*60)
        
        workflow_complexities = [
            ("simple", 2),
            ("medium", 5),
            ("complex", 10),
        ]
        
        for complexity, num_steps in workflow_complexities:
            system = MockMultiAgentSystem()
            for i in range(num_steps):
                system.add_agent(MockAgent(f"agent_{i}", "generic", delay_ms=5))
            system.initialize_all()
            
            workflow = [{"agent": f"agent_{i}", "action": f"step_{i}"} for i in range(num_steps)]
            
            def execute_with_memory():
                return system.execute_workflow(workflow, {"task": "test"})
            
            result = self.runner.run(
                f"workflow_memory_{complexity}",
                "memory",
                execute_with_memory,
                memory_profile=True,
                metadata={"complexity": complexity, "num_steps": num_steps}
            )
            self.runner.print_result(result)


class RegressionBenchmarks:
    """Benchmarks for detecting performance regressions."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
    
    def run_regression_suite(self):
        """Run core regression benchmarks."""
        print("\n" + "="*60)
        print("🔄 REGRESSION BENCHMARK SUITE")
        print("="*60)
        
        # Core multi-agent operations
        baseline_ops = [
            ("agent_init", lambda: MockAgent("test", "generic", 10).initialize()),
            ("agent_process", lambda: MockAgent("test", "generic", 10).process({"task": "test"})),
            ("context_pass", lambda: MockAgent("test", "generic").pass_context({"key": "value"})),
        ]
        
        for name, func in baseline_ops:
            result = self.runner.run(
                f"regression_{name}",
                "regression",
                func,
                metadata={"baseline_op": name}
            )
            self.runner.print_result(result)


def run_all_benchmarks(args) -> BenchmarkSuite:
    """Run all benchmark suites."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    suite = BenchmarkSuite(
        name="Multi-Agent Orchestration Benchmark Suite",
        timestamp=timestamp,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=os.name
    )
    
    # Setup
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", args.iterations))
    warmup = int(os.environ.get("BENCHMARK_WARMUP", args.warmup))
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    
    # Run benchmarks by category
    if args.category in ["all", "initialization"]:
        init_bench = AgentInitializationBenchmarks(runner)
        init_bench.benchmark_single_agent_init()
        init_bench.benchmark_multi_agent_init(args.num_agents)
    
    if args.category in ["all", "response"]:
        response_bench = AgentResponseBenchmarks(runner)
        response_bench.benchmark_planning_response()
        response_bench.benchmark_backend_generation()
        response_bench.benchmark_frontend_generation()
    
    if args.category in ["all", "workflow"]:
        workflow_bench = WorkflowBenchmarks(runner)
        workflow_bench.benchmark_full_workflow()
        workflow_bench.benchmark_concurrent_execution()
    
    if args.category in ["all", "context"]:
        context_bench = ContextPassingBenchmarks(runner)
        context_bench.benchmark_context_passing()
        context_bench.benchmark_context_buildup()
    
    if args.category in ["all", "retry"]:
        retry_bench = RetryMechanismBenchmarks(runner)
        retry_bench.benchmark_retry_overhead()
        retry_bench.benchmark_retry_backoff_strategies()
    
    if args.category in ["all", "memory"]:
        memory_bench = MemoryBenchmarks(runner)
        memory_bench.benchmark_agent_memory()
        memory_bench.benchmark_workflow_memory()
    
    if args.category in ["all", "regression"]:
        regression_bench = RegressionBenchmarks(runner)
        regression_bench.run_regression_suite()
    
    # Collect all results
    for result in runner.results:
        suite.add_result(result)
    
    return suite


def print_summary(suite: BenchmarkSuite):
    """Print a summary of all benchmark results."""
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    # Group by category
    by_category = {}
    for result in suite.results:
        cat = result.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)
    
    for category, results in sorted(by_category.items()):
        print(f"\n{category.upper()}:")
        print("-" * 70)
        for result in results:
            throughput_str = f" ({result.throughput:.1f} ops/sec)" if result.throughput else ""
            print(f"  {result.name:50s} {result.mean_time*1000:>8.4f}ms{throughput_str}")
    
    print("\n" + "="*70)
    print(f"Total Benchmarks: {len(suite.results)}")
    print(f"Categories: {len(by_category)}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestration Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/test_multi_agent_benchmark.py --all
  python benchmarks/test_multi_agent_benchmark.py --category initialization
  python benchmarks/test_multi_agent_benchmark.py --num-agents 5 10 20
  python benchmarks/test_multi_agent_benchmark.py --memory-profiling --output results.json
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--category",
        choices=["all", "initialization", "response", "workflow", "context", "retry", "memory", "regression"],
        default="all",
        help="Benchmark category to run"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Number of agents to benchmark (default: 5 10 20)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--memory-profiling",
        action="store_true",
        help="Enable memory profiling"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for benchmark results (JSON)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Compare against baseline results file"
    )
    
    args = parser.parse_args()
    
    if args.all:
        args.category = "all"
    
    print("="*70)
    print("🚀 Multi-Agent Orchestration Benchmark Suite")
    print("="*70)
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Memory Profiling: {args.memory_profiling}")
    print(f"Category: {args.category}")
    print("="*70)
    
    # Run benchmarks
    suite = run_all_benchmarks(args)
    
    # Print summary
    print_summary(suite)
    
    # Save results
    if args.output:
        suite.save(args.output)
    
    # Compare against baseline if provided
    if args.baseline and os.path.exists(args.baseline):
        print("\n" + "="*70)
        print("📊 BASELINE COMPARISON")
        print("="*70)
        with open(args.baseline, 'r') as f:
            baseline = json.load(f)
        
        baseline_results = {r['name']: r for r in baseline.get('results', [])}
        
        for result in suite.results:
            if result.name in baseline_results:
                baseline_time = baseline_results[result.name]['mean_time']
                current_time = result.mean_time
                delta = ((current_time - baseline_time) / baseline_time) * 100
                
                if abs(delta) > 5:  # >5% change
                    indicator = "🔴" if delta > 0 else "🟢"
                    print(f"{indicator} {result.name:50s} {delta:+.1f}% change")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance Benchmark Suite
Comprehensive performance benchmarks for the Nexus multimodal model.

Covers:
- Token throughput benchmarks
- Memory usage benchmarks
- Latency benchmarks
- End-to-end inference benchmarks
"""

import pytest
import torch
import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nexus.models.omni.inference import OmniInference, GenerationConfig
from nexus.core.training.student_trainer import NexusDistillationTrainer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    tokens_per_second: float
    memory_used_mb: float
    latency_ms: float
    batch_size: int
    max_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "memory_used_mb": self.memory_used_mb,
            "latency_ms": self.latency_ms,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
        }


class TestTokenThroughput:
    """Token throughput benchmarks."""

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_single_token_throughput_short(self, inference, benchmark):
        """Benchmark token throughput for short generation (100 tokens)."""

        def generate_tokens():
            return inference.generate(
                prompt="Explain quantum computing in simple terms.",
                config=GenerationConfig(max_new_tokens=100, temperature=0.7),
            )

        result = benchmark(generate_tokens)
        assert result.tokens_per_second > 0, "Token throughput must be positive"

    @pytest.mark.benchmark
    def test_single_token_throughput_medium(self, inference, benchmark):
        """Benchmark token throughput for medium generation (500 tokens)."""

        def generate_tokens():
            return inference.generate(
                prompt="Write a detailed analysis of machine learning architectures including transformers, CNNs, and RNNs.",
                config=GenerationConfig(max_new_tokens=500, temperature=0.7),
            )

        result = benchmark(generate_tokens)
        assert result.tokens_per_second > 0, "Token throughput must be positive"

    @pytest.mark.benchmark
    def test_single_token_throughput_long(self, inference, benchmark):
        """Benchmark token throughput for long generation (1000 tokens)."""

        def generate_tokens():
            return inference.generate(
                prompt="Write a comprehensive essay on the history of artificial intelligence from 1950 to 2024.",
                config=GenerationConfig(max_new_tokens=1000, temperature=0.7),
            )

        result = benchmark(generate_tokens)
        assert result.tokens_per_second > 0, "Token throughput must be positive"

    @pytest.mark.benchmark
    def test_batch_token_throughput_small_batch(self, inference, benchmark):
        """Benchmark token throughput with batch size 4."""
        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis.",
            "Describe the solar system.",
            "Define machine learning.",
        ]

        def generate_batch():
            return inference.batch_generate(prompts, max_tokens=100)

        result = benchmark(generate_batch)
        # Batch throughput should be higher than single
        assert result.tokens_per_second > 0, "Batch throughput must be positive"

    @pytest.mark.benchmark
    def test_batch_token_throughput_medium_batch(self, inference, benchmark):
        """Benchmark token throughput with batch size 8."""
        prompts = [
            f"Question {i}: Explain the concept of attention mechanisms in neural networks."
            for i in range(8)
        ]

        def generate_batch():
            return inference.batch_generate(prompts, max_tokens=100)

        result = benchmark(generate_batch)
        assert result.tokens_per_second > 0, "Medium batch throughput must be positive"

    @pytest.mark.benchmark
    def test_continuous_generation_throughput(self, inference, benchmark):
        """Benchmark sustained token generation over multiple prompts."""
        prompts = [
            "Define neural networks.",
            "Explain gradient descent.",
            "Describe backpropagation.",
            "What are embeddings?",
            "Explain transformers architecture.",
        ]

        def continuous_generation():
            total_tokens = 0
            for prompt in prompts:
                result = inference.generate(
                    prompt=prompt, config=GenerationConfig(max_new_tokens=200)
                )
                total_tokens += result.token_count
            return total_tokens

        result = benchmark(continuous_generation)
        assert result > 0, "Must generate positive number of tokens"


class TestLatencyBenchmarks:
    """Latency measurement benchmarks."""

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_first_token_latency(self, inference, benchmark):
        """Measure time to first token (TTFT)."""

        def measure_ttft():
            return inference.generate(
                prompt="What is 2+2?",
                config=GenerationConfig(max_new_tokens=50, temperature=0.0),
            )

        result = benchmark(measure_ttft)
        # TTFT should be under 100ms for cached prompts
        assert result.time_to_first_token_ms < 1000, "TTFT should be under 1 second"

    @pytest.mark.benchmark
    def test_inter_token_latency(self, inference, benchmark):
        """Measure inter-token latency (ITL)."""

        def measure_itl():
            result = inference.generate(
                prompt="Write a detailed explanation of computer science.",
                config=GenerationConfig(max_new_tokens=100, temperature=0.7),
            )
            return result

        result = benchmark(measure_itl)
        # Average ITL should be under 50ms
        assert result.avg_inter_token_latency_ms < 100, "ITL should be under 100ms"

    @pytest.mark.benchmark
    def test_prompt_processing_latency(self, inference, benchmark):
        """Measure prompt encoding latency."""

        def measure_prompt_latency():
            # Measure only prompt encoding, not generation
            start = time.perf_counter()
            encoded = inference.encode_prompt("Analyze the impact of AI on healthcare.")
            encoding_time = time.perf_counter() - start
            return encoding_time

        result = benchmark(measure_prompt_latency)
        assert result < 0.5, "Prompt encoding should be under 500ms"

    @pytest.mark.benchmark
    def test_varying_prompt_length_latency(self, inference, benchmark):
        """Measure latency across different prompt lengths."""
        latencies = []

        for length in [10, 50, 100, 500, 1000]:
            prompt = "word " * length

            def test_latency(p=prompt):
                return inference.generate(
                    prompt=p, config=GenerationConfig(max_new_tokens=50)
                )

            result = benchmark(test_latency)
            latencies.append(
                {"prompt_tokens": length, "latency_ms": result.total_latency_ms}
            )

        # Verify latency grows sub-linearly with prompt length
        short_latency = latencies[0]["latency_ms"]
        long_latency = latencies[-1]["latency_ms"]
        length_ratio = 100  # 1000/10 tokens
        latency_ratio = long_latency / short_latency if short_latency > 0 else 1

        assert latency_ratio < length_ratio, (
            "Latency should grow sub-linearly with prompt length"
        )

    @pytest.mark.benchmark
    def test_generation_time_vs_token_count(self, inference, benchmark):
        """Verify generation time scales linearly with token count."""
        times = []

        for tokens in [50, 100, 200, 500]:

            def generate_n_tokens(n=tokens):
                start = time.perf_counter()
                inference.generate(
                    prompt="Explain artificial intelligence.",
                    config=GenerationConfig(max_new_tokens=n, temperature=0.0),
                )
                return time.perf_counter() - start

            result = benchmark(generate_n_tokens)
            times.append({"tokens": tokens, "time": result})

        # Check linear scaling
        times_array = np.array([t["time"] for t in times])
        tokens_array = np.array([t["tokens"] for t in times])

        # Fit linear regression and check R-squared
        coeffs = np.polyfit(tokens_array, times_array, 1)
        predicted = np.polyval(coeffs, tokens_array)
        ss_res = np.sum((times_array - predicted) ** 2)
        ss_tot = np.sum((times_array - np.mean(times_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        assert r_squared > 0.9, "Generation time should scale linearly with token count"


class TestMemoryUsage:
    """Memory usage benchmarks."""

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_baseline_memory_short_generation(self, inference, benchmark):
        """Measure baseline memory for short generation."""

        def generate_short():
            return inference.generate(
                prompt="Hi", config=GenerationConfig(max_new_tokens=10)
            )

        result = benchmark(generate_short)
        assert result.memory_peak_mb < 8192, "Short generation should use less than 8GB"

    @pytest.mark.benchmark
    def test_memory_medium_generation(self, inference, benchmark):
        """Measure memory for medium generation."""

        def generate_medium():
            return inference.generate(
                prompt="Write a detailed essay on climate change.",
                config=GenerationConfig(max_new_tokens=500),
            )

        result = benchmark(generate_medium)
        assert result.memory_peak_mb < 10240, (
            "Medium generation should use less than 10GB"
        )

    @pytest.mark.benchmark
    def test_memory_long_generation(self, inference, benchmark):
        """Measure memory for long generation."""

        def generate_long():
            return inference.generate(
                prompt="Write a comprehensive book chapter on software engineering.",
                config=GenerationConfig(max_new_tokens=1000),
            )

        result = benchmark(generate_long)
        assert result.memory_peak_mb < 12288, (
            "Long generation should use less than 12GB"
        )

    @pytest.mark.benchmark
    def test_batch_memory_small_batch(self, inference, benchmark):
        """Measure memory for small batch."""

        def generate_batch():
            prompts = ["Query 1", "Query 2", "Query 3", "Query 4"]
            return inference.batch_generate(prompts, max_tokens=100)

        result = benchmark(generate_batch)
        assert result.memory_peak_mb < 10240, "Small batch should use less than 10GB"

    @pytest.mark.benchmark
    def test_batch_memory_large_batch(self, inference, benchmark):
        """Measure memory for large batch."""

        def generate_batch():
            prompts = [f"Query {i}" for i in range(16)]
            return inference.batch_generate(prompts, max_tokens=100)

        result = benchmark(generate_batch)
        assert result.memory_peak_mb < 16384, "Large batch should use less than 16GB"

    @pytest.mark.benchmark
    def test_memory_after_cleanup(self, inference, benchmark):
        """Verify memory cleanup after generation."""
        # Generate to warm up
        inference.generate("Warmup prompt.", config=GenerationConfig(max_new_tokens=10))

        initial_memory = inference.get_memory_usage()

        # Generate large output
        inference.generate(
            "Generate lots of text." * 10, config=GenerationConfig(max_new_tokens=500)
        )

        after_memory = inference.get_memory_usage()

        # Should not leak significant memory
        memory_diff = after_memory - initial_memory
        assert memory_diff < 100, "Should not leak more than 100MB"


class TestEndToEndInference:
    """End-to-end inference benchmarks."""

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_simple_question_answering(self, inference, benchmark):
        """End-to-end benchmark for Q&A."""

        def qa_pipeline():
            return inference.generate(
                prompt="What are the main benefits of exercise?",
                config=GenerationConfig(max_new_tokens=200),
            )

        result = benchmark(qa_pipeline)
        assert result.tokens_per_second > 5, "Q&A should achieve reasonable throughput"

    @pytest.mark.benchmark
    def test_code_generation(self, inference, benchmark):
        """End-to-end benchmark for code generation."""

        def code_pipeline():
            return inference.generate(
                prompt="Write a Python function to calculate fibonacci numbers efficiently.",
                config=GenerationConfig(max_new_tokens=300),
            )

        result = benchmark(code_pipeline)
        assert result.tokens_per_second > 3, (
            "Code generation should achieve reasonable throughput"
        )

    @pytest.mark.benchmark
    def test_summarization_pipeline(self, inference, benchmark):
        """End-to-end benchmark for summarization."""
        text = """
        Artificial intelligence (AI) leverages computers and machines to mimic the 
        problem-solving and decision-making capabilities of the human mind. As a 
        constellation of technologies, AI relates to a system's ability to adapt and 
        learn from external data to achieve specific goals. From 1950s symbolic AI to 
        today's deep learning, AI has evolved significantly. Machine learning, a subset 
        of AI, uses statistical techniques to enable computers to learn from data without 
        being explicitly programmed. Deep learning, in turn, uses multiple layers of 
        neural networks to progressively extract higher-level features from raw input.
        """

        def summarize():
            return inference.generate(
                prompt=f"Summarize this text:\n\n{text}",
                config=GenerationConfig(max_new_tokens=100),
            )

        result = benchmark(summarize)
        assert result.tokens_per_second > 10, (
            "Summarization should achieve good throughput"
        )

    @pytest.mark.benchmark
    def test_creative_writing_pipeline(self, inference, benchmark):
        """End-to-end benchmark for creative writing."""

        def creative_pipeline():
            return inference.generate(
                prompt="Write the opening paragraph of a science fiction story.",
                config=GenerationConfig(max_new_tokens=300),
            )

        result = benchmark(creative_pipeline)
        assert result.tokens_per_second > 2, (
            "Creative writing should achieve acceptable throughput"
        )

    @pytest.mark.benchmark
    def test_conversational_pipeline(self, inference, benchmark):
        """End-to-end benchmark for conversational AI."""
        conversation = [
            "Hello! How are you today?",
            "That's great to hear. What do you do for work?",
            "Interesting! How did you get into that field?",
        ]

        def conversation_pipeline():
            responses = []
            for msg in conversation:
                response = inference.generate(
                    prompt=msg, config=GenerationConfig(max_new_tokens=150)
                )
                responses.append(response)
            return responses

        result = benchmark(conversation_pipeline)
        assert len(result) == 3, "Should complete full conversation"

    @pytest.mark.benchmark
    def test_multiple_choice_pipeline(self, inference, benchmark):
        """End-to-end benchmark for multiple choice answering."""
        question = """
        Which of the following is NOT a type of neural network?
        A) Convolutional Neural Network
        B) Recurrent Neural Network  
        C) Transformer
        D) Decision Tree
        E) Graph Neural Network
        """

        def mc_pipeline():
            return inference.generate(
                prompt=f"Solve this multiple choice question:\n{question}",
                config=GenerationConfig(max_new_tokens=50),
            )

        result = benchmark(mc_pipeline)
        assert result.tokens_per_second > 5, "MC answering should be fast"


class TestStreamingPerformance:
    """Streaming performance benchmarks."""

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_streaming_token_rate(self, inference, benchmark):
        """Benchmark streaming token generation rate."""

        def stream_generation():
            tokens = []
            for token in inference.generate_stream(
                prompt="Write about the future of AI.",
                config=GenerationConfig(max_new_tokens=100),
            ):
                tokens.append(token)
            return len(tokens)

        result = benchmark(stream_generation)
        assert result == 100, "Should generate exactly 100 tokens"

    @pytest.mark.benchmark
    def test_streaming_latency(self, inference, benchmark):
        """Benchmark streaming latency per token."""
        latencies = []

        def measure_latencies():
            token_latencies = []
            start_time = time.perf_counter()

            for i, token in enumerate(
                inference.generate_stream(
                    prompt="Explain quantum computing.",
                    config=GenerationConfig(max_new_tokens=50),
                )
            ):
                token_time = time.perf_counter() - start_time
                token_latencies.append(token_time)
                start_time = time.perf_counter()

            return token_latencies

        result = benchmark(measure_latencies)
        avg_latency = np.mean(result)
        assert avg_latency < 0.1, "Average token latency should be under 100ms"


class TestConcurrentRequests:
    """Concurrent request handling benchmarks."""

    @pytest.fixture
    def inference(self):
        """Set up inference engine."""
        try:
            inference = OmniInference(
                model_path="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            yield inference
            del inference
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    @pytest.mark.benchmark
    def test_sequential_requests(self, inference, benchmark):
        """Benchmark sequential request handling."""

        def sequential_requests():
            results = []
            for i in range(5):
                result = inference.generate(
                    prompt=f"Question {i}: What is 1+1?",
                    config=GenerationConfig(max_new_tokens=20),
                )
                results.append(result)
            return results

        result = benchmark(sequential_requests)
        assert len(result) == 5, "Should complete all 5 requests"

    @pytest.mark.benchmark
    def test_request_throughput_over_time(self, inference, benchmark):
        """Benchmark sustained request throughput."""

        def sustained_throughput():
            total_requests = 0
            start = time.perf_counter()

            while time.perf_counter() - start < 10:  # 10 second window
                inference.generate(
                    prompt="Short query.", config=GenerationConfig(max_new_tokens=10)
                )
                total_requests += 1

            return total_requests

        result = benchmark(sustained_throughput)
        assert result >= 5, "Should handle at least 5 requests in 10 seconds"

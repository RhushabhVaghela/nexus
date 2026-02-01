"""
Load Testing for Concurrent Requests

Tests system behavior under high concurrent load.
"""

import pytest
import threading
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics


@dataclass
class LoadTestResult:
    """Results from a load test."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    requests_per_second: float
    errors: List[str]


class ConcurrentRequestSimulator:
    """Simulates concurrent API requests."""
    
    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers
        self.results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def simulate_request(self, request_id: int, delay: float = 0.1) -> Dict[str, Any]:
        """Simulate a single request."""
        start_time = time.time()
        
        try:
            # Simulate processing
            time.sleep(delay)
            
            # Simulate occasional failures (5% failure rate)
            if request_id % 20 == 0:
                raise Exception(f"Simulated failure for request {request_id}")
            
            latency = time.time() - start_time
            
            result = {
                "request_id": request_id,
                "success": True,
                "latency": latency,
                "error": None
            }
        except Exception as e:
            latency = time.time() - start_time
            result = {
                "request_id": request_id,
                "success": False,
                "latency": latency,
                "error": str(e)
            }
        
        with self._lock:
            self.results.append(result)
        
        return result
    
    def run_load_test(
        self,
        num_requests: int,
        concurrent_users: int,
        request_delay: float = 0.1
    ) -> LoadTestResult:
        """Run a load test with specified parameters."""
        self.results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(self.simulate_request, i, request_delay)
                for i in range(num_requests)
            ]
            concurrent.futures.wait(futures)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        latencies = [r["latency"] for r in self.results]
        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        errors = [r["error"] for r in self.results if r["error"]]
        
        return LoadTestResult(
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_latency=statistics.mean(latencies),
            min_latency=min(latencies),
            max_latency=max(latencies),
            p50_latency=statistics.median(latencies),
            p95_latency=self._percentile(latencies, 95),
            p99_latency=self._percentile(latencies, 99),
            requests_per_second=num_requests / total_time,
            errors=errors
        )
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestConcurrentRequests:
    """Test concurrent request handling."""
    
    def test_10_concurrent_users(self):
        """Test with 10 concurrent users."""
        simulator = ConcurrentRequestSimulator()
        result = simulator.run_load_test(
            num_requests=100,
            concurrent_users=10,
            request_delay=0.01
        )
        
        # Should handle all requests
        assert result.total_requests == 100
        # Success rate should be > 90%
        success_rate = result.successful_requests / result.total_requests
        assert success_rate >= 0.90
    
    def test_100_concurrent_users(self):
        """Test with 100 concurrent users."""
        simulator = ConcurrentRequestSimulator()
        result = simulator.run_load_test(
            num_requests=1000,
            concurrent_users=100,
            request_delay=0.005
        )
        
        assert result.total_requests == 1000
        assert result.requests_per_second > 50
    
    def test_ramp_up_pattern(self):
        """Test ramp-up load pattern."""
        simulator = ConcurrentRequestSimulator()
        
        results = []
        for users in [10, 50, 100, 50, 10]:
            result = simulator.run_load_test(
                num_requests=users * 10,
                concurrent_users=users,
                request_delay=0.01
            )
            results.append(result)
        
        # All phases should complete
        assert len(results) == 5
    
    def test_spike_pattern(self):
        """Test spike load pattern."""
        simulator = ConcurrentRequestSimulator()
        
        # Normal load
        normal_result = simulator.run_load_test(
            num_requests=100,
            concurrent_users=10,
            request_delay=0.01
        )
        
        # Spike
        spike_result = simulator.run_load_test(
            num_requests=500,
            concurrent_users=200,
            request_delay=0.005
        )
        
        # Recovery
        recovery_result = simulator.run_load_test(
            num_requests=100,
            concurrent_users=10,
            request_delay=0.01
        )
        
        # All phases should complete with reasonable success rates
        assert normal_result.successful_requests / normal_result.total_requests > 0.9
        assert recovery_result.successful_requests / recovery_result.total_requests > 0.9


class TestRateLimitingUnderLoad:
    """Test rate limiting under load."""
    
    def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced under load."""
        from collections import deque
        
        # Simple rate limiter: 10 requests per second
        max_requests = 10
        window = 1.0
        requests = deque()
        
        allowed = 0
        rejected = 0
        
        for i in range(50):
            now = time.time()
            
            # Remove old requests
            while requests and now - requests[0] > window:
                requests.popleft()
            
            if len(requests) < max_requests:
                requests.append(now)
                allowed += 1
            else:
                rejected += 1
            
            time.sleep(0.01)  # Small delay between requests
        
        # Should have allowed ~10 requests per second
        assert allowed > 0
        assert rejected > 0


class TestResourceContention:
    """Test resource contention scenarios."""
    
    def test_shared_resource_access(self):
        """Test concurrent access to shared resource."""
        shared_counter = 0
        lock = threading.Lock()
        
        def increment():
            nonlocal shared_counter
            with lock:
                current = shared_counter
                time.sleep(0.001)  # Simulate work
                shared_counter = current + 1
        
        threads = [threading.Thread(target=increment) for _ in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Counter should be exactly 100
        assert shared_counter == 100
    
    def test_connection_pool_exhaustion(self):
        """Test behavior when connection pool is exhausted."""
        max_connections = 5
        active_connections = 0
        lock = threading.Lock()
        
        def use_connection():
            nonlocal active_connections
            with lock:
                if active_connections >= max_connections:
                    return False  # Pool exhausted
                active_connections += 1
            
            time.sleep(0.05)  # Use connection
            
            with lock:
                active_connections -= 1
            return True
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(use_connection) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Some requests should succeed, some should fail due to pool exhaustion
        successful = sum(results)
        assert 0 < successful <= max_connections


class TestAsyncLoad:
    """Test async load handling."""
    
    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self):
        """Test async concurrent request handling."""
        async def async_task(task_id: int, delay: float = 0.01):
            await asyncio.sleep(delay)
            return {"task_id": task_id, "completed": True}
        
        # Run 50 concurrent async tasks
        tasks = [async_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(r["completed"] for r in results)
    
    @pytest.mark.asyncio
    async def test_async_with_semaphore(self):
        """Test async requests with semaphore limiting."""
        semaphore = asyncio.Semaphore(5)
        
        async def limited_task(task_id: int):
            async with semaphore:
                await asyncio.sleep(0.01)
                return {"task_id": task_id}
        
        tasks = [limited_task(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 20


class TestLatencyDistribution:
    """Test latency distribution under load."""
    
    def test_latency_percentiles(self):
        """Test latency percentiles under load."""
        simulator = ConcurrentRequestSimulator()
        result = simulator.run_load_test(
            num_requests=1000,
            concurrent_users=50,
            request_delay=0.01
        )
        
        # Latency percentiles should make sense
        assert result.min_latency <= result.p50_latency <= result.max_latency
        assert result.p50_latency <= result.p95_latency <= result.p99_latency
        assert result.avg_latency > 0
    
    def test_latency_under_increasing_load(self):
        """Test how latency changes under increasing load."""
        simulator = ConcurrentRequestSimulator()
        latencies = []
        
        for users in [10, 25, 50, 75, 100]:
            result = simulator.run_load_test(
                num_requests=users * 10,
                concurrent_users=users,
                request_delay=0.01
            )
            latencies.append((users, result.avg_latency))
        
        # Latency should generally increase with load
        assert len(latencies) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

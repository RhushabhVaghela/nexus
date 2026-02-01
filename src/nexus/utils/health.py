"""
Health Checks Implementation for Nexus

Provides Kubernetes-compatible health probes and component health monitoring.
Supports liveness, readiness, and startup probes.
"""

import os
import asyncio
import json
import logging
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ProbeType(Enum):
    """Types of health probes."""
    LIVENESS = "liveness"      # Is the application running?
    READINESS = "readiness"    # Is the application ready to serve?
    STARTUP = "startup"        # Has the application started?


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": round(self.response_time * 1000, 2),
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "metadata": self.metadata
        }


@dataclass
class HealthReport:
    """Complete health report."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: float
    version: str = "1.0.0"
    service: str = "nexus"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "service": self.service,
            "version": self.version,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "checks": [c.to_dict() for c in self.checks],
            "summary": {
                "total": len(self.checks),
                "healthy": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                "unhealthy": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
                "degraded": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED)
            }
        }
    
    def to_kubernetes_probe(self, probe_type: ProbeType) -> Tuple[int, str]:
        """
        Convert to Kubernetes probe response.
        
        Returns:
            Tuple of (status_code, response_body)
        """
        if probe_type == ProbeType.LIVENESS:
            # Liveness: only fail if completely down
            if self.status == HealthStatus.UNHEALTHY:
                return 503, json.dumps({"status": "unhealthy"})
            return 200, json.dumps({"status": "healthy"})
        
        elif probe_type == ProbeType.READINESS:
            # Readiness: fail if not ready to serve
            if self.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN):
                return 503, json.dumps({"status": "not_ready"})
            return 200, json.dumps({"status": "ready"})
        
        elif probe_type == ProbeType.STARTUP:
            # Startup: fail if not started
            if self.status == HealthStatus.UNKNOWN:
                return 503, json.dumps({"status": "starting"})
            return 200, json.dumps({"status": "started"})
        
        return 200, json.dumps({"status": "unknown"})


class HealthCheck(ABC):
    """Base class for health checks."""
    
    def __init__(
        self,
        name: str,
        timeout: float = 5.0,
        critical: bool = True
    ):
        self.name = name
        self.timeout = timeout
        self.critical = critical
        self._last_result: Optional[HealthCheckResult] = None
        self._lock = threading.Lock()
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform health check.
        
        This method must be implemented by subclasses to perform
        the actual health check logic.
        
        Returns:
            HealthCheckResult with the status of the health check
        """
        pass
    
    def run(self) -> HealthCheckResult:
        """Run health check with timeout and error handling."""
        start_time = time.time()
        
        try:
            result = self.check()
            result.response_time = time.time() - start_time
        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
        
        with self._lock:
            self._last_result = result
        
        return result
    
    async def run_async(self) -> HealthCheckResult:
        """Run health check asynchronously."""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self.run),
            timeout=self.timeout
        )
    
    def get_last_result(self) -> Optional[HealthCheckResult]:
        """Get last check result."""
        with self._lock:
            return self._last_result


class SystemHealthCheck(HealthCheck):
    """Check system resources."""
    
    def __init__(
        self,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 90.0,
        min_disk_gb: float = 1.0
    ):
        super().__init__("system", timeout=5.0, critical=True)
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.min_disk_gb = min_disk_gb
    
    def check(self) -> HealthCheckResult:
        """Check system health."""
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = HealthStatus.HEALTHY if cpu_percent < self.max_cpu_percent else HealthStatus.DEGRADED
            
            # Memory
            memory = psutil.virtual_memory()
            memory_status = HealthStatus.HEALTHY if memory.percent < self.max_memory_percent else HealthStatus.DEGRADED
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024 ** 3)
            disk_status = HealthStatus.HEALTHY if disk_gb > self.min_disk_gb else HealthStatus.UNHEALTHY
            
            # Overall status
            if disk_status == HealthStatus.UNHEALTHY:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {disk_gb:.1f}GB free"
            elif cpu_status == HealthStatus.DEGRADED or memory_status == HealthStatus.DEGRADED:
                status = HealthStatus.DEGRADED
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                response_time=0,
                timestamp=time.time(),
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_free_gb": disk_gb,
                    "disk_total_gb": disk.total / (1024 ** 3)
                }
            )
            
        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system checks",
                response_time=0,
                timestamp=time.time()
            )


class GPUHealthCheck(HealthCheck):
    """Check GPU health and availability."""
    
    def __init__(
        self,
        max_memory_percent: float = 95.0,
        max_temperature: float = 85.0
    ):
        super().__init__("gpu", timeout=5.0, critical=False)
        self.max_memory_percent = max_memory_percent
        self.max_temperature = max_temperature
    
    def check(self) -> HealthCheckResult:
        """Check GPU health."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="GPU not available (CPU mode)",
                    response_time=0,
                    timestamp=time.time()
                )
            
            devices = []
            all_healthy = True
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                
                # Memory
                memory_stats = torch.cuda.memory_stats(i)
                allocated = memory_stats.get("allocated_bytes.all.current", 0)
                reserved = memory_stats.get("reserved_bytes.all.current", 0)
                memory_percent = (allocated / reserved * 100) if reserved > 0 else 0
                
                device_info = {
                    "id": i,
                    "name": device_name,
                    "memory_percent": memory_percent
                }
                
                # Check if using pynvml for temperature
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    device_info["temperature"] = temp
                    
                    if temp > self.max_temperature:
                        all_healthy = False
                        device_info["warning"] = f"High temperature: {temp}°C"
                except:
                    pass
                
                if memory_percent > self.max_memory_percent:
                    all_healthy = False
                    device_info["warning"] = f"High memory usage: {memory_percent:.1f}%"
                
                devices.append(device_info)
            
            status = HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=f"GPU check passed for {len(devices)} device(s)",
                response_time=0,
                timestamp=time.time(),
                metadata={"devices": devices}
            )
            
        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="PyTorch not available for GPU checks",
                response_time=0,
                timestamp=time.time()
            )


class ModelHealthCheck(HealthCheck):
    """Check model loading and inference health."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        test_inference: bool = False
    ):
        super().__init__("model", timeout=30.0, critical=True)
        self.model_path = model_path
        self.test_inference = test_inference
        self._model_loaded = False
    
    def check(self) -> HealthCheckResult:
        """Check model health."""
        try:
            import torch
            
            metadata = {}
            
            # Check if model path exists
            if self.model_path:
                import os
                if not os.path.exists(self.model_path):
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Model path not found: {self.model_path}",
                        response_time=0,
                        timestamp=time.time()
                    )
                metadata["model_path"] = self.model_path
            
            # Check CUDA availability for inference
            if torch.cuda.is_available():
                metadata["cuda_available"] = True
                metadata["cuda_devices"] = torch.cuda.device_count()
                
                # Try a simple CUDA operation
                try:
                    x = torch.randn(10, 10).cuda()
                    y = x @ x.T
                    torch.cuda.synchronize()
                    metadata["cuda_test"] = "passed"
                except Exception as e:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"CUDA test failed: {e}",
                        response_time=0,
                        timestamp=time.time(),
                        metadata=metadata
                    )
            else:
                metadata["cuda_available"] = False
            
            self._model_loaded = True
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Model health check passed",
                response_time=0,
                timestamp=time.time(),
                metadata=metadata
            )
            
        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="PyTorch not available for model checks",
                response_time=0,
                timestamp=time.time()
            )


class DatabaseHealthCheck(HealthCheck):
    """Check database connectivity.
    
    Supports SQLite, PostgreSQL, MySQL, and other databases.
    Uses SQLAlchemy if available for broader database support.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        query: str = "SELECT 1"
    ):
        super().__init__("database", timeout=5.0, critical=True)
        self.connection_string = connection_string
        self.query = query
    
    def check(self) -> HealthCheckResult:
        """Check database health by attempting a connection and simple query."""
        start_time = time.time()
        
        # If no connection string provided, check for common environment variables
        if not self.connection_string:
            self.connection_string = (
                os.environ.get('DATABASE_URL') or 
                os.environ.get('DB_CONNECTION_STRING') or
                os.environ.get('POSTGRES_URL')
            )
        
        if not self.connection_string:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="No database connection string configured",
                response_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={"hint": "Set DATABASE_URL environment variable or pass connection_string"}
            )
        
        try:
            # Try SQLAlchemy first (supports multiple database types)
            try:
                from sqlalchemy import create_engine, text
                from sqlalchemy.exc import SQLAlchemyError
                
                engine = create_engine(self.connection_string, connect_args={'connect_timeout': int(self.timeout)})
                with engine.connect() as conn:
                    result = conn.execute(text(self.query))
                    result.fetchone()
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Database connection successful using SQLAlchemy",
                    response_time=time.time() - start_time,
                    timestamp=time.time(),
                    metadata={
                        "connection_type": "sqlalchemy",
                        "query": self.query,
                        "connection_string_masked": self._mask_connection_string()
                    }
                )
            except ImportError:
                pass  # Fall through to direct driver checks
            
            # Try SQLite
            if self.connection_string.startswith('sqlite:'):
                import sqlite3
                db_path = self.connection_string.replace('sqlite:///', '').replace('sqlite:', '')
                conn = sqlite3.connect(db_path, timeout=self.timeout)
                conn.execute(self.query)
                conn.close()
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="SQLite database connection successful",
                    response_time=time.time() - start_time,
                    timestamp=time.time(),
                    metadata={"connection_type": "sqlite", "db_path": db_path}
                )
            
            # Try psycopg2 for PostgreSQL
            try:
                import psycopg2
                conn = psycopg2.connect(self.connection_string, connect_timeout=int(self.timeout))
                cursor = conn.cursor()
                cursor.execute(self.query)
                cursor.fetchone()
                cursor.close()
                conn.close()
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="PostgreSQL database connection successful",
                    response_time=time.time() - start_time,
                    timestamp=time.time(),
                    metadata={"connection_type": "psycopg2"}
                )
            except ImportError:
                pass
            
            # Try pymysql for MySQL
            try:
                import pymysql
                # Parse connection string for pymysql
                import re
                match = re.match(r'mysql://([^:]+):([^@]+)@([^/]+)/(.+)', self.connection_string)
                if match:
                    user, password, host, database = match.groups()
                    conn = pymysql.connect(
                        host=host,
                        user=user,
                        password=password,
                        database=database,
                        connect_timeout=int(self.timeout)
                    )
                    with conn.cursor() as cursor:
                        cursor.execute(self.query)
                        cursor.fetchone()
                    conn.close()
                    
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="MySQL database connection successful",
                        response_time=time.time() - start_time,
                        timestamp=time.time(),
                        metadata={"connection_type": "pymysql"}
                    )
            except ImportError:
                pass
            
            # If we get here, no database driver was available
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="No suitable database driver found. Install sqlalchemy, psycopg2, or pymysql.",
                response_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={
                    "connection_string_masked": self._mask_connection_string(),
                    "available_drivers": self._get_available_drivers()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=time.time(),
                metadata={
                    "error_type": type(e).__name__,
                    "connection_string_masked": self._mask_connection_string()
                }
            )
    
    def _mask_connection_string(self) -> str:
        """Mask sensitive information in connection string for logging."""
        if not self.connection_string:
            return "None"
        import re
        # Mask password in connection string
        return re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1*****\3', self.connection_string)
    
    def _get_available_drivers(self) -> list:
        """Check which database drivers are available."""
        drivers = []
        try:
            import sqlalchemy
            drivers.append("sqlalchemy")
        except ImportError:
            pass
        try:
            import psycopg2
            drivers.append("psycopg2")
        except ImportError:
            pass
        try:
            import pymysql
            drivers.append("pymysql")
        except ImportError:
            pass
        try:
            import sqlite3
            drivers.append("sqlite3")
        except ImportError:
            pass
        return drivers


class APIHealthCheck(HealthCheck):
    """Check external API availability."""
    
    def __init__(
        self,
        endpoint: str,
        method: str = "GET",
        expected_status: int = 200,
        timeout: float = 5.0
    ):
        super().__init__(f"api_{endpoint}", timeout=timeout, critical=False)
        self.endpoint = endpoint
        self.method = method
        self.expected_status = expected_status
    
    def check(self) -> HealthCheckResult:
        """Check API health."""
        try:
            import requests
            
            start = time.time()
            response = requests.request(
                self.method,
                self.endpoint,
                timeout=self.timeout
            )
            response_time = time.time() - start
            
            if response.status_code == self.expected_status:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"API responded with {response.status_code}",
                    response_time=response_time,
                    timestamp=time.time(),
                    metadata={
                        "endpoint": self.endpoint,
                        "status_code": response.status_code
                    }
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"API returned unexpected status: {response.status_code}",
                    response_time=response_time,
                    timestamp=time.time(),
                    metadata={
                        "endpoint": self.endpoint,
                        "expected": self.expected_status,
                        "actual": response.status_code
                    }
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"API check failed: {e}",
                response_time=0,
                timestamp=time.time(),
                metadata={"endpoint": self.endpoint}
            )


class HealthCheckRegistry:
    """Registry for managing health checks."""
    
    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
    
    def register(self, check: HealthCheck) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[check.name] = check
    
    def unregister(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                return True
            return False
    
    def get(self, name: str) -> Optional[HealthCheck]:
        """Get health check by name."""
        with self._lock:
            return self._checks.get(name)
    
    def get_all(self) -> List[HealthCheck]:
        """Get all registered health checks."""
        with self._lock:
            return list(self._checks.values())
    
    def run_all(self) -> HealthReport:
        """Run all health checks and generate report."""
        with self._lock:
            checks = list(self._checks.values())
        
        results = []
        critical_failed = False
        
        for check in checks:
            result = check.run()
            results.append(result)
            
            if check.critical and result.status == HealthStatus.UNHEALTHY:
                critical_failed = True
        
        # Determine overall status
        if critical_failed:
            status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            status = HealthStatus.DEGRADED
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return HealthReport(
            status=status,
            checks=results,
            timestamp=time.time()
        )
    
    async def run_all_async(self) -> HealthReport:
        """Run all health checks asynchronously."""
        with self._lock:
            checks = list(self._checks.values())
        
        tasks = [check.run_async() for check in checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        critical_failed = False
        
        for check, result in zip(checks, results):
            if isinstance(result, Exception):
                result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed with exception: {result}",
                    response_time=0,
                    timestamp=time.time()
                )
            processed_results.append(result)
            
            if check.critical and result.status == HealthStatus.UNHEALTHY:
                critical_failed = True
        
        # Determine overall status
        if critical_failed:
            status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in processed_results):
            status = HealthStatus.DEGRADED
        elif any(r.status == HealthStatus.DEGRADED for r in processed_results):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return HealthReport(
            status=status,
            checks=processed_results,
            timestamp=time.time()
        )


# Global registry instance
_registry: Optional[HealthCheckRegistry] = None
_lock = threading.Lock()


def get_health_registry() -> HealthCheckRegistry:
    """Get or create global health registry."""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = HealthCheckRegistry()
    return _registry


def configure_health_checks(
    check_gpu: bool = True,
    check_system: bool = True,
    model_path: Optional[str] = None
) -> HealthCheckRegistry:
    """
    Configure standard health checks.
    
    Args:
        check_gpu: Enable GPU health checks
        check_system: Enable system health checks
        model_path: Path to model for model health check
        
    Returns:
        Configured HealthCheckRegistry
    """
    global _registry
    with _lock:
        _registry = HealthCheckRegistry()
        
        if check_system:
            _registry.register(SystemHealthCheck())
        
        if check_gpu:
            _registry.register(GPUHealthCheck())
        
        if model_path:
            _registry.register(ModelHealthCheck(model_path))
    
    return _registry


def check_health() -> HealthReport:
    """Run all health checks and return report."""
    return get_health_registry().run_all()


async def check_health_async() -> HealthReport:
    """Run all health checks asynchronously."""
    return await get_health_registry().run_all_async()


# Kubernetes probe handlers

def liveness_probe() -> Tuple[int, str]:
    """
    Kubernetes liveness probe handler.
    
    Returns:
        Tuple of (status_code, response_body)
    """
    report = check_health()
    return report.to_kubernetes_probe(ProbeType.LIVENESS)


def readiness_probe() -> Tuple[int, str]:
    """
    Kubernetes readiness probe handler.
    
    Returns:
        Tuple of (status_code, response_body)
    """
    report = check_health()
    return report.to_kubernetes_probe(ProbeType.READINESS)


def startup_probe() -> Tuple[int, str]:
    """
    Kubernetes startup probe handler.
    
    Returns:
        Tuple of (status_code, response_body)
    """
    report = check_health()
    return report.to_kubernetes_probe(ProbeType.STARTUP)
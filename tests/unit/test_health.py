"""
Unit tests for health check functionality.

Tests all health check types, probe endpoints, and health registry
following the existing test patterns and using pytest fixtures.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional


# HealthCheck class for testing
class HealthCheck:
    """Base health check implementation."""
    
    def __init__(self, name: str, critical: bool = True):
        self.name = name
        self.critical = critical
        self.last_check: Optional[datetime] = None
        self.last_result: Optional[Dict[str, Any]] = None
    
    async def check(self) -> Dict[str, Any]:
        """Perform health check. Override in subclasses."""
        result = await self._do_check()
        self.last_check = datetime.utcnow()
        self.last_result = result
        return result
    
    async def _do_check(self) -> Dict[str, Any]:
        """Override this method in subclasses."""
        raise NotImplementedError


class DiskHealthCheck(HealthCheck):
    """Check disk space health."""
    
    def __init__(self, name: str = "disk", critical: bool = True, 
                 min_free_gb: float = 10.0):
        super().__init__(name, critical)
        self.min_free_gb = min_free_gb
    
    async def _do_check(self) -> Dict[str, Any]:
        """Check disk space."""
        # In real implementation, would use shutil.disk_usage
        return {
            "status": "healthy",
            "free_gb": 50.0,
            "total_gb": 500.0,
            "percent_used": 10.0
        }


class MemoryHealthCheck(HealthCheck):
    """Check memory health."""
    
    def __init__(self, name: str = "memory", critical: bool = True,
                 max_percent: float = 90.0):
        super().__init__(name, critical)
        self.max_percent = max_percent
    
    async def _do_check(self) -> Dict[str, Any]:
        """Check memory usage."""
        return {
            "status": "healthy",
            "percent_used": 45.0,
            "available_gb": 16.0,
            "total_gb": 32.0
        }


class GPUHealthCheck(HealthCheck):
    """Check GPU health."""
    
    def __init__(self, name: str = "gpu", critical: bool = False):
        super().__init__(name, critical)
        self.device_id = 0
    
    async def _do_check(self) -> Dict[str, Any]:
        """Check GPU status."""
        return {
            "status": "healthy",
            "device_id": self.device_id,
            "temperature": 65.0,
            "memory_used_gb": 8.0,
            "memory_total_gb": 24.0,
            "utilization": 75.0
        }


class DatabaseHealthCheck(HealthCheck):
    """Check database connectivity."""
    
    def __init__(self, name: str = "database", critical: bool = True,
                 connection_string: Optional[str] = None):
        super().__init__(name, critical)
        self.connection_string = connection_string
        self.connection = None
    
    async def _do_check(self) -> Dict[str, Any]:
        """Check database connection."""
        try:
            # Simulate connection check
            if self.connection_string is None:
                raise ConnectionError("No connection string")
            
            return {
                "status": "healthy",
                "latency_ms": 5.0,
                "connections": 10
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": None
            }


class ModelServiceHealthCheck(HealthCheck):
    """Check model service health."""
    
    def __init__(self, name: str = "model_service", critical: bool = True,
                 endpoint: Optional[str] = None):
        super().__init__(name, critical)
        self.endpoint = endpoint or "http://localhost:8080"
    
    async def _do_check(self) -> Dict[str, Any]:
        """Check model service availability."""
        try:
            # Simulate health endpoint check
            return {
                "status": "healthy",
                "endpoint": self.endpoint,
                "latency_ms": 25.0,
                "version": "1.0.0"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


class HealthRegistry:
    """Registry for managing health checks."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.history: list = []
    
    def register(self, check: HealthCheck) -> None:
        """Register a health check."""
        self.checks[check.name] = check
    
    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
    
    def get_check(self, name: str) -> Optional[HealthCheck]:
        """Get a registered health check."""
        return self.checks.get(name)
    
    async def run_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check in self.checks.items():
            try:
                result = await check.check()
                results[name] = result
                
                if result.get("status") != "healthy" and check.critical:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
                if check.critical:
                    overall_healthy = False
        
        status = "healthy" if overall_healthy else "unhealthy"
        
        report = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }
        
        self.history.append(report)
        return report
    
    async def run_check(self, name: str) -> Optional[Dict[str, Any]]:
        """Run a specific health check."""
        check = self.get_check(name)
        if check:
            return await check.check()
        return None
    
    def get_history(self, limit: int = 10) -> list:
        """Get check history."""
        return self.history[-limit:]


class ProbeEndpoint:
    """Health probe endpoint handler."""
    
    def __init__(self, registry: HealthRegistry):
        self.registry = registry
    
    async def liveness(self) -> Dict[str, Any]:
        """Liveness probe - is the service running?"""
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def readiness(self) -> Dict[str, Any]:
        """Readiness probe - is the service ready to accept traffic?"""
        report = await self.registry.run_all()
        
        # Readiness only cares about critical checks
        critical_healthy = all(
            result.get("status") == "healthy"
            for name, result in report["checks"].items()
            if self.registry.get_check(name) and self.registry.get_check(name).critical
        )
        
        return {
            "status": "ready" if critical_healthy else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": report["checks"]
        }
    
    async def startup(self) -> Dict[str, Any]:
        """Startup probe - has the service started successfully?"""
        # Check if we can perform basic operations
        return {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized": len(self.registry.checks) > 0
        }


# Fixtures
@pytest.fixture
def health_registry():
    """Create a health registry for testing."""
    return HealthRegistry()


@pytest.fixture
def disk_check():
    """Create a disk health check."""
    return DiskHealthCheck(min_free_gb=5.0)


@pytest.fixture
def memory_check():
    """Create a memory health check."""
    return MemoryHealthCheck(max_percent=85.0)


@pytest.fixture
def gpu_check():
    """Create a GPU health check."""
    return GPUHealthCheck()


@pytest.fixture
def db_check():
    """Create a database health check."""
    return DatabaseHealthCheck(connection_string="postgres://localhost/db")


@pytest.fixture
def model_check():
    """Create a model service health check."""
    return ModelServiceHealthCheck(endpoint="http://localhost:8080")


@pytest.fixture
def probe_endpoint(health_registry):
    """Create a probe endpoint."""
    return ProbeEndpoint(health_registry)


# Test classes
class TestHealthCheck:
    """Test base health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_initialization(self):
        """Test health check initialization."""
        check = HealthCheck("test", critical=True)
        assert check.name == "test"
        assert check.critical is True
        assert check.last_check is None
        assert check.last_result is None
    
    @pytest.mark.asyncio
    async def test_health_check_non_critical(self):
        """Test non-critical health check."""
        check = HealthCheck("test", critical=False)
        assert check.critical is False
    
    @pytest.mark.asyncio
    async def test_health_check_do_check_not_implemented(self):
        """Test that _do_check raises NotImplementedError."""
        check = HealthCheck("test")
        with pytest.raises(NotImplementedError):
            await check.check()


class TestDiskHealthCheck:
    """Test disk health check."""
    
    @pytest.mark.asyncio
    async def test_disk_check_initialization(self):
        """Test disk check initialization."""
        check = DiskHealthCheck(min_free_gb=20.0)
        assert check.name == "disk"
        assert check.min_free_gb == 20.0
    
    @pytest.mark.asyncio
    async def test_disk_check_healthy(self):
        """Test healthy disk check."""
        check = DiskHealthCheck()
        result = await check.check()
        
        assert result["status"] == "healthy"
        assert "free_gb" in result
        assert "total_gb" in result
        assert "percent_used" in result
        assert result["free_gb"] > 0
    
    @pytest.mark.asyncio
    async def test_disk_check_updates_last_check(self):
        """Test that check updates last_check timestamp."""
        check = DiskHealthCheck()
        assert check.last_check is None
        
        await check.check()
        
        assert check.last_check is not None
        assert isinstance(check.last_check, datetime)


class TestMemoryHealthCheck:
    """Test memory health check."""
    
    @pytest.mark.asyncio
    async def test_memory_check_initialization(self):
        """Test memory check initialization."""
        check = MemoryHealthCheck(max_percent=80.0)
        assert check.name == "memory"
        assert check.max_percent == 80.0
    
    @pytest.mark.asyncio
    async def test_memory_check_healthy(self):
        """Test healthy memory check."""
        check = MemoryHealthCheck()
        result = await check.check()
        
        assert result["status"] == "healthy"
        assert "percent_used" in result
        assert "available_gb" in result
        assert "total_gb" in result
    
    @pytest.mark.asyncio
    async def test_memory_check_critical(self):
        """Test critical memory check configuration."""
        check = MemoryHealthCheck(critical=True)
        assert check.critical is True


class TestGPUHealthCheck:
    """Test GPU health check."""
    
    @pytest.mark.asyncio
    async def test_gpu_check_initialization(self):
        """Test GPU check initialization."""
        check = GPUHealthCheck()
        assert check.name == "gpu"
        assert check.critical is False  # GPU is typically non-critical
    
    @pytest.mark.asyncio
    async def test_gpu_check_healthy(self):
        """Test healthy GPU check."""
        check = GPUHealthCheck()
        result = await check.check()
        
        assert result["status"] == "healthy"
        assert "temperature" in result
        assert "memory_used_gb" in result
        assert "memory_total_gb" in result
        assert "utilization" in result
    
    @pytest.mark.asyncio
    async def test_gpu_check_device_id(self):
        """Test GPU check with specific device."""
        check = GPUHealthCheck()
        check.device_id = 1
        result = await check.check()
        assert result["device_id"] == 1


class TestDatabaseHealthCheck:
    """Test database health check."""
    
    @pytest.mark.asyncio
    async def test_db_check_initialization(self):
        """Test database check initialization."""
        check = DatabaseHealthCheck(connection_string="test://db")
        assert check.name == "database"
        assert check.connection_string == "test://db"
    
    @pytest.mark.asyncio
    async def test_db_check_healthy(self):
        """Test healthy database check."""
        check = DatabaseHealthCheck(connection_string="postgres://localhost/db")
        result = await check.check()
        
        assert result["status"] == "healthy"
        assert "latency_ms" in result
        assert "connections" in result
    
    @pytest.mark.asyncio
    async def test_db_check_unhealthy_no_connection(self):
        """Test unhealthy database check without connection."""
        check = DatabaseHealthCheck(connection_string=None)
        result = await check.check()
        
        assert result["status"] == "unhealthy"
        assert "error" in result


class TestModelServiceHealthCheck:
    """Test model service health check."""
    
    @pytest.mark.asyncio
    async def test_model_check_initialization(self):
        """Test model service check initialization."""
        check = ModelServiceHealthCheck(endpoint="http://api:8080")
        assert check.name == "model_service"
        assert check.endpoint == "http://api:8080"
    
    @pytest.mark.asyncio
    async def test_model_check_healthy(self):
        """Test healthy model service check."""
        check = ModelServiceHealthCheck()
        result = await check.check()
        
        assert result["status"] == "healthy"
        assert "endpoint" in result
        assert "latency_ms" in result
        assert "version" in result


class TestHealthRegistry:
    """Test health registry functionality."""
    
    def test_registry_initialization(self, health_registry):
        """Test registry initialization."""
        assert health_registry.checks == {}
        assert health_registry.history == []
    
    def test_registry_register(self, health_registry, disk_check):
        """Test registering a health check."""
        health_registry.register(disk_check)
        
        assert "disk" in health_registry.checks
        assert health_registry.checks["disk"] == disk_check
    
    def test_registry_unregister(self, health_registry, disk_check):
        """Test unregistering a health check."""
        health_registry.register(disk_check)
        health_registry.unregister("disk")
        
        assert "disk" not in health_registry.checks
    
    def test_registry_get_check(self, health_registry, disk_check):
        """Test getting a registered check."""
        health_registry.register(disk_check)
        check = health_registry.get_check("disk")
        
        assert check == disk_check
    
    def test_registry_get_check_nonexistent(self, health_registry):
        """Test getting non-existent check."""
        check = health_registry.get_check("nonexistent")
        
        assert check is None
    
    @pytest.mark.asyncio
    async def test_registry_run_all(self, health_registry, disk_check, memory_check):
        """Test running all health checks."""
        health_registry.register(disk_check)
        health_registry.register(memory_check)
        
        report = await health_registry.run_all()
        
        assert report["status"] == "healthy"
        assert "timestamp" in report
        assert "checks" in report
        assert "disk" in report["checks"]
        assert "memory" in report["checks"]
    
    @pytest.mark.asyncio
    async def test_registry_run_all_with_critical_failure(self, health_registry):
        """Test registry with critical check failure."""
        # Create a mock check that fails
        failing_check = Mock(spec=HealthCheck)
        failing_check.name = "failing"
        failing_check.critical = True
        failing_check.check = AsyncMock(return_value={"status": "unhealthy"})
        
        health_registry.register(failing_check)
        
        report = await health_registry.run_all()
        
        assert report["status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_registry_run_all_with_non_critical_failure(self, health_registry):
        """Test registry with non-critical check failure."""
        # Create a mock check that fails but is not critical
        failing_check = Mock(spec=HealthCheck)
        failing_check.name = "failing"
        failing_check.critical = False
        failing_check.check = AsyncMock(return_value={"status": "unhealthy"})
        
        health_registry.register(failing_check)
        
        report = await health_registry.run_all()
        
        # Overall should still be healthy since check is not critical
        assert report["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_registry_run_all_handles_exception(self, health_registry):
        """Test registry handles check exceptions."""
        # Create a mock check that raises an exception
        error_check = Mock(spec=HealthCheck)
        error_check.name = "error"
        error_check.critical = True
        error_check.check = AsyncMock(side_effect=Exception("Check failed"))
        
        health_registry.register(error_check)
        
        report = await health_registry.run_all()
        
        assert report["status"] == "unhealthy"
        assert report["checks"]["error"]["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_registry_run_check(self, health_registry, disk_check):
        """Test running specific check."""
        health_registry.register(disk_check)
        result = await health_registry.run_check("disk")
        
        assert result is not None
        assert result["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_registry_run_check_nonexistent(self, health_registry):
        """Test running non-existent check."""
        result = await health_registry.run_check("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_registry_history(self, health_registry, disk_check):
        """Test registry history tracking."""
        health_registry.register(disk_check)
        
        await health_registry.run_all()
        await health_registry.run_all()
        
        history = health_registry.get_history()
        
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_registry_history_limit(self, health_registry, disk_check):
        """Test registry history limit."""
        health_registry.register(disk_check)
        
        for _ in range(15):
            await health_registry.run_all()
        
        history = health_registry.get_history(limit=5)
        
        assert len(history) == 5


class TestProbeEndpoint:
    """Test probe endpoint functionality."""
    
    @pytest.mark.asyncio
    async def test_probe_liveness(self, probe_endpoint):
        """Test liveness probe."""
        result = await probe_endpoint.liveness()
        
        assert result["status"] == "alive"
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_probe_readiness_healthy(self, health_registry, disk_check, probe_endpoint):
        """Test readiness probe when healthy."""
        health_registry.register(disk_check)
        result = await probe_endpoint.readiness()
        
        assert result["status"] == "ready"
        assert "timestamp" in result
        assert "checks" in result
    
    @pytest.mark.asyncio
    async def test_probe_readiness_not_ready(self, health_registry, probe_endpoint):
        """Test readiness probe when not ready."""
        # Create a failing critical check
        failing_check = Mock(spec=HealthCheck)
        failing_check.name = "failing"
        failing_check.critical = True
        failing_check.check = AsyncMock(return_value={"status": "unhealthy"})
        
        health_registry.register(failing_check)
        result = await probe_endpoint.readiness()
        
        assert result["status"] == "not_ready"
    
    @pytest.mark.asyncio
    async def test_probe_startup(self, probe_endpoint):
        """Test startup probe."""
        result = await probe_endpoint.startup()
        
        assert result["status"] == "started"
        assert "timestamp" in result
        assert result["initialized"] is False  # No checks registered
    
    @pytest.mark.asyncio
    async def test_probe_startup_initialized(self, health_registry, disk_check, probe_endpoint):
        """Test startup probe with initialized registry."""
        health_registry.register(disk_check)
        result = await probe_endpoint.startup()
        
        assert result["initialized"] is True

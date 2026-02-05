# Exception Hierarchy Migration Guide

This guide helps you migrate your code to use the unified Nexus exception hierarchy.

## Overview

The Nexus exception hierarchy has been unified to:
- Eliminate duplicate exception classes
- Provide a consistent base exception type
- Maintain backward compatibility with deprecation warnings

## New Exception Structure

```
NexusBaseError
├── NexusValueError
├── NexusTypeError
├── NexusIOError
├── NexusConfigError
└── NexusRuntimeError
    ├── CircuitBreakerOpen
    ├── RateLimitExceeded
    ├── BulkheadFull
    ├── BulkheadTimeout
    ├── NexusTimeoutError
    ├── SLIError
    │   ├── UnsupportedArchitectureError
    │   ├── WeightLoadingError
    │   ├── LayerCreationError
    │   ├── MoEConfigurationError
    │   ├── FormatDetectionError
    │   └── WeightMapError
    ├── TrainingError
    └── InferenceError
```

## Migration Steps

### Before (Old Way)

```python
# Importing from various locations
from src.core.resilience import CircuitBreakerOpen
from src.utils.circuit_breaker import CircuitBreakerOpen
from src.security.rate_limiter import RateLimitExceeded
from src.utils.rate_limiter import RateLimitExceeded
```

### After (New Way)

```python
# Import from unified location
from src.core.exceptions import (
    CircuitBreakerOpen,
    RateLimitExceeded,
    NexusBaseError,
    NexusRuntimeError,
)

# Or use the core module convenience import
from src.core import CircuitBreakerOpen, RateLimitExceeded
```

## Catching Exceptions

### Catch by Specific Type

```python
try:
    await circuit_breaker.call(service_call)
except CircuitBreakerOpen as e:
    logger.error(f"Circuit {e.name} is open: {e.last_error}")
    # Fallback logic
```

### Catch by Base Type

```python
try:
    await rate_limited_function()
except NexusRuntimeError as e:
    # Handles CircuitBreakerOpen, RateLimitExceeded, etc.
    logger.error(f"Runtime error: {e}")
```

### Catch Multiple Exception Types

```python
try:
    await operation()
except (CircuitBreakerOpen, RateLimitExceeded) as e:
    # Handle resilience-related errors
    retry_after = getattr(e, 'retry_after', 0)
except NexusBaseError as e:
    # Handle any Nexus error
    logger.error(f"Nexus error: {e}")
```

## Exception Attributes

### CircuitBreakerOpen

```python
from src.core.exceptions import CircuitBreakerOpen

try:
    raise CircuitBreakerOpen("database", "Connection timeout")
except CircuitBreakerOpen as e:
    print(f"Circuit: {e.name}")      # "database"
    print(f"Last error: {e.last_error}")  # "Connection timeout"
```

### RateLimitExceeded

```python
from src.core.exceptions import RateLimitExceeded

try:
    raise RateLimitExceeded("user123", 100, 30.5)
except RateLimitExceeded as e:
    print(f"Key: {e.key}")            # "user123"
    print(f"Limit: {e.limit}")        # 100
    print(f"Retry after: {e.retry_after}")  # 30.5
```

## Backward Compatibility

Old exception classes are still available but deprecated:

```python
# These still work but show deprecation warnings
from src.core.resilience import CircuitBreakerOpen  # Deprecated
from src.utils.circuit_breaker import CircuitBreakerOpen  # Deprecated
from src.security.rate_limiter import RateLimitExceeded  # Deprecated
from src.utils.rate_limiter import RateLimitExceeded  # Deprecated
```

### Deprecation Warning Example

When using deprecated imports, you'll see:

```
DeprecationWarning: CircuitBreakerOpen from src.core.resilience is deprecated. 
Please use CircuitBreakerOpen from src.core.exceptions instead. 
This will be removed in a future version.
```

### Code Fix for Deprecation Warnings

**Before:**
```python
from src.core.resilience import CircuitBreakerOpen
```

**After:**
```python
from src.core.exceptions import CircuitBreakerOpen
```

## Updating Test Files

### Old Test Code

```python
import pytest
from src.core.resilience import CircuitBreakerOpen

def test_circuit_open():
    with pytest.raises(CircuitBreakerOpen) as exc_info:
        raise CircuitBreakerOpen("test_circuit", "test error")
```

### New Test Code

```python
import pytest
from src.core.exceptions import CircuitBreakerOpen

def test_circuit_open():
    with pytest.raises(CircuitBreakerOpen) as exc_info:
        raise CircuitBreakerOpen("test_circuit", "test error")
    
    # Can also catch by base type
    with pytest.raises(Exception) as exc_info:
        raise CircuitBreakerOpen("test", "error")
    
    assert isinstance(exc_info.value, CircuitBreakerOpen)
```

## Exception Hierarchy Benefits

1. **Single Source of Truth**: No more duplicate exception definitions
2. **Consistent API**: All exceptions have consistent attributes and behavior
3. **Better Organization**: Clear hierarchy makes error handling more intuitive
4. **Future-Proof**: New exceptions can be added to the appropriate category
5. **Backward Compatible**: Existing code continues to work with deprecation warnings

## Migration Timeline

- **Phase 1** (Current): Old exceptions work with deprecation warnings
- **Phase 2** (Future): Old exceptions removed from their original locations
- **Phase 3** (Future): Only unified exceptions remain

## Support

For questions or issues:
- Check the deprecation warnings in your code
- Review this migration guide
- Look at the unified exceptions in `src/core/exceptions.py`

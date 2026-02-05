"""
Distributed Tracing Implementation for Nexus

Provides OpenTelemetry-compatible distributed tracing with Jaeger integration.
Supports automatic span creation and context propagation.
"""

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import threading

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("opentelemetry not available, tracing will be local only")


@dataclass
class SpanContext:
    """Local span context for when OpenTelemetry is not available."""
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    sampled: bool = True
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """Local span implementation."""
    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "unset"
    status_description: str = ""
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value
    
    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes."""
        self.attributes.update(attributes)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def record_exception(self, exception: Exception) -> None:
        """Record exception in span."""
        self.events.append({
            "name": "exception",
            "timestamp": time.time(),
            "attributes": {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            }
        })
        self.set_status("error", str(exception))
    
    def set_status(self, status: str, description: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_description = description
    
    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Get span duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_id": self.context.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_description": self.status_description,
        }


class LocalTracer:
    """Local tracer implementation when OpenTelemetry is not available."""
    
    def __init__(self, name: str = "nexus"):
        self.name = name
        self._active_spans: Dict[str, Span] = {}
        self._span_stack: ContextVar[Optional[Span]] = ContextVar('span_stack', default=None)
        self._lock = threading.Lock()
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return uuid.uuid4().hex[:16]
    
    def _generate_trace_id(self) -> str:
        """Generate trace ID."""
        return uuid.uuid4().hex[:32]
    
    def start_span(
        self,
        name: str,
        context: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        if context is None:
            # Check for active span
            parent = self._span_stack.get()
            if parent:
                context = SpanContext(
                    trace_id=parent.context.trace_id,
                    span_id=self._generate_id(),
                    parent_id=parent.context.span_id
                )
            else:
                context = SpanContext(
                    trace_id=self._generate_trace_id(),
                    span_id=self._generate_id()
                )
        
        span = Span(name=name, context=context)
        if attributes:
            span.set_attributes(attributes)
        
        with self._lock:
            self._active_spans[context.span_id] = span
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()
        with self._lock:
            if span.context.span_id in self._active_spans:
                del self._active_spans[span.context.span_id]
    
    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Start span as current context."""
        span = self.start_span(name, attributes=attributes)
        token = self._span_stack.set(span)
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self.end_span(span)
            self._span_stack.reset(token)
    
    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self._span_stack.get()


class TracerManager:
    """Manages tracers and span exporters."""
    
    def __init__(
        self,
        service_name: str = "nexus",
        jaeger_endpoint: Optional[str] = None,
        jaeger_agent_host: str = "localhost",
        jaeger_agent_port: int = 6831,
        sample_rate: float = 1.0,
        console_export: bool = False
    ):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.sample_rate = sample_rate
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup_opentelemetry(
                jaeger_agent_host,
                jaeger_agent_port,
                console_export
            )
        else:
            self.tracer = LocalTracer(service_name)
            self.propagator = None
    
    def _setup_opentelemetry(
        self,
        agent_host: str,
        agent_port: int,
        console_export: bool
    ):
        """Setup OpenTelemetry tracer."""
        # Create tracer provider
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        
        # Add Jaeger exporter if configured
        if self.jaeger_endpoint or agent_host:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=agent_host,
                    agent_port=agent_port,
                )
                provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
                logger.info(f"Jaeger exporter configured: {agent_host}:{agent_port}")
            except Exception as e:
                logger.error(f"Failed to configure Jaeger exporter: {e}")
        
        # Add console exporter if requested
        if console_export:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
        
        self.tracer = trace.get_tracer(self.service_name)
        self.propagator = TraceContextTextMapPropagator()
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Start a new span.
        
        Args:
            name: Span name
            kind: Span kind (internal, server, client, producer, consumer)
            attributes: Initial span attributes
            
        Yields:
            Active span
        """
        if OPENTELEMETRY_AVAILABLE:
            span_kind = getattr(SpanKind, kind.upper(), SpanKind.INTERNAL)
            with self.tracer.start_as_current_span(
                name,
                kind=span_kind,
                attributes=attributes
            ) as span:
                yield span
        else:
            with self.tracer.start_as_current_span(name, attributes) as span:
                with self._lock:
                    self._spans.append(span)
                yield span
    
    def start_manual_span(
        self,
        name: str,
        parent: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Start a manual span (not in context)."""
        if OPENTELEMETRY_AVAILABLE:
            ctx = trace.set_span_in_context(parent) if parent else None
            return self.tracer.start_span(name, context=ctx, attributes=attributes)
        else:
            parent_ctx = parent.context if parent else None
            return self.tracer.start_span(name, parent_ctx, attributes)
    
    def get_current_span(self) -> Optional[Any]:
        """Get current span."""
        if OPENTELEMETRY_AVAILABLE:
            return trace.get_current_span()
        else:
            return self.tracer.get_current_span()
    
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier."""
        if OPENTELEMETRY_AVAILABLE and self.propagator:
            self.propagator.inject(carrier)
    
    def extract_context(self, carrier: Dict[str, str]) -> Any:
        """Extract trace context from carrier."""
        if OPENTELEMETRY_AVAILABLE and self.propagator:
            return self.propagator.extract(carrier)
        return None
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to current span."""
        span = self.get_current_span()
        if span:
            if OPENTELEMETRY_AVAILABLE:
                span.add_event(name, attributes)
            else:
                span.add_event(name, attributes)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on current span."""
        span = self.get_current_span()
        if span:
            span.set_attribute(key, value)
    
    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes on current span."""
        span = self.get_current_span()
        if span:
            if OPENTELEMETRY_AVAILABLE:
                span.set_attributes(attributes)
            else:
                span.set_attributes(attributes)
    
    def record_exception(self, exception: Exception) -> None:
        """Record exception on current span."""
        span = self.get_current_span()
        if span:
            if OPENTELEMETRY_AVAILABLE:
                span.record_exception(exception)
                span.set_status(Status(StatusCode.ERROR, str(exception)))
            else:
                span.record_exception(exception)
    
    def get_exported_spans(self) -> List[Dict[str, Any]]:
        """Get all exported spans (local mode only)."""
        if OPENTELEMETRY_AVAILABLE:
            return []
        
        with self._lock:
            return [span.to_dict() for span in self._spans if span.end_time]


# Global tracer manager
_tracer_manager: Optional[TracerManager] = None
_lock = threading.Lock()


def get_tracer_manager() -> TracerManager:
    """Get or create global tracer manager."""
    global _tracer_manager
    if _tracer_manager is None:
        with _lock:
            if _tracer_manager is None:
                _tracer_manager = TracerManager()
    return _tracer_manager


def configure_tracing(
    service_name: str = "nexus",
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    sample_rate: float = 1.0,
    console_export: bool = False
) -> TracerManager:
    """
    Configure distributed tracing.
    
    Args:
        service_name: Name of the service
        jaeger_host: Jaeger agent host
        jaeger_port: Jaeger agent port
        sample_rate: Sampling rate (0.0 to 1.0)
        console_export: Also export to console
        
    Returns:
        Configured TracerManager
    """
    global _tracer_manager
    with _lock:
        _tracer_manager = TracerManager(
            service_name=service_name,
            jaeger_agent_host=jaeger_host,
            jaeger_agent_port=jaeger_port,
            sample_rate=sample_rate,
            console_export=console_export
        )
    return _tracer_manager


def trace_span(
    name: Optional[str] = None,
    kind: str = "internal",
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator to trace function execution.
    
    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Additional attributes
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_tracer_manager()
            attrs = attributes or {}
            attrs["function.name"] = func.__name__
            attrs["function.module"] = func.__module__
            
            with manager.start_span(span_name, kind, attrs):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_tracer_manager()
            attrs = attributes or {}
            attrs["function.name"] = func.__name__
            attrs["function.module"] = func.__module__
            
            with manager.start_span(span_name, kind, attrs):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def trace_method(
    name: Optional[str] = None,
    kind: str = "internal"
):
    """Decorator to trace method execution (includes class name)."""
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            manager = get_tracer_manager()
            attrs = {
                "function.name": func.__name__,
                "function.module": func.__module__,
                "class.name": type(self).__name__
            }
            
            with manager.start_span(span_name, kind, attrs):
                return func(self, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            manager = get_tracer_manager()
            attrs = {
                "function.name": func.__name__,
                "function.module": func.__module__,
                "class.name": type(self).__name__
            }
            
            with manager.start_span(span_name, kind, attrs):
                if asyncio.iscoroutinefunction(func):
                    return await func(self, *args, **kwargs)
                return func(self, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Import asyncio at end to avoid circular imports
import asyncio
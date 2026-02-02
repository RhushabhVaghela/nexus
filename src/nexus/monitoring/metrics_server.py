"""
Prometheus Metrics Server for Nexus

Provides HTTP endpoint for Prometheus scraping and metrics export.

Author: Nexus Team
"""

import threading
import logging
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
    Info,
)

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    registry: Optional[CollectorRegistry] = None
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/metrics':
            self._handle_metrics()
        elif path == '/health':
            self._handle_health()
        elif path == '/':
            self._handle_root()
        else:
            self._handle_404()
    
    def _handle_metrics(self):
        """Handle metrics endpoint."""
        try:
            if self.registry is None:
                self.registry = CollectorRegistry()
            
            output = generate_latest(self.registry)
            
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.send_header('Content-Length', str(len(output)))
            self.end_headers()
            self.wfile.write(output)
            
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            self._handle_error(str(e))
    
    def _handle_health(self):
        """Handle health check endpoint."""
        response = b'{"status": "healthy"}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def _handle_root(self):
        """Handle root endpoint."""
        response = b'Nexus Metrics Server. Visit /metrics for Prometheus metrics.'
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def _handle_404(self):
        """Handle 404 errors."""
        response = b'Not Found'
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def _handle_error(self, message: str):
        """Handle internal errors."""
        response = f'Error: {message}'.encode()
        self.send_response(500)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"{self.address_string()} - {format % args}")


class MetricsServer:
    """
    Prometheus metrics server for Nexus.
    
    Provides HTTP endpoint for Prometheus scraping on configurable port.
    
    Example:
        >>> server = MetricsServer(port=9090)
        >>> server.start()
        >>> 
        >>> # Metrics available at http://localhost:9090/metrics
        >>> 
        >>> server.stop()
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        registry: Optional[CollectorRegistry] = None
    ):
        """
        Initialize metrics server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            registry: Prometheus registry (default: global)
        """
        self.host = host
        self.port = port
        self.registry = registry or CollectorRegistry()
        
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Set registry on handler class
        MetricsHandler.registry = self.registry
        
        logger.info(f"MetricsServer initialized (host={host}, port={port})")
    
    def start(self):
        """Start the metrics server."""
        if self._running:
            logger.warning("Server already running")
            return
        
        try:
            self._server = HTTPServer(
                (self.host, self.port),
                MetricsHandler
            )
            
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="metrics-server"
            )
            self._thread.start()
            self._running = True
            
            logger.info(f"Metrics server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop(self):
        """Stop the metrics server."""
        if not self._running:
            return
        
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        self._running = False
        logger.info("Metrics server stopped")
    
    def get_registry(self) -> CollectorRegistry:
        """Get the Prometheus registry."""
        return self.registry
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


# Global server instance
_global_server: Optional[MetricsServer] = None
_global_lock = threading.Lock()


def start_metrics_server(
    host: str = "0.0.0.0",
    port: int = 9090,
    registry: Optional[CollectorRegistry] = None
) -> MetricsServer:
    """
    Start the global metrics server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        registry: Prometheus registry
        
    Returns:
        MetricsServer instance
    """
    global _global_server
    
    with _global_lock:
        if _global_server is None:
            _global_server = MetricsServer(host, port, registry)
            _global_server.start()
        
        return _global_server


def stop_metrics_server():
    """Stop the global metrics server."""
    global _global_server
    
    with _global_lock:
        if _global_server is not None:
            _global_server.stop()
            _global_server = None


def get_metrics_server() -> Optional[MetricsServer]:
    """Get the global metrics server instance."""
    return _global_server

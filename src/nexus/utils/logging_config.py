"""
utils/logging_config.py
Shared logging configuration for all data generation scripts.
Ensures consistent log format across finetuned, repetitive, and other generators.

Features:
- RotatingFileHandler with size-based rotation (10MB default)
- Time-based rotation (daily)
- gzip compression for old logs
- 30-day retention policy
- Structured JSON logging option
- Separate log levels per module
"""

import os
import sys
import json
import gzip
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading


# ═══════════════════════════════════════════════════════════════
# LOG FORMAT TEMPLATES
# ═══════════════════════════════════════════════════════════════

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Simple format for console output
CONSOLE_FORMAT = "%(asctime)s - %(message)s"

# Progress log template (consistent across all generators)
PROGRESS_TEMPLATE = (
    "✓ Total: {total:,} ({rate:.0f}/sec) | "
    "Train: {train:,} Val: {val:,} Test: {test:,} | "
    "Dedup: {dedup} | ETA: {eta:.1f}h"
)

# Benchmark log template
BENCHMARK_TEMPLATE = (
    "📥 {name:<10} | Split: {split:<10} | "
    "Processed: {current:>6}/{total:<6} | Status: {status}"
)


# ═══════════════════════════════════════════════════════════════
# CUSTOM HANDLERS
# ═══════════════════════════════════════════════════════════════

class GzipRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that compresses old logs with gzip.
    
    Features:
    - Rotates logs when they reach maxBytes
    - Compresses rotated logs with gzip
    - Maintains backupCount files
    """
    
    def __init__(self, filename: str, mode: str = 'a', 
                 maxBytes: int = 10*1024*1024, backupCount: int = 5,
                 encoding: Optional[str] = None, delay: bool = False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.baseFilename = filename
    
    def doRollover(self):
        """Perform rollover with gzip compression."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if os.path.exists(self.baseFilename):
            # Rotate existing backups
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}.gz"
                dfn = f"{self.baseFilename}.{i + 1}.gz"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            # Compress and move the current log file
            dfn = f"{self.baseFilename}.1.gz"
            if os.path.exists(dfn):
                os.remove(dfn)
            
            # Compress the current log
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            os.remove(self.baseFilename)
        
        if not self.delay:
            self.stream = self._open()


class TimedGzipRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Time-based rotating file handler with gzip compression.
    
    Features:
    - Rotates logs based on time (daily by default)
    - Compresses rotated logs with gzip
    - Supports retention policy (deletes old logs)
    """
    
    def __init__(self, filename: str, when: str = 'midnight', 
                 interval: int = 1, backupCount: int = 30,
                 encoding: Optional[str] = None, delay: bool = False,
                 utc: bool = False, atTime: Optional[datetime] = None,
                 retention_days: int = 30):
        super().__init__(filename, when, interval, backupCount, 
                        encoding, delay, utc, atTime)
        self.retention_days = retention_days
    
    def doRollover(self):
        """Perform rollover with gzip compression and cleanup."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Get the time that this sequence started at
        current_time = int(datetime.now().timestamp())
        
        # Find the rotation destination
        dst_now = datetime.now().strftime(self.suffix)
        
        if os.path.exists(self.baseFilename):
            # Compress the current log file
            dfn = self.baseFilename + "." + dst_now + ".gz"
            if os.path.exists(dfn):
                os.remove(dfn)
            
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            os.remove(self.baseFilename)
        
        if not self.delay:
            self.stream = self._open()
        
        # Clean up old logs
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove logs older than retention period."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        log_dir = Path(self.baseFilename).parent
        
        if not log_dir.exists():
            return
        
        for log_file in log_dir.glob("*.gz"):
            try:
                # Extract timestamp from filename
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    log_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, extra_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.extra_fields = extra_fields or {}
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'hostname': self.hostname,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add extra fields
        log_obj.update(self.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add any extra attributes from the record
        for key, value in record.__dict__.items():
            if key not in ['timestamp', 'level', 'logger', 'message', 'hostname',
                          'thread', 'threadName', 'exc_info', 'args', 'msg',
                          'created', 'msecs', 'relativeCreated', 'levelno',
                          'pathname', 'filename', 'module', 'lineno', 'funcName',
                          'exc_text', 'stack_info']:
                if not key.startswith('_'):
                    log_obj[key] = value
        
        return json.dumps(log_obj, default=str)


# ═══════════════════════════════════════════════════════════════
# LOG LEVEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Default log levels for different modules
MODULE_LOG_LEVELS = {
    'nexus': logging.INFO,
    'nexus.training': logging.INFO,
    'nexus.inference': logging.INFO,
    'nexus.data': logging.INFO,
    'nexus.model': logging.INFO,
    'transformers': logging.WARNING,
    'torch': logging.WARNING,
    'datasets': logging.WARNING,
    'urllib3': logging.WARNING,
    'requests': logging.WARNING,
}


def configure_module_levels(levels: Optional[Dict[str, int]] = None):
    """Configure log levels for specific modules."""
    levels = levels or MODULE_LOG_LEVELS
    for module, level in levels.items():
        logging.getLogger(module).setLevel(level)


# ═══════════════════════════════════════════════════════════════
# ENHANCED LOGGER SETUP
# ═══════════════════════════════════════════════════════════════

@dataclass
class LoggerConfig:
    """Configuration for logger setup."""
    name: str
    log_file: Optional[str] = None
    level: int = logging.INFO
    console_output: bool = True
    use_json_format: bool = False
    rotation_type: str = "size"  # "size" or "time"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    retention_days: int = 30
    when: str = "midnight"  # for time-based rotation
    extra_fields: Optional[Dict[str, Any]] = None


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Create a configured logger for data generation scripts.
    
    Backward compatible with the original implementation.
    For advanced features, use setup_logger_advanced().
    """
    config = LoggerConfig(
        name=name,
        log_file=log_file,
        level=level,
        console_output=console_output
    )
    return setup_logger_advanced(config)


def setup_logger_advanced(config: LoggerConfig) -> logging.Logger:
    """
    Create a configured logger with advanced features.
    
    Features:
    - Size-based or time-based rotation
    - gzip compression for old logs
    - JSON formatting option
    - Retention policy
    """
    logger = logging.getLogger(config.name)
    logger.setLevel(config.level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if config.use_json_format:
        formatter = JSONFormatter(extra_fields=config.extra_fields)
    else:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # File handler with rotation
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config.rotation_type == "size":
            file_handler = GzipRotatingFileHandler(
                config.log_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count
            )
        else:  # time-based
            file_handler = TimedGzipRotatingFileHandler(
                config.log_file,
                when=config.when,
                backupCount=config.backup_count,
                retention_days=config.retention_days
            )
        
        file_handler.setLevel(config.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        # Use simpler format for console
        console_formatter = logging.Formatter(
            CONSOLE_FORMAT if not config.use_json_format else LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_rotating_logger(
    name: str,
    log_dir: str = "logs",
    log_filename: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_gzip: bool = True,
    console_output: bool = True,
    use_json: bool = False
) -> logging.Logger:
    """
    Convenience function to set up a rotating file logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_filename: Log file name (default: {name}.log)
        level: Log level
        max_bytes: Maximum bytes per log file
        backup_count: Number of backup files to keep
        use_gzip: Whether to compress old logs
        console_output: Whether to output to console
        use_json: Whether to use JSON formatting
    """
    if log_filename is None:
        log_filename = f"{name}.log"
    
    log_file = os.path.join(log_dir, log_filename)
    
    config = LoggerConfig(
        name=name,
        log_file=log_file,
        level=level,
        console_output=console_output,
        use_json_format=use_json,
        rotation_type="size",
        max_bytes=max_bytes,
        backup_count=backup_count
    )
    
    return setup_logger_advanced(config)


def setup_daily_logger(
    name: str,
    log_dir: str = "logs",
    log_filename: Optional[str] = None,
    level: int = logging.INFO,
    retention_days: int = 30,
    console_output: bool = True,
    use_json: bool = False
) -> logging.Logger:
    """
    Convenience function to set up a daily rotating logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_filename: Log file name (default: {name}.log)
        level: Log level
        retention_days: Days to retain log files
        console_output: Whether to output to console
        use_json: Whether to use JSON formatting
    """
    if log_filename is None:
        log_filename = f"{name}.log"
    
    log_file = os.path.join(log_dir, log_filename)
    
    config = LoggerConfig(
        name=name,
        log_file=log_file,
        level=level,
        console_output=console_output,
        use_json_format=use_json,
        rotation_type="time",
        retention_days=retention_days
    )
    
    return setup_logger_advanced(config)


# ═══════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════

def log_progress(
    logger: logging.Logger,
    total: int,
    rate: float = 0,
    train: int = 0,
    val: int = 0,
    test: int = 0,
    dedup: int = 0,
    eta: float = 0
):
    """
    Log generation progress in consistent format.
    """
    msg = PROGRESS_TEMPLATE.format(
        total=total,
        rate=rate,
        train=train,
        val=val,
        test=test,
        dedup=dedup,
        eta=eta
    )
    logger.info(msg)


def log_header(
    logger: logging.Logger,
    title: str,
    config: Dict[str, Any]
):
    """
    Log generation header with configuration.
    """
    logger.info("=" * 60)
    logger.info(f"🚀 {title}")
    
    for key, value in config.items():
        if isinstance(value, int):
            logger.info(f"   {key}: {value:,}")
        else:
            logger.info(f"   {key}: {value}")
    
    logger.info("=" * 60)


def log_completion(
    logger: logging.Logger,
    title: Union[str, int],
    train: Any = None,
    val: Any = None,
    test: Any = None,
    dedup: Any = None,
    elapsed_hours: float = 0
):
    """
    Log generation completion summary.
    Supports both (logger, title, results_dict) and (logger, total, train, val, test, dedup, time)
    """
    logger.info("=" * 60)
    logger.info("✅ GENERATION COMPLETE")
    
    if isinstance(title, str) and isinstance(train, dict):
        # Format: log_completion(logger, "Title", {"Total": 100, ...})
        logger.info(f"   Task: {title}")
        for k, v in train.items():
            if isinstance(v, int):
                logger.info(f"   {k}: {v:,}")
            else:
                logger.info(f"   {k}: {v}")
    else:
        # Format: log_completion(logger, total, train, val, test, dedup, elapsed)
        logger.info(f"   Total samples: {title:,}" if isinstance(title, int) else f"   {title}")
        if train is not None: logger.info(f"   Train: {train:,}")
        if val is not None: logger.info(f"   Val: {val:,}")
        if test is not None: logger.info(f"   Test: {test:,}")
        if dedup is not None: logger.info(f"   Duplicates skipped: {dedup:,}")
        if elapsed_hours: logger.info(f"   Time: {elapsed_hours:.2f} hours")
        
    logger.info("=" * 60)


def log_benchmark_progress(
    logger: logging.Logger,
    name: str,
    split: str,
    current: int,
    total: int,
    status: str = "Processing"
):
    """
    Log benchmark processing progress.
    """
    msg = BENCHMARK_TEMPLATE.format(
        name=name,
        split=split,
        current=current,
        total=total if total else "?",
        status=status
    )
    logger.info(msg)


def log_structured(
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
    **kwargs
):
    """Log a structured message with additional fields."""
    extra = {'structured_data': kwargs}
    logger.log(level, message, extra=extra)


# ═══════════════════════════════════════════════════════════════
# GLOBAL LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

_logging_initialized = False
_init_lock = threading.Lock()


def init_logging(
    log_dir: str = "logs",
    default_level: int = logging.INFO,
    module_levels: Optional[Dict[str, int]] = None,
    use_json: bool = False,
    enable_rotation: bool = True
):
    """
    Initialize global logging configuration.
    
    This should be called once at application startup.
    """
    global _logging_initialized
    
    with _init_lock:
        if _logging_initialized:
            return
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(default_level)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Configure module levels
        configure_module_levels(module_levels)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(default_level)
        formatter = JSONFormatter() if use_json else logging.Formatter(
            LOG_FORMAT, datefmt=LOG_DATE_FORMAT
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_rotation:
            log_file = os.path.join(log_dir, "nexus.log")
            if use_json:
                file_handler = TimedGzipRotatingFileHandler(
                    log_file,
                    when="midnight",
                    backupCount=30,
                    retention_days=30
                )
            else:
                file_handler = GzipRotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
            file_handler.setLevel(default_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        _logging_initialized = True
        root_logger.info(f"Logging initialized. Log directory: {log_dir}")


def shutdown_logging():
    """Shutdown logging and flush all handlers."""
    logging.shutdown()


# ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════

# Keep the original function signatures for backward compatibility
__all__ = [
    'setup_logger',
    'setup_logger_advanced',
    'setup_rotating_logger',
    'setup_daily_logger',
    'log_progress',
    'log_header',
    'log_completion',
    'log_benchmark_progress',
    'log_structured',
    'init_logging',
    'shutdown_logging',
    'LoggerConfig',
    'GzipRotatingFileHandler',
    'TimedGzipRotatingFileHandler',
    'JSONFormatter',
]

"""
utils/logging_config.py
Shared logging configuration for all data generation scripts.
Ensures consistent log format across finetuned, repetitive, and other generators.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any


# ═══════════════════════════════════════════════════════════════
# LOG FORMAT TEMPLATE
# ═══════════════════════════════════════════════════════════════

LOG_FORMAT = "%(asctime)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Progress log template (consistent across all generators)
PROGRESS_TEMPLATE = (
    "✓ Total: {total:,} ({rate:.0f}/sec) | "
    "Train: {train:,} Val: {val:,} Test: {test:,} | "
    "Dedup: {dedup} | ETA: {eta:.1f}h"
)


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Create a configured logger for data generation scripts.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (relative to project root)
        level: Logging level
        console_output: Whether to also output to console
        
    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_progress(
    logger: logging.Logger,
    total: int,
    rate: float,
    train: int,
    val: int,
    test: int,
    dedup: int,
    eta: float
):
    """
    Log generation progress in consistent format.
    
    Args:
        logger: Logger instance
        total: Total samples generated
        rate: Samples per second
        train: Train split count
        val: Validation split count
        test: Test split count
        dedup: Duplicates skipped
        eta: Estimated time remaining in hours
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
    
    Args:
        logger: Logger instance
        title: Generation title (e.g., "FINETUNED DATASET GENERATION")
        config: Configuration dictionary
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
    total: int,
    train: int,
    val: int,
    test: int,
    dedup: int,
    elapsed_hours: float
):
    """
    Log generation completion summary.
    """
    logger.info("=" * 60)
    logger.info("✅ GENERATION COMPLETE")
    logger.info(f"   Total samples: {total:,}")
    logger.info(f"   Train: {train:,}")
    logger.info(f"   Val: {val:,}")
    logger.info(f"   Test: {test:,}")
    logger.info(f"   Duplicates skipped: {dedup:,}")
    logger.info(f"   Time: {elapsed_hours:.2f} hours")
    logger.info("=" * 60)


# Benchmark log template
BENCHMARK_TEMPLATE = (
    "📥 {name:<10} | Split: {split:<10} | "
    "Processed: {current:>6}/{total:<6} | Status: {status}"
)

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

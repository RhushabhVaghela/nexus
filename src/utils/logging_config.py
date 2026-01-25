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
    title: Any, # Can be str or total int (compatibility)
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

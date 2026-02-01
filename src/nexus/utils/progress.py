"""
src/utils/progress.py

Progress bar utilities with tqdm integration for all long operations.
Provides training epoch progress, data loading progress, model download progress,
multi-stage pipeline progress, and nested progress bars for concurrent operations.
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any, List, Callable, Iterator, Union, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to import tqdm
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    from tqdm.contrib.concurrent import process_map, thread_map
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available. Progress bars will be disabled.")


@dataclass
class ProgressStats:
    """Statistics for tracking progress."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.completed / elapsed
    
    @property
    def estimated_time_remaining(self) -> float:
        """Get estimated time remaining in seconds."""
        rate = self.items_per_second
        if rate == 0:
            return 0.0
        remaining = self.total - self.completed
        return remaining / rate
    
    def format_eta(self) -> str:
        """Format estimated time remaining."""
        eta = self.estimated_time_remaining
        if eta < 60:
            return f"{eta:.0f}s"
        elif eta < 3600:
            return f"{eta/60:.1f}m"
        else:
            return f"{eta/3600:.1f}h"
    
    def format_elapsed(self) -> str:
        """Format elapsed time."""
        elapsed = self.elapsed_time
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}m"
        else:
            return f"{elapsed/3600:.1f}h"


class ProgressManager:
    """Manager for progress bars with support for nested and concurrent operations."""
    
    def __init__(self, enabled: bool = True, 
                 default_desc: str = "Processing",
                 position: Optional[int] = None):
        """Initialize the progress manager."""
        self.enabled = enabled and TQDM_AVAILABLE
        self.default_desc = default_desc
        self.position = position
        self._bars: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def create_bar(self, 
                   total: Optional[int] = None,
                   desc: Optional[str] = None,
                   unit: str = "it",
                   leave: bool = True,
                   position: Optional[int] = None,
                   bar_id: Optional[str] = None,
                   **kwargs) -> Any:
        """Create a new progress bar."""
        if not self.enabled:
            return DummyProgressBar(total)
        
        desc = desc or self.default_desc
        position = position if position is not None else self.position
        
        bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            position=position,
            dynamic_ncols=True,
            **kwargs
        )
        
        if bar_id:
            with self._lock:
                self._bars[bar_id] = bar
        
        return bar
    
    def get_bar(self, bar_id: str) -> Optional[Any]:
        """Get a progress bar by ID."""
        with self._lock:
            return self._bars.get(bar_id)
    
    def update_bar(self, bar_id: str, n: int = 1, **kwargs):
        """Update a progress bar by ID."""
        bar = self.get_bar(bar_id)
        if bar:
            bar.update(n)
            if kwargs:
                bar.set_postfix(**kwargs)
    
    def close_bar(self, bar_id: str):
        """Close a progress bar by ID."""
        with self._lock:
            bar = self._bars.pop(bar_id, None)
            if bar:
                bar.close()
    
    def close_all(self):
        """Close all progress bars."""
        with self._lock:
            for bar in self._bars.values():
                bar.close()
            self._bars.clear()
    
    @contextmanager
    def bar(self, iterable: Optional[Iterator] = None, 
            total: Optional[int] = None,
            desc: Optional[str] = None,
            **kwargs):
        """Context manager for a progress bar."""
        bar = self.create_bar(total=total, desc=desc, **kwargs)
        try:
            if iterable is not None:
                yield bar(iterable)
            else:
                yield bar
        finally:
            bar.close()


class DummyProgressBar:
    """Dummy progress bar for when tqdm is not available."""
    
    def __init__(self, total: Optional[int] = None):
        self.total = total
        self.n = 0
    
    def update(self, n: int = 1):
        self.n += n
    
    def set_postfix(self, **kwargs):
        pass
    
    def set_description(self, desc: str):
        pass
    
    def close(self):
        pass
    
    def __iter__(self):
        return iter([])
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class TrainingProgress:
    """Progress tracking for training with epoch, loss, and accuracy metrics."""
    
    def __init__(self, 
                 epochs: int,
                 steps_per_epoch: Optional[int] = None,
                 desc: str = "Training",
                 enabled: bool = True):
        """Initialize training progress."""
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.desc = desc
        self.enabled = enabled and TQDM_AVAILABLE
        self.stats = ProgressStats(total=epochs)
        self._epoch_bar = None
        self._step_bar = None
        self._current_epoch = 0
        self._losses: List[float] = []
        self._accuracies: List[float] = []
    
    def start(self):
        """Start training progress tracking."""
        self.stats.start_time = time.time()
        if self.enabled:
            self._epoch_bar = tqdm(
                total=self.epochs,
                desc=self.desc,
                unit="epoch",
                position=0,
                leave=True,
                dynamic_ncols=True
            )
    
    def start_epoch(self, epoch: int):
        """Start a new epoch."""
        self._current_epoch = epoch
        if self.enabled and self.steps_per_epoch:
            self._step_bar = tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch}/{self.epochs}",
                unit="step",
                position=1,
                leave=False,
                dynamic_ncols=True
            )
    
    def update_step(self, 
                   loss: Optional[float] = None,
                   accuracy: Optional[float] = None,
                   learning_rate: Optional[float] = None,
                   **kwargs):
        """Update step progress."""
        if self._step_bar and self.enabled:
            self._step_bar.update(1)
            
            postfix = {}
            if loss is not None:
                self._losses.append(loss)
                postfix['loss'] = f"{loss:.4f}"
            if accuracy is not None:
                self._accuracies.append(accuracy)
                postfix['acc'] = f"{accuracy:.4f}"
            if learning_rate is not None:
                postfix['lr'] = f"{learning_rate:.2e}"
            postfix.update(kwargs)
            
            if postfix:
                self._step_bar.set_postfix(**postfix)
    
    def end_epoch(self, 
                 epoch_loss: Optional[float] = None,
                 epoch_accuracy: Optional[float] = None,
                 **kwargs):
        """End the current epoch."""
        self.stats.completed += 1
        
        if self._epoch_bar and self.enabled:
            postfix = {}
            if epoch_loss is not None:
                postfix['loss'] = f"{epoch_loss:.4f}"
            if epoch_accuracy is not None:
                postfix['acc'] = f"{epoch_accuracy:.4f}"
            postfix.update(kwargs)
            
            if postfix:
                self._epoch_bar.set_postfix(**postfix)
            
            self._epoch_bar.update(1)
        
        if self._step_bar:
            self._step_bar.close()
            self._step_bar = None
    
    def finish(self, final_message: Optional[str] = None):
        """Finish training progress."""
        self.stats.end_time = time.time()
        
        if self._step_bar:
            self._step_bar.close()
        
        if self._epoch_bar:
            if final_message:
                self._epoch_bar.set_description(final_message)
            self._epoch_bar.close()
        
        # Log summary
        logger.info(f"Training completed in {self.stats.format_elapsed()}")
        if self._losses:
            avg_loss = sum(self._losses) / len(self._losses)
            logger.info(f"Average loss: {avg_loss:.4f}")
        if self._accuracies:
            avg_acc = sum(self._accuracies) / len(self._accuracies)
            logger.info(f"Average accuracy: {avg_acc:.4f}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class DataLoadingProgress:
    """Progress tracking for data loading operations."""
    
    def __init__(self,
                 total_samples: Optional[int] = None,
                 desc: str = "Loading data",
                 batch_size: int = 1,
                 enabled: bool = True):
        """Initialize data loading progress."""
        self.total_samples = total_samples
        self.desc = desc
        self.batch_size = batch_size
        self.enabled = enabled and TQDM_AVAILABLE
        self.stats = ProgressStats(total=total_samples or 0)
        self._bar = None
    
    def start(self):
        """Start data loading progress."""
        self.stats.start_time = time.time()
        if self.enabled:
            total_batches = (self.total_samples // self.batch_size) if self.total_samples else None
            self._bar = tqdm(
                total=total_batches,
                desc=self.desc,
                unit="batch",
                dynamic_ncols=True
            )
    
    def update(self, n_batches: int = 1, n_samples: Optional[int] = None, **kwargs):
        """Update progress."""
        if self._bar and self.enabled:
            self._bar.update(n_batches)
            if kwargs:
                self._bar.set_postfix(**kwargs)
        
        if n_samples:
            self.stats.completed += n_samples
    
    def set_postfix(self, **kwargs):
        """Set postfix information."""
        if self._bar:
            self._bar.set_postfix(**kwargs)
    
    def finish(self):
        """Finish data loading."""
        self.stats.end_time = time.time()
        if self._bar:
            self._bar.close()
        
        logger.info(f"Data loading completed: {self.stats.completed} samples in {self.stats.format_elapsed()}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class DownloadProgress:
    """Progress tracking for model/file downloads with speed and ETA."""
    
    def __init__(self,
                 total_bytes: Optional[int] = None,
                 desc: str = "Downloading",
                 enabled: bool = True):
        """Initialize download progress."""
        self.total_bytes = total_bytes
        self.desc = desc
        self.enabled = enabled and TQDM_AVAILABLE
        self._bar = None
        self._downloaded = 0
    
    def start(self):
        """Start download progress."""
        if self.enabled:
            self._bar = tqdm(
                total=self.total_bytes,
                desc=self.desc,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True
            )
    
    def update(self, n_bytes: int):
        """Update download progress."""
        self._downloaded += n_bytes
        if self._bar and self.enabled:
            self._bar.update(n_bytes)
    
    def set_total(self, total_bytes: int):
        """Set or update total bytes."""
        self.total_bytes = total_bytes
        if self._bar:
            self._bar.total = total_bytes
            self._bar.refresh()
    
    def finish(self):
        """Finish download."""
        if self._bar:
            self._bar.close()
        
        downloaded_mb = self._downloaded / (1024 * 1024)
        logger.info(f"Download completed: {downloaded_mb:.2f} MB")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class PipelineProgress:
    """Progress tracking for multi-stage pipelines."""
    
    def __init__(self,
                 stages: List[str],
                 desc: str = "Pipeline",
                 enabled: bool = True):
        """Initialize pipeline progress."""
        self.stages = stages
        self.desc = desc
        self.enabled = enabled and TQDM_AVAILABLE
        self._stage_bars: Dict[str, Any] = {}
        self._current_stage_idx = 0
        self._overall_bar = None
    
    def start(self):
        """Start pipeline progress."""
        if self.enabled:
            self._overall_bar = tqdm(
                total=len(self.stages),
                desc=self.desc,
                unit="stage",
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format='{desc}: {n_fmt}/{total_fmt} stages [{bar:40}] {percentage:3.0f}% | {elapsed}<{remaining}'
            )
    
    def start_stage(self, stage_name: str, 
                   total: Optional[int] = None,
                   unit: str = "it"):
        """Start a pipeline stage."""
        if not self.enabled:
            return DummyProgressBar(total)
        
        # Close previous stage bar if exists
        if stage_name in self._stage_bars:
            self._stage_bars[stage_name].close()
        
        bar = tqdm(
            total=total,
            desc=f"  {stage_name}",
            unit=unit,
            position=1,
            leave=False,
            dynamic_ncols=True
        )
        
        self._stage_bars[stage_name] = bar
        return bar
    
    def end_stage(self, stage_name: str):
        """End a pipeline stage."""
        if stage_name in self._stage_bars:
            self._stage_bars[stage_name].close()
            del self._stage_bars[stage_name]
        
        if self._overall_bar:
            self._overall_bar.update(1)
            self._current_stage_idx += 1
    
    def update_stage(self, stage_name: str, n: int = 1, **kwargs):
        """Update a stage's progress."""
        if stage_name in self._stage_bars:
            bar = self._stage_bars[stage_name]
            bar.update(n)
            if kwargs:
                bar.set_postfix(**kwargs)
    
    def finish(self):
        """Finish pipeline."""
        for bar in self._stage_bars.values():
            bar.close()
        self._stage_bars.clear()
        
        if self._overall_bar:
            self._overall_bar.close()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class ConcurrentProgress:
    """Progress tracking for concurrent operations with nested bars."""
    
    def __init__(self,
                 total_operations: int,
                 desc: str = "Concurrent operations",
                 enabled: bool = True):
        """Initialize concurrent progress."""
        self.total_operations = total_operations
        self.desc = desc
        self.enabled = enabled and TQDM_AVAILABLE
        self._bars: List[Any] = []
        self._completed = 0
    
    def create_operation_bar(self,
                            operation_id: str,
                            total: Optional[int] = None,
                            desc: Optional[str] = None) -> Any:
        """Create a progress bar for a concurrent operation."""
        if not self.enabled:
            return DummyProgressBar(total)
        
        position = len(self._bars) + 1
        bar = tqdm(
            total=total,
            desc=desc or operation_id,
            position=position,
            leave=False,
            dynamic_ncols=True
        )
        
        self._bars.append(bar)
        return bar
    
    def complete_operation(self, bar: Any):
        """Mark an operation as complete."""
        if bar in self._bars:
            bar.close()
            self._bars.remove(bar)
            self._completed += 1
    
    def close_all(self):
        """Close all operation bars."""
        for bar in self._bars:
            bar.close()
        self._bars.clear()


# Convenience functions for common use cases

def progress_iter(iterable: Iterator,
                  total: Optional[int] = None,
                  desc: str = "Processing",
                  unit: str = "it",
                  enabled: bool = True,
                  **kwargs) -> Iterator:
    """Wrap an iterable with a progress bar."""
    if not enabled or not TQDM_AVAILABLE:
        return iterable
    
    return tqdm(iterable, total=total, desc=desc, unit=unit, **kwargs)


@contextmanager
def progress_context(total: Optional[int] = None,
                    desc: str = "Processing",
                    unit: str = "it",
                    enabled: bool = True,
                    **kwargs):
    """Context manager for a simple progress bar."""
    if not enabled or not TQDM_AVAILABLE:
        yield DummyProgressBar(total)
    else:
        bar = tqdm(total=total, desc=desc, unit=unit, **kwargs)
        try:
            yield bar
        finally:
            bar.close()


def download_with_progress(url: str,
                          output_path: str,
                          desc: Optional[str] = None,
                          chunk_size: int = 8192,
                          enabled: bool = True) -> str:
    """Download a file with progress bar."""
    import requests
    
    desc = desc or f"Downloading {os.path.basename(url)}"
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with DownloadProgress(total_bytes=total_size, desc=desc, enabled=enabled) as progress:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    
    return output_path


def map_with_progress(func: Callable,
                     iterable: Iterator,
                     total: Optional[int] = None,
                     desc: str = "Processing",
                     max_workers: Optional[int] = None,
                     use_threads: bool = True,
                     enabled: bool = True) -> List[Any]:
    """Map a function over an iterable with progress tracking."""
    if not enabled or not TQDM_AVAILABLE:
        return list(map(func, iterable))
    
    if max_workers is None or max_workers == 1:
        # Sequential processing with progress bar
        results = []
        for item in tqdm(iterable, total=total, desc=desc):
            results.append(func(item))
        return results
    
    # Parallel processing with progress bar
    if use_threads:
        return thread_map(func, iterable, max_workers=max_workers, desc=desc, total=total)
    else:
        return process_map(func, iterable, max_workers=max_workers, desc=desc, total=total)


# Decorator for adding progress to functions

def with_progress(desc: Optional[str] = None, 
                  unit: str = "it",
                  show_eta: bool = True):
    """Decorator to add progress bar to a function."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Try to get total from kwargs or first argument
            total = kwargs.get('total')
            if total is None and len(args) > 0 and isinstance(args[0], (list, tuple)):
                total = len(args[0])
            
            description = desc or func.__name__
            
            with progress_context(total=total, desc=description, unit=unit) as bar:
                kwargs['_progress_bar'] = bar
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global progress manager instance
_global_manager: Optional[ProgressManager] = None

def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ProgressManager()
    return _global_manager


def set_progress_enabled(enabled: bool):
    """Enable or disable progress bars globally."""
    manager = get_progress_manager()
    manager.enabled = enabled and TQDM_AVAILABLE


# Example usage and testing

def _example_usage():
    """Example of how to use the progress utilities."""
    import time
    import random
    
    # Example 1: Simple iteration
    print("Example 1: Simple iteration")
    for item in progress_iter(range(100), desc="Processing items"):
        time.sleep(0.01)
    
    # Example 2: Training progress
    print("\nExample 2: Training progress")
    with TrainingProgress(epochs=3, steps_per_epoch=50, desc="Training model") as train_prog:
        for epoch in range(1, 4):
            train_prog.start_epoch(epoch)
            for step in range(50):
                loss = 1.0 / (step + 1)
                acc = 0.5 + step / 100
                train_prog.update_step(loss=loss, accuracy=acc, learning_rate=1e-4)
                time.sleep(0.01)
            train_prog.end_epoch(epoch_loss=0.5, epoch_accuracy=0.75)
    
    # Example 3: Pipeline progress
    print("\nExample 3: Pipeline progress")
    stages = ["Load data", "Preprocess", "Train", "Evaluate", "Save"]
    with PipelineProgress(stages, desc="ML Pipeline") as pipe:
        for stage in stages:
            stage_bar = pipe.start_stage(stage, total=20)
            for i in range(20):
                stage_bar.update(1)
                time.sleep(0.05)
            pipe.end_stage(stage)
    
    # Example 4: Download progress
    print("\nExample 4: Download progress")
    with DownloadProgress(total_bytes=1024*1024*10, desc="Downloading model") as dl:
        for _ in range(100):
            dl.update(1024*1024*0.1)
            time.sleep(0.01)


if __name__ == "__main__":
    _example_usage()

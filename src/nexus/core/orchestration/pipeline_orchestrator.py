#!/usr/bin/env python3
"""
pipeline_orchestrator.py
Unified orchestrator for chaining multiple training stages together.

This module provides a comprehensive pipeline orchestration system that:
1. Chains stages sequentially (Reasoning → Tools → Vision)
2. Manages inter-stage data flow and checkpoints
3. Handles stage dependencies with parallel execution support
4. Provides unified progress tracking and error recovery
5. Supports rollback, continue, and stop failure modes

Classes:
- PipelineConfig: Configuration for the entire pipeline
- PipelineStage: Wrapper for individual stages
- CheckpointManager: Manages stage checkpoints
- PipelineResult: Result dataclass for pipeline execution
- PipelineProgress: Progress tracking dataclass
- PipelineError: Error handling dataclass
- PipelineOrchestrator: Main orchestrator class

Example Usage:
    config = PipelineConfig(
        stages=["reasoning", "tools", "vision"],
        dependencies={"tools": ["reasoning"], "vision": ["reasoning", "tools"]},
        checkpoint_dir="checkpoints/pipeline",
        parallel_stages=["reasoning", "tools"],
        failure_mode="rollback"
    )

    orchestrator = PipelineOrchestrator(config)
    orchestrator.add_stage(reasoning_stage, dependencies=[])
    orchestrator.add_stage(tools_stage, dependencies=["reasoning"])
    orchestrator.add_stage(vision_stage, dependencies=["reasoning", "tools"])

    result = orchestrator.execute_pipeline({"input_data": "..."})
"""

import os
import sys
import json
import time
import logging
import traceback
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import with try/except to handle missing dependencies gracefully
try:
    from pydantic import BaseModel, ValidationError

    PYDANTIC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Create minimal pydantic-like classes for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    class ValidationError(Exception):
        pass

    PYDANTIC_AVAILABLE = False


class FailureMode(Enum):
    """
    Failure modes for pipeline execution.

    Attributes:
        STOP: Stop execution on first failure
        CONTINUE: Continue to next stage on failure
        ROLLBACK: Rollback to last checkpoint on failure
    """

    STOP = "stop"
    CONTINUE = "continue"
    ROLLBACK = "rollback"


class StageStatus(Enum):
    """
    Status of a stage in the pipeline.

    Attributes:
        PENDING: Stage not yet executed
        RUNNING: Stage currently executing
        COMPLETED: Stage finished successfully
        FAILED: Stage execution failed
        SKIPPED: Stage was skipped (dependencies not met)
        ROLLBACK: Stage is being rolled back
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLBACK = "rollback"


@dataclass
class PipelineConfig:
    """
    Configuration for the entire pipeline.

    This class defines all configuration parameters for pipeline execution,
    including stage ordering, dependencies, checkpoint management, and error handling.

    Attributes:
        stages: List of stage names in execution order
        dependencies: Dict mapping stage names to their dependencies
        checkpoint_dir: Directory for pipeline checkpoints
        resume_from: Stage name to resume from (None for fresh start)
        parallel_stages: List of stages that can run in parallel
        failure_mode: How to handle stage failures (stop, continue, rollback)
        timeout_per_stage: Optional timeout in seconds per stage
        enable_logging: Enable detailed logging
        save_intermediate_outputs: Save outputs between stages
        max_workers: Maximum parallel workers for concurrent stages
        enable_rollback: Enable automatic rollback on failure
        rollback_stages: Number of stages to rollback on failure
    """

    stages: List[str]
    dependencies: Dict[str, List[str]]
    checkpoint_dir: str = "checkpoints/pipeline"
    resume_from: Optional[str] = None
    parallel_stages: List[str] = field(default_factory=list)
    failure_mode: str = "stop"
    timeout_per_stage: Optional[int] = None
    enable_logging: bool = True
    save_intermediate_outputs: bool = True
    max_workers: int = 2
    enable_rollback: bool = False
    rollback_stages: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage")

        # Check all dependency keys are valid stages
        if not set(self.dependencies.keys()).issubset(set(self.stages)):
            invalid_deps = set(self.dependencies.keys()) - set(self.stages)
            raise ValueError(f"Dependencies contain invalid stages: {invalid_deps}")

        # Check all dependency values reference valid stages
        all_stages_set = set(self.stages)
        for stage_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in all_stages_set:
                    raise ValueError(
                        f"Dependency '{dep}' of stage '{stage_name}' is not a valid stage"
                    )

        # Validate resume_from points to a valid stage
        if self.resume_from and self.resume_from not in self.stages:
            raise ValueError(f"resume_from stage '{self.resume_from}' not in stages")

        # Validate failure_mode
        if self.failure_mode not in [mode.value for mode in FailureMode]:
            raise ValueError(f"Invalid failure_mode: {self.failure_mode}")

        # Validate parallel_stages are in stages list
        invalid_parallel = set(self.parallel_stages) - set(self.stages)
        if invalid_parallel:
            raise ValueError(
                f"Parallel stages contain invalid stages: {invalid_parallel}"
            )

        # Validate no circular dependencies
        self._validate_dependencies()

        # Log configuration
        self._log_config()

    def _log_config(self) -> None:
        """Log configuration summary."""
        config_logger = logging.getLogger("pipeline_config")
        config_logger.info(f"Pipeline Config: {len(self.stages)} stages")
        config_logger.info(f"  - Stages: {self.stages}")
        config_logger.info(f"  - Parallel stages: {self.parallel_stages}")
        config_logger.info(f"  - Failure mode: {self.failure_mode}")
        config_logger.info(f"  - Checkpoint dir: {self.checkpoint_dir}")

    def _validate_dependencies(self) -> None:
        """
        Check for circular dependencies using DFS algorithm.

        Raises:
            ValueError: If circular dependency is detected
        """

        def get_deps(stage: str, visited: Set[str], stack: Set[str]) -> bool:
            """
            Detect circular dependencies using depth-first search.

            Args:
                stage: Current stage to check
                visited: Set of already visited stages
                stack: Set of stages in current DFS path

            Returns:
                True if circular dependency found, False otherwise
            """
            if stage in stack:
                return True
            if stage in visited:
                return False

            visited.add(stage)
            stack.add(stage)

            for dep in self.dependencies.get(stage, []):
                if get_deps(dep, visited.copy(), stack.copy()):
                    return True

            stack.remove(stage)
            return False

        for stage in self.stages:
            if get_deps(stage, set(), set()):
                raise ValueError(
                    f"Circular dependency detected involving stage: {stage}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "stages": self.stages,
            "dependencies": self.dependencies,
            "checkpoint_dir": self.checkpoint_dir,
            "resume_from": self.resume_from,
            "parallel_stages": self.parallel_stages,
            "failure_mode": self.failure_mode,
            "timeout_per_stage": self.timeout_per_stage,
            "enable_logging": self.enable_logging,
            "save_intermediate_outputs": self.save_intermediate_outputs,
            "max_workers": self.max_workers,
            "enable_rollback": self.enable_rollback,
            "rollback_stages": self.rollback_stages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from dictionary."""
        return cls(
            stages=data["stages"],
            dependencies=data["dependencies"],
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints/pipeline"),
            resume_from=data.get("resume_from"),
            parallel_stages=data.get("parallel_stages", []),
            failure_mode=data.get("failure_mode", "stop"),
            timeout_per_stage=data.get("timeout_per_stage"),
            enable_logging=data.get("enable_logging", True),
            save_intermediate_outputs=data.get("save_intermediate_outputs", True),
            max_workers=data.get("max_workers", 2),
            enable_rollback=data.get("enable_rollback", False),
            rollback_stages=data.get("rollback_stages", 1),
        )


@dataclass
class CheckpointInfo:
    """
    Information about a saved checkpoint.

    This dataclass encapsulates metadata about a checkpoint saved during
    pipeline execution, including timing, step count, and custom metadata.

    Attributes:
        stage_name: Name of the stage that created this checkpoint
        checkpoint_path: File system path to the checkpoint
        timestamp: ISO format timestamp of checkpoint creation
        step: Training step number at checkpoint time
        metadata: Additional metadata dictionary
    """

    stage_name: str
    checkpoint_path: str
    timestamp: str
    step: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint info to dictionary."""
        return {
            "stage_name": self.stage_name,
            "checkpoint_path": self.checkpoint_path,
            "timestamp": self.timestamp,
            "step": self.step,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Create CheckpointInfo from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            checkpoint_path=data["checkpoint_path"],
            timestamp=data["timestamp"],
            step=data["step"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PipelineError:
    """
    Represents an error that occurred during pipeline execution.

    This class captures comprehensive error information including stage name,
    error type, message, timestamp, stack trace, and recoverability status.

    Attributes:
        stage_name: Name of the stage where error occurred
        error_type: Type/class of the error
        error_message: Human-readable error message
        timestamp: ISO format timestamp of error occurrence
        stack_trace: Optional stack trace string
        recoverable: Whether the error allows recovery/continuation
        context: Additional context information
    """

    stage_name: str
    error_type: str
    error_message: str
    timestamp: str
    stack_trace: Optional[str] = None
    recoverable: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "stage_name": self.stage_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
            "recoverable": self.recoverable,
            "context": self.context,
        }

    @classmethod
    def from_exception(
        cls, stage_name: str, exception: Exception, recoverable: bool = False
    ) -> "PipelineError":
        """Create PipelineError from an exception instance."""
        return cls(
            stage_name=stage_name,
            error_type=type(exception).__name__,
            error_message=str(exception),
            timestamp=datetime.now().isoformat(),
            stack_trace=traceback.format_exc(),
            recoverable=recoverable,
        )


@dataclass
class PipelineProgress:
    """
    Tracks progress of pipeline execution.

    This dataclass provides real-time information about pipeline execution
    including completion status, timing, and checkpoint information.

    Attributes:
        current_stage: Name of currently executing stage
        total_stages: Total number of stages in pipeline
        completed_stages: List of successfully completed stage names
        failed_stages: List of stage names that failed
        skipped_stages: List of stage names that were skipped
        current_step: Current step within the current stage
        total_steps: Total steps in the current stage
        start_time: ISO format timestamp when pipeline started
        elapsed_seconds: Elapsed time in seconds
        last_checkpoint: Information about the last checkpoint
        stage_start_time: Timestamp when current stage started
    """

    current_stage: Optional[str]
    total_stages: int
    completed_stages: List[str]
    failed_stages: List[str]
    skipped_stages: List[str]
    current_step: int
    total_steps: int
    start_time: Optional[str]
    elapsed_seconds: float
    last_checkpoint: Optional[CheckpointInfo]
    stage_start_time: Optional[str] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage based on completed stages."""
        total = self.total_stages
        if total == 0:
            return 0.0
        completed = len(self.completed_stages)
        return (completed / total) * 100

    @property
    def is_complete(self) -> bool:
        """Check if pipeline execution is complete."""
        completed_or_skipped = len(self.completed_stages) + len(self.skipped_stages)
        return completed_or_skipped == self.total_stages or len(self.failed_stages) > 0

    @property
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully with no failures."""
        return self.is_complete and len(self.failed_stages) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary."""
        return {
            "current_stage": self.current_stage,
            "total_stages": self.total_stages,
            "completed_stages": self.completed_stages,
            "failed_stages": self.failed_stages,
            "skipped_stages": self.skipped_stages,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "start_time": self.start_time,
            "elapsed_seconds": self.elapsed_seconds,
            "progress_percentage": self.progress_percentage,
            "is_complete": self.is_complete,
            "is_successful": self.is_successful,
            "last_checkpoint": self.last_checkpoint.to_dict()
            if self.last_checkpoint
            else None,
            "stage_start_time": self.stage_start_time,
        }


@dataclass
class PipelineResult:
    """
    Result of pipeline execution.

    This dataclass encapsulates the complete result of pipeline execution
    including success status, outputs, checkpoints, timing, and errors.

    Attributes:
        success: Whether pipeline completed successfully
        outputs: Dictionary of stage outputs keyed by stage name
        checkpoints: List of all checkpoints created during execution
        duration: Total execution duration in seconds
        errors: List of errors that occurred during execution
        progress: Final pipeline progress information
        final_stage: Name of the last executed stage
        rollback_performed: Whether rollback was triggered
    """

    success: bool
    outputs: Dict[str, Any]
    checkpoints: List[CheckpointInfo]
    duration: float
    errors: List[PipelineError]
    progress: PipelineProgress
    final_stage: Optional[str] = None
    rollback_performed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "outputs": self.outputs,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "duration": self.duration,
            "errors": [err.to_dict() for err in self.errors],
            "progress": self.progress.to_dict(),
            "final_stage": self.final_stage,
            "rollback_performed": self.rollback_performed,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save result to JSON file."""
        if isinstance(filepath, Path):
            filepath = str(filepath)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = [
            f"Pipeline {'SUCCEEDED' if self.success else 'FAILED'}",
            f"Duration: {self.duration:.2f}s",
            f"Stages: {len(self.progress.completed_stages)}/{self.progress.total_stages} completed",
            f"Errors: {len(self.errors)}",
        ]
        if self.errors:
            lines.append("Errors:")
            for err in self.errors[:5]:  # Show first 5 errors
                lines.append(f"  - {err.stage_name}: {err.error_message}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more errors")
        return "\n".join(lines)


class CheckpointManager:
    """
    Manages checkpoints for pipeline stages.

    This class handles saving, loading, and listing checkpoints
    for all pipeline stages. It provides persistence and recovery
    capabilities for long-running pipelines with automatic cleanup.

    Attributes:
        checkpoint_dir: Base directory for all checkpoints
        max_checkpoints: Maximum checkpoints to keep per stage (0 = unlimited)
        logger: Logger instance for checkpoint operations
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/pipeline",
        max_checkpoints: int = 5,
        auto_cleanup: bool = True,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Base directory for storing checkpoints
            max_checkpoints: Maximum checkpoints per stage (0 = unlimited)
            auto_cleanup: Whether to automatically cleanup old checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger("checkpoint_manager")

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create stage-specific directories
        self.stage_dirs: Dict[str, Path] = {}

        # Thread-safe operations
        self._lock = Lock()

        self.logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def _get_stage_dir(self, stage_name: str) -> Path:
        """Get or create directory for a stage's checkpoints."""
        if stage_name not in self.stage_dirs:
            stage_dir = self.checkpoint_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            self.stage_dirs[stage_name] = stage_dir
        return self.stage_dirs[stage_name]

    def save_checkpoint(
        self,
        stage_name: str,
        state: Dict[str, Any],
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> CheckpointInfo:
        """
        Save a checkpoint for a stage.

        Args:
            stage_name: Name of the stage
            state: State dictionary to save
            step: Current training step
            metadata: Optional metadata to store
            force: Force save even if max_checkpoints reached

        Returns:
            CheckpointInfo with checkpoint details
        """
        with self._lock:
            stage_dir = self._get_stage_dir(stage_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate checkpoint name
            checkpoint_name = f"checkpoint_{stage_name}_step{step}_{timestamp}"
            checkpoint_path = stage_dir / checkpoint_name

            # Create checkpoint directory
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save state to state.json
            state_file = checkpoint_path / "state.json"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

            # Build metadata
            meta: Dict[str, Any] = {
                "stage_name": stage_name,
                "timestamp": timestamp,
                "step": step,
                "checkpoint_name": checkpoint_name,
            }
            if metadata:
                meta.update(metadata)

            # Save metadata to metadata.json
            metadata_file = checkpoint_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(meta, f, indent=2)

            self.logger.info(f"Saved checkpoint for {stage_name}: {checkpoint_path}")

            # Auto cleanup old checkpoints
            if self.auto_cleanup and self.max_checkpoints > 0:
                self._cleanup_old_checkpoints(stage_name, stage_dir)

            return CheckpointInfo(
                stage_name=stage_name,
                checkpoint_path=str(checkpoint_path),
                timestamp=timestamp,
                step=step,
                metadata=meta,
            )

    def load_checkpoint(
        self,
        stage_name: str,
        checkpoint_path: Optional[str] = None,
        latest: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for a stage.

        Args:
            stage_name: Name of the stage
            checkpoint_path: Specific checkpoint path, or None for latest
            latest: If True and checkpoint_path is None, load latest

        Returns:
            State dictionary, or None if no checkpoint found
        """
        with self._lock:
            stage_dir = self._get_stage_dir(stage_name)

            target_path: Optional[Path] = None

            if checkpoint_path:
                target_path = Path(checkpoint_path)
            elif latest:
                # Get latest checkpoint
                checkpoints = sorted(
                    stage_dir.glob("checkpoint_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not checkpoints:
                    self.logger.warning(f"No checkpoints found for stage: {stage_name}")
                    return None
                target_path = checkpoints[0]

            if not target_path or not target_path.exists():
                self.logger.warning(f"Checkpoint path not found: {target_path}")
                return None

            state_file = target_path / "state.json"
            if not state_file.exists():
                self.logger.warning(f"State file not found: {state_file}")
                return None

            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse checkpoint state: {e}")
                return None

    def list_checkpoints(
        self,
        stage_name: Optional[str] = None,
        sort_by: str = "timestamp",
        ascending: bool = False,
    ) -> List[CheckpointInfo]:
        """
        List all checkpoints, optionally filtered by stage.

        Args:
            stage_name: Optional stage name to filter by
            sort_by: Field to sort by (timestamp, step)
            ascending: Sort in ascending order

        Returns:
            List of CheckpointInfo objects
        """
        checkpoints: List[CheckpointInfo] = []

        if stage_name:
            stage_dir = self._get_stage_dir(stage_name)
            checkpoint_dirs = stage_dir.glob("checkpoint_*")
        else:
            checkpoint_dirs = []
            for stage in self.stage_dirs.values():
                checkpoint_dirs.extend(stage.glob("checkpoint_*"))

        for ckpt_dir in checkpoint_dirs:
            metadata_file = ckpt_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    checkpoints.append(CheckpointInfo.from_dict(metadata))
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Invalid checkpoint metadata: {e}")
                    continue

        # Sort checkpoints
        reverse = not ascending
        if sort_by == "timestamp":
            checkpoints.sort(key=lambda c: c.timestamp, reverse=reverse)
        elif sort_by == "step":
            checkpoints.sort(key=lambda c: c.step, reverse=reverse)

        return checkpoints

    def get_latest_checkpoint(self, stage_name: str) -> Optional[CheckpointInfo]:
        """
        Get the latest checkpoint for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Latest CheckpointInfo, or None if no checkpoints
        """
        checkpoints = self.list_checkpoints(stage_name, sort_by="timestamp")
        return checkpoints[0] if checkpoints else None

    def get_checkpoint_by_step(
        self, stage_name: str, step: int
    ) -> Optional[CheckpointInfo]:
        """
        Get checkpoint for a specific step.

        Args:
            stage_name: Name of the stage
            step: Training step number

        Returns:
            CheckpointInfo for the step, or None if not found
        """
        checkpoints = self.list_checkpoints(stage_name)
        for cp in checkpoints:
            if cp.step == step:
                return cp
        return None

    def delete_checkpoints(
        self, stage_name: str, keep_latest: int = 0, before_step: Optional[int] = None
    ) -> int:
        """
        Delete checkpoints for a stage.

        Args:
            stage_name: Name of the stage
            keep_latest: Number of latest checkpoints to keep
            before_step: Delete checkpoints before this step number

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints(
            stage_name, sort_by="timestamp", ascending=True
        )

        # Filter checkpoints to delete
        to_delete = []
        for i, checkpoint in enumerate(checkpoints):
            if i < keep_latest:
                continue
            if before_step is not None and checkpoint.step < before_step:
                to_delete.append(checkpoint)
            elif before_step is None:
                to_delete.append(checkpoint)

        deleted = 0
        for checkpoint in to_delete:
            try:
                path = Path(checkpoint.checkpoint_path)
                if path.exists():
                    import shutil

                    shutil.rmtree(path)
                    deleted += 1
                    self.logger.info(f"Deleted checkpoint: {path}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to delete checkpoint {checkpoint.checkpoint_path}: {e}"
                )

        return deleted

    def _cleanup_old_checkpoints(self, stage_name: str, stage_dir: Path) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return

        checkpoints = sorted(
            stage_dir.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime
        )

        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            try:
                import shutil

                shutil.rmtree(oldest)
                self.logger.info(f"Cleaned up old checkpoint: {oldest}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup checkpoint {oldest}: {e}")

    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        checkpoint_count = 0
        stage_usage: Dict[str, Dict[str, Any]] = {}

        for stage_name in self.stage_dirs:
            stage_dir = self.stage_dirs[stage_name]
            stage_size = 0
            stage_checkpoints = 0

            for checkpoint_dir in stage_dir.glob("checkpoint_*"):
                if checkpoint_dir.is_dir():
                    for file in checkpoint_dir.rglob("*"):
                        if file.is_file():
                            size = file.stat().st_size
                            stage_size += size
                            total_size += size
                    stage_checkpoints += 1

            stage_usage[stage_name] = {
                "checkpoint_count": stage_checkpoints,
                "size_bytes": stage_size,
                "size_human": self._human_size(stage_size),
            }
            checkpoint_count += stage_checkpoints

        return {
            "total_checkpoints": checkpoint_count,
            "total_size_bytes": total_size,
            "total_size_human": self._human_size(total_size),
            "by_stage": stage_usage,
        }

    @staticmethod
    def _human_size(size_bytes: float) -> str:
        """Convert bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


class PipelineOrchestrator:
    """
    Main orchestrator for executing pipeline stages.

    This class manages the execution of multiple stages, handling:
    - Sequential and parallel stage execution with dependency management
    - Inter-stage data flow and checkpoint management
    - Error handling with stop, continue, and rollback modes
    - Progress tracking and reporting
    - Recovery from checkpoints

    Attributes:
        config: Pipeline configuration
        stages: Dictionary of registered stages keyed by name
        checkpoint_manager: Manager for stage checkpoints
        execution_order: Computed order of stage execution
        logger: Logger instance for orchestrator events
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.stages: Dict[str, Any] = {}
        self.execution_order: List[str] = []

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir, max_checkpoints=5
        )

        # Execution state
        self._stage_status: Dict[str, StageStatus] = {}
        self._stage_outputs: Dict[str, Any] = {}
        self._stage_inputs: Dict[str, Any] = {}
        self._errors: List[PipelineError] = []
        self._start_time: Optional[float] = None
        self._stage_start_times: Dict[str, float] = {}
        self._rollback_stack: List[Tuple[str, CheckpointInfo]] = []
        self._completed_stages: Set[str] = set()

        # Thread lock for thread-safe execution
        self._execution_lock = Lock()

        # Setup logging
        self.logger = self._setup_logger()

        # Compute execution order based on dependencies
        self._compute_execution_order()

        # Initialize stage status
        self._initialize_stage_status()

        self.logger.info(
            f"PipelineOrchestrator initialized with {len(self.stages)} stages"
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the orchestrator."""
        logger = logging.getLogger("pipeline_orchestrator")
        log_level = logging.INFO if self.config.enable_logging else logging.WARNING
        logger.setLevel(log_level)

        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            f"[%(asctime)s] [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        try:
            log_file = Path(self.config.checkpoint_dir) / "pipeline.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning(f"Failed to create log file: {e}")

        return logger

    def _compute_execution_order(self) -> None:
        """
        Compute the topological order of stage execution.

        Uses Kahn's algorithm for topological sorting based on
        stage dependencies to ensure correct execution order.
        """
        # Build dependency graph (reverse: dep -> dependent)
        graph: Dict[str, List[str]] = {stage: [] for stage in self.config.stages}
        in_degree: Dict[str, int] = {stage: 0 for stage in self.config.stages}

        for stage_name, deps in self.config.dependencies.items():
            if stage_name in graph:
                for dep in deps:
                    if dep in graph:
                        graph[dep].append(stage_name)
                        in_degree[stage_name] = in_degree.get(stage_name, 0) + 1

        # Kahn's algorithm for topological sort
        queue: List[str] = [
            stage for stage in self.config.stages if in_degree.get(stage, 0) == 0
        ]
        self.execution_order = []

        while queue:
            current = queue.pop(0)
            self.execution_order.append(current)

            for neighbor in graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(self.execution_order) != len(self.config.stages):
            remaining = set(self.config.stages) - set(self.execution_order)
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        self.logger.info(f"Computed execution order: {self.execution_order}")

    def _initialize_stage_status(self) -> None:
        """Initialize status for all stages."""
        for stage_name in self.config.stages:
            self._stage_status[stage_name] = StageStatus.PENDING
        self.logger.debug(f"Initialized status for {len(self.config.stages)} stages")

    def add_stage(
        self,
        stage_name: str,
        stage_instance: Any,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Add a stage to the pipeline.

        Args:
            stage_name: Unique identifier for the stage
            stage_instance: The stage implementation instance
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
            config: Additional stage-specific configuration
            timeout: Stage-specific timeout override

        Raises:
            ValueError: If stage name already exists or dependency invalid
        """
        if stage_name in self.stages:
            raise ValueError(f"Stage '{stage_name}' already added to pipeline")

        # Validate dependencies exist
        deps = self.config.dependencies.get(stage_name, [])
        for dep in deps:
            if dep not in self.config.stages:
                raise ValueError(
                    f"Dependency '{dep}' not found for stage '{stage_name}'"
                )

        # Create stage wrapper
        stage_wrapper = {
            "name": stage_name,
            "instance": stage_instance,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "config": config or {},
            "timeout": timeout or self.config.timeout_per_stage,
        }

        self.stages[stage_name] = stage_wrapper
        self._stage_status[stage_name] = StageStatus.PENDING

        self.logger.info(f"Added stage: {stage_name} (dependencies: {deps})")

    def remove_stage(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Remove a stage from the pipeline.

        Args:
            stage_name: Name of the stage to remove

        Returns:
            The removed stage data, or None if not found
        """
        if stage_name not in self.stages:
            return None

        stage = self.stages.pop(stage_name)
        self._stage_status.pop(stage_name, None)
        self._stage_outputs.pop(stage_name, None)
        self._stage_inputs.pop(stage_name, None)

        self.logger.info(f"Removed stage: {stage_name}")
        return stage

    def get_stage(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a stage by name.

        Args:
            stage_name: Name of the stage

        Returns:
            The stage data, or None if not found
        """
        return self.stages.get(stage_name)

    def validate_dag(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the pipeline DAG (Directed Acyclic Graph).

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self._compute_execution_order()
            return True, None
        except ValueError as e:
            self.logger.error(f"DAG validation failed: {e}")
            return False, str(e)

    def _check_dependencies_satisfied(self, stage_name: str) -> bool:
        """Check if all dependencies for a stage are completed."""
        deps = self.config.dependencies.get(stage_name, [])

        for dep in deps:
            if self._stage_status.get(dep) == StageStatus.COMPLETED:
                continue
            # Check if we have output from dependency
            if dep in self._stage_outputs:
                continue
            return False

        return True

    def _get_stage_input(self, stage_name: str) -> Dict[str, Any]:
        """Get input data for a stage from dependencies."""
        deps = self.config.dependencies.get(stage_name, [])

        if not deps:
            # First stage - use global input
            return self._initial_input.copy()

        # Merge outputs from dependencies
        combined_input: Dict[str, Any] = {}

        for dep in deps:
            if dep in self._stage_outputs:
                # Mark the source of each piece of data
                combined_input[f"from_{dep}"] = self._stage_outputs[dep]

        return combined_input

    def _execute_stage(
        self, stage_name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single stage.

        Args:
            stage_name: Name of the stage to execute
            input_data: Input data for the stage

        Returns:
            Output data from the stage

        Raises:
            Exception: If stage execution fails
        """
        stage = self.stages[stage_name]
        stage_instance = stage["instance"]

        self.logger.info(f"Executing stage: {stage_name}")
        self._stage_status[stage_name] = StageStatus.RUNNING
        self._stage_start_times[stage_name] = time.time()

        # Save initial checkpoint
        checkpoint = self.checkpoint_manager.save_checkpoint(
            stage_name=stage_name,
            state={"input": input_data, "status": "running"},
            step=0,
            metadata={"stage": stage_name, "action": "start"},
        )

        # Track for potential rollback
        self._rollback_stack.append((stage_name, checkpoint))

        try:
            # Determine execution method
            if hasattr(stage_instance, "run"):
                result = stage_instance.run(input_data)
            elif hasattr(stage_instance, "train"):
                # For stages with train method
                if hasattr(stage_instance, "prepare"):
                    stage_instance.prepare(input_data)
                result = stage_instance.train()
            elif hasattr(stage_instance, "execute"):
                result = stage_instance.execute(input_data)
            elif hasattr(stage_instance, "__call__"):
                result = stage_instance(input_data)
            else:
                raise NotImplementedError(
                    f"Stage {stage_name} must have one of: run(), train(), execute(), or __call__()"
                )

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"output": result}

            # Save final checkpoint
            final_checkpoint = self.checkpoint_manager.save_checkpoint(
                stage_name=stage_name,
                state={
                    "input": input_data,
                    "output": result,
                    "status": "completed",
                },
                step=getattr(stage_instance, "current_step", 1),
                metadata={"stage": stage_name, "action": "complete", "success": True},
            )

            self._stage_status[stage_name] = StageStatus.COMPLETED
            self._completed_stages.add(stage_name)

            elapsed = time.time() - self._stage_start_times[stage_name]
            self.logger.info(f"Stage completed: {stage_name} ({elapsed:.2f}s)")

            return result

        except Exception as e:
            self._stage_status[stage_name] = StageStatus.FAILED

            # Save error checkpoint
            error_checkpoint = self.checkpoint_manager.save_checkpoint(
                stage_name=stage_name,
                state={
                    "input": input_data,
                    "status": "failed",
                    "error": str(e),
                },
                step=getattr(stage_instance, "current_step", 0),
                metadata={
                    "stage": stage_name,
                    "action": "error",
                    "success": False,
                    "error": str(e),
                },
            )

            # Create error record
            error = PipelineError.from_exception(stage_name, e, recoverable=True)
            self._errors.append(error)

            self.logger.error(f"Stage failed: {stage_name} - {e}")
            raise

    def _execute_parallel_stages(self, stage_names: List[str]) -> Dict[str, Any]:
        """
        Execute multiple stages in parallel.

        Args:
            stage_names: List of stage names to execute in parallel

        Returns:
            Dictionary of outputs keyed by stage name
        """
        if not stage_names:
            return {}

        self.logger.info(f"Executing stages in parallel: {stage_names}")

        outputs: Dict[str, Any] = {}
        errors: List[PipelineError] = []

        def execute_single_stage(
            stage_name: str,
        ) -> Tuple[str, Any, Optional[Any]]:
            """Execute a single stage and return result or error."""
            try:
                input_data = self._get_stage_input(stage_name)
                output = self._execute_stage(stage_name, input_data)
                return stage_name, output, None
            except Exception as e:
                error = PipelineError.from_exception(stage_name, e)
                return stage_name, None, error

        # Execute stages with thread pool
        max_workers = min(self.config.max_workers, len(stage_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(execute_single_stage, stage_name): stage_name
                for stage_name in stage_names
            }

            for future in as_completed(futures):
                stage_name, output, error = future.result()

                if error:
                    errors.append(error)
                    self.logger.error(f"Parallel stage failed: {stage_name} - {error}")
                else:
                    outputs[stage_name] = output

        # Store outputs
        for stage_name, output in outputs.items():
            self._stage_outputs[stage_name] = output

        # Store errors
        self._errors.extend(errors)

        return outputs

    def _should_continue_on_failure(self) -> bool:
        """Check if execution should continue after a failure."""
        return FailureMode(self.config.failure_mode) == FailureMode.CONTINUE

    def _should_rollback(self) -> bool:
        """Check if execution should rollback on failure."""
        return FailureMode(self.config.failure_mode) == FailureMode.ROLLBACK

    def _rollback(self, stages_to_rollback: int = 1) -> None:
        """
        Perform rollback to previous checkpoint.

        Args:
            stages_to_rollback: Number of stages to rollback
        """
        self.logger.warning(f"Rolling back {stages_to_rollback} stage(s)")

        # Pop stages from rollback stack
        rolled_back = 0
        while rolled_back < stages_to_rollback and self._rollback_stack:
            stage_name, checkpoint = self._rollback_stack.pop()

            # Reset stage status
            self._stage_status[stage_name] = StageStatus.ROLLBACK
            self._completed_stages.discard(stage_name)

            # Remove outputs
            self._stage_outputs.pop(stage_name, None)
            self._stage_inputs.pop(stage_name, None)

            self.logger.info(f"Rolled back stage: {stage_name}")
            rolled_back += 1

        # Update status of remaining completed stages
        for stage_name in self._completed_stages:
            self._stage_status[stage_name] = StageStatus.COMPLETED

        self.logger.warning(f"Rollback complete: {rolled_back} stage(s) rolled back")

    def _rollback_to_checkpoint(self, checkpoint: CheckpointInfo) -> bool:
        """
        Rollback to a specific checkpoint.

        Args:
            checkpoint: CheckpointInfo to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        self.logger.info(f"Rolling back to checkpoint: {checkpoint.checkpoint_path}")

        # Load checkpoint state
        state = self.checkpoint_manager.load_checkpoint(
            checkpoint.stage_name, checkpoint.checkpoint_path
        )

        if not state:
            self.logger.error(
                f"Failed to load checkpoint: {checkpoint.checkpoint_path}"
            )
            return False

        # Remove all stages after the rollback checkpoint
        try:
            stage_index = self.execution_order.index(checkpoint.stage_name)
            stages_to_remove = self.execution_order[stage_index + 1 :]

            for stage_name in stages_to_remove:
                if stage_name in self._rollback_stack:
                    # Find and remove from rollback stack
                    for i, (name, cp) in enumerate(self._rollback_stack):
                        if name == stage_name:
                            self._rollback_stack.pop(i)
                            break

                self._stage_status[stage_name] = StageStatus.ROLLBACK
                self._completed_stages.discard(stage_name)
                self._stage_outputs.pop(stage_name, None)
                self._stage_inputs.pop(stage_name, None)

                self.logger.info(f"Reset stage during rollback: {stage_name}")

            return True

        except ValueError:
            self.logger.error(
                f"Stage not found in execution order: {checkpoint.stage_name}"
            )
            return False

    def get_progress(self) -> PipelineProgress:
        """
        Get current progress of pipeline execution.

        Returns:
            PipelineProgress with current status
        """
        elapsed = 0.0
        if self._start_time:
            elapsed = time.time() - self._start_time

        completed = list(self._completed_stages)
        failed = [
            s
            for s, status in self._stage_status.items()
            if status == StageStatus.FAILED
        ]
        skipped = [
            s
            for s, status in self._stage_status.items()
            if status == StageStatus.SKIPPED
        ]

        current = None
        current_start_time = None
        for stage_name in self.execution_order:
            status = self._stage_status.get(stage_name)
            if status == StageStatus.RUNNING:
                current = stage_name
                current_start_time = self._stage_start_times.get(stage_name)
                break

        last_checkpoint = None
        if self._rollback_stack:
            last_checkpoint = self._rollback_stack[-1][1]

        # Calculate current step
        current_step = len(completed) + 1 if current else len(completed)

        return PipelineProgress(
            current_stage=current,
            total_stages=len(self.stages),
            completed_stages=completed,
            failed_stages=failed,
            skipped_stages=skipped,
            current_step=current_step,
            total_steps=len(self.stages),
            start_time=datetime.fromtimestamp(self._start_time).isoformat()
            if self._start_time
            else None,
            elapsed_seconds=elapsed,
            last_checkpoint=last_checkpoint,
            stage_start_time=datetime.fromtimestamp(current_start_time).isoformat()
            if current_start_time
            else None,
        )

    def execute_pipeline(
        self,
        input_data: Dict[str, Any],
        dry_run: bool = False,
        rollback_on_failure: Optional[bool] = None,
    ) -> PipelineResult:
        """
        Execute all stages in the pipeline.

        Args:
            input_data: Initial input data for the pipeline
            dry_run: If True, simulate execution without running stages
            rollback_on_failure: Override for config rollback setting

        Returns:
            PipelineResult with execution results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting pipeline execution")
        self.logger.info("=" * 60)

        self._start_time = time.time()
        self._initial_input = input_data.copy()
        self._errors = []
        self._stage_outputs = {}
        self._stage_inputs = {}
        self._rollback_stack = []
        self._completed_stages = set()

        # Determine rollback behavior
        do_rollback = (
            rollback_on_failure
            if rollback_on_failure is not None
            else self._should_rollback()
        )

        # Check for resume
        resume_from = self.config.resume_from
        if resume_from and resume_from in self.stages:
            self.logger.info(f"Resuming from stage: {resume_from}")
            # Load outputs for resumed stages
            for stage_name in self.execution_order:
                if stage_name == resume_from:
                    break
                if stage_name in self.stages:
                    checkpoint = self.checkpoint_manager.get_latest_checkpoint(
                        stage_name
                    )
                    if checkpoint:
                        state = self.checkpoint_manager.load_checkpoint(stage_name)
                        if state and "output" in state:
                            self._stage_outputs[stage_name] = state["output"]
                            self._stage_status[stage_name] = StageStatus.COMPLETED
                            self._completed_stages.add(stage_name)
                            # Add to rollback stack
                            self._rollback_stack.append((stage_name, checkpoint))

        # Execute stages in order
        last_executed_stage = None

        for stage_name in self.execution_order:
            # Skip stages before resume point
            if resume_from and stage_name != resume_from:
                if stage_name in self.stages:
                    self._stage_status[stage_name] = StageStatus.SKIPPED
                continue

            if stage_name not in self.stages:
                self.logger.warning(f"Stage {stage_name} not registered, skipping")
                continue

            # Check if dependencies are satisfied
            if not self._check_dependencies_satisfied(stage_name):
                deps = self.config.dependencies.get(stage_name, [])
                self.logger.warning(
                    f"Skipping {stage_name}: dependencies not satisfied: {deps}"
                )
                self._stage_status[stage_name] = StageStatus.SKIPPED
                continue

            # Check if this stage can run in parallel
            if (
                stage_name in self.config.parallel_stages
                and self.config.parallel_stages.index(stage_name)
                < len(self.config.parallel_stages) - 1
            ):
                # Find all parallel stages that can run
                parallel_group = []
                remaining_parallel = self.config.parallel_stages[
                    self.config.parallel_stages.index(stage_name) :
                ]

                for ps in remaining_parallel:
                    if ps in self.stages and self._check_dependencies_satisfied(ps):
                        parallel_group.append(ps)

                if len(parallel_group) > 1:
                    # Execute in parallel
                    self._execute_parallel_stages(parallel_group)
                    for ps in parallel_group:
                        if ps in self._stage_outputs:
                            last_executed_stage = ps
                    continue

            # Execute single stage
            if dry_run:
                self.logger.info(f"[DRY-RUN] Would execute stage: {stage_name}")
                self._stage_outputs[stage_name] = {"dry_run": True}
                self._stage_status[stage_name] = StageStatus.COMPLETED
                self._completed_stages.add(stage_name)
                last_executed_stage = stage_name
                continue

            try:
                # Get input for this stage
                stage_input = self._get_stage_input(stage_name)
                self._stage_inputs[stage_name] = stage_input

                # Execute stage
                output = self._execute_stage(stage_name, stage_input)
                self._stage_outputs[stage_name] = output
                last_executed_stage = stage_name

            except Exception as e:
                self.logger.error(f"Stage {stage_name} failed: {e}")

                if do_rollback and self.config.enable_rollback:
                    self.logger.warning("Triggering rollback...")
                    self._rollback(self.config.rollback_stages)
                    rollback_result = PipelineResult(
                        success=False,
                        outputs=self._stage_outputs,
                        checkpoints=self.checkpoint_manager.list_checkpoints(),
                        duration=time.time() - self._start_time,
                        errors=self._errors,
                        progress=self.get_progress(),
                        final_stage=last_executed_stage,
                        rollback_performed=True,
                    )
                    return rollback_result

                elif self._should_continue_on_failure():
                    self.logger.info("Continuing to next stage...")
                    continue

                # Stop on failure
                self.logger.error("Pipeline execution stopped due to failure")
                break

        # Calculate results
        duration = time.time() - self._start_time
        success = (
            len(self._completed_stages) == len(self.stages) and len(self._errors) == 0
        )

        # Get all checkpoints
        all_checkpoints = self.checkpoint_manager.list_checkpoints()

        # Create result
        result = PipelineResult(
            success=success,
            outputs=self._stage_outputs,
            checkpoints=all_checkpoints,
            duration=duration,
            errors=self._errors,
            progress=self.get_progress(),
            final_stage=last_executed_stage,
            rollback_performed=False,
        )

        # Save result
        result_path = Path(self.config.checkpoint_dir) / "pipeline_result.json"
        result.save(str(result_path))

        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline execution {'completed' if success else 'failed'}")
        self.logger.info(f"Duration: {duration:.2f}s")
        self.logger.info(
            f"Completed stages: {len(self._completed_stages)}/{len(self.stages)}"
        )
        self.logger.info(f"Errors: {len(self._errors)}")
        self.logger.info("=" * 60)

        return result

    def reset(self) -> None:
        """Reset the orchestrator state for a new execution."""
        self._stage_status = {s: StageStatus.PENDING for s in self.stages}
        self._stage_outputs = {}
        self._stage_inputs = {}
        self._errors = []
        self._start_time = None
        self._stage_start_times = {}
        self._rollback_stack = []
        self._completed_stages = set()
        self.logger.info("Pipeline orchestrator reset")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the orchestrator."""
        return {
            "config": self.config.to_dict(),
            "registered_stages": list(self.stages.keys()),
            "execution_order": self.execution_order,
            "stage_status": {
                s: status.value for s, status in self._stage_status.items()
            },
            "completed_count": len(self._completed_stages),
            "error_count": len(self._errors),
            "rollback_stack_depth": len(self._rollback_stack),
            "checkpoint_storage": self.checkpoint_manager.get_storage_usage(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineOrchestrator("
            f"stages={list(self.stages.keys())}, "
            f"execution_order={self.execution_order}, "
            f"failure_mode={self.config.failure_mode})"
        )

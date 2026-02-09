"""
QAD (Quantization-Aware Distillation) Loss Module for Nexus SLI

Implements knowledge distillation loss between FP32 teacher and NVFP4 student models.
Key features:
- KL divergence loss with temperature scaling
- Label smoothing support
- Gradient scaling for stability
- Per-layer distillation tracking

Author: Nexus Team
"""

import logging
import warnings
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .exceptions import SLIError

logger = logging.getLogger(__name__)


class QADLossType(Enum):
    """Types of QAD distillation losses."""
    KL_DIVERGENCE = "kl_divergence"
    MSE = "mse"
    COSINE = "cosine"
    COMBINED = "combined"


@dataclass
class QADLossConfig:
    """Configuration for QAD distillation loss.
    
    Attributes:
        temperature: Temperature for softening distributions (1.0-2.0)
        alpha: Weight for distillation loss vs hard target loss (0.0-1.0)
        beta: Weight for hidden state matching loss
        label_smoothing: Label smoothing factor (0.0-0.1)
        loss_type: Type of distillation loss
        use_attention_matching: Enable attention output matching
        use_hidden_matching: Enable hidden state matching
        gradient_clip: Gradient clipping threshold
        adaptive_temperature: Enable adaptive temperature scaling
        min_temperature: Minimum temperature for adaptive scaling
        max_temperature: Maximum temperature for adaptive scaling
    """
    temperature: float = 1.5
    alpha: float = 0.7
    beta: float = 0.3
    label_smoothing: float = 0.1
    loss_type: QADLossType = QADLossType.KL_DIVERGENCE
    use_attention_matching: bool = True
    use_hidden_matching: bool = True
    gradient_clip: float = 1.0
    adaptive_temperature: bool = False
    min_temperature: float = 1.0
    max_temperature: float = 2.0
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        
        if not 0.0 <= self.label_smoothing <= 0.5:
            raise ValueError(f"label_smoothing must be in [0, 0.5], got {self.label_smoothing}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        if self.adaptive_temperature:
            if not (self.min_temperature < self.max_temperature):
                raise ValueError("min_temperature must be less than max_temperature")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'label_smoothing': self.label_smoothing,
            'loss_type': self.loss_type.value,
            'use_attention_matching': self.use_attention_matching,
            'use_hidden_matching': self.use_hidden_matching,
            'gradient_clip': self.gradient_clip,
            'adaptive_temperature': self.adaptive_temperature,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QADLossConfig':
        """Create config from dictionary."""
        return cls(
            temperature=data.get('temperature', 1.5),
            alpha=data.get('alpha', 0.7),
            beta=data.get('beta', 0.3),
            label_smoothing=data.get('label_smoothing', 0.1),
            loss_type=QADLossType(data.get('loss_type', 'kl_divergence')),
            use_attention_matching=data.get('use_attention_matching', True),
            use_hidden_matching=data.get('use_hidden_matching', True),
            gradient_clip=data.get('gradient_clip', 1.0),
            adaptive_temperature=data.get('adaptive_temperature', False),
            min_temperature=data.get('min_temperature', 1.0),
            max_temperature=data.get('max_temperature', 2.0),
        )


class QADLossError(SLIError):
    """Raised when QAD loss computation fails."""
    
    def __init__(self, message: str, layer_idx: Optional[int] = None):
        self.layer_idx = layer_idx
        if layer_idx is not None:
            msg = f"QAD loss error at layer {layer_idx}: {message}"
        else:
            msg = f"QAD loss error: {message}"
        super().__init__(msg)


@dataclass
class QADLossStats:
    """Statistics for QAD distillation."""
    total_loss: float = 0.0
    distillation_loss: float = 0.0
    hard_target_loss: float = 0.0
    hidden_matching_loss: float = 0.0
    attention_matching_loss: float = 0.0
    temperature: float = 1.0
    step: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary."""
        return {
            'total_loss': self.total_loss,
            'distillation_loss': self.distillation_loss,
            'hard_target_loss': self.hard_target_loss,
            'hidden_matching_loss': self.hidden_matching_loss,
            'attention_matching_loss': self.attention_matching_loss,
            'temperature': self.temperature,
            'step': self.step,
        }


class QADDistillationLoss(nn.Module):
    """QAD (Quantization-Aware Distillation) Loss Module.
    
    Implements knowledge distillation from a full-precision (FP32) teacher
    model to a quantized (NVFP4) student model. The distillation process
    helps the quantized model maintain accuracy despite lower precision.
    
    The loss combines:
    1. Distillation loss: KL divergence between teacher and student outputs
    2. Hard target loss: Cross-entropy with ground truth labels
    3. Hidden matching loss: MSE between intermediate representations
    4. Attention matching loss: MSE between attention outputs
    
    Example:
        >>> teacher = load_fp32_model()
        >>> student = load_nvfp4_model()
        >>> qad_loss = QADDistillationLoss(QADLossConfig(temperature=1.5))
        >>> 
        >>> # Forward pass
        >>> teacher_logits = teacher(inputs)
        >>> student_logits = student(inputs)
        >>> 
        >>> # Compute loss
        >>> loss = qad_loss(
        ...     student_logits=student_logits,
        ...     teacher_logits=teacher_logits,
        ...     labels=labels,
        ...     hidden_student=student_hidden,
        ...     hidden_teacher=teacher_hidden
        ... )
    """
    
    def __init__(self, config: Optional[QADLossConfig] = None):
        """Initialize QAD distillation loss.
        
        Args:
            config: QAD loss configuration
        """
        super().__init__()
        self.config = config or QADLossConfig()
        
        # Statistics tracking
        self._stats = QADLossStats()
        self._history: List[QADLossStats] = []
        self._max_history = 1000
        self._lock = threading.RLock()
        
        # Adaptive temperature state
        self._current_temperature = self.config.temperature
        self._loss_history: List[float] = []
        
        logger.info(f"QADDistillationLoss initialized (temperature: {self.config.temperature})")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hidden_student: Optional[torch.Tensor] = None,
        hidden_teacher: Optional[torch.Tensor] = None,
        attention_student: Optional[torch.Tensor] = None,
        attention_teacher: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute QAD distillation loss.
        
        Args:
            student_logits: Logits from quantized student model
            teacher_logits: Logits from FP32 teacher model
            labels: Ground truth labels (optional)
            hidden_student: Student hidden states (optional)
            hidden_teacher: Teacher hidden states (optional)
            attention_student: Student attention outputs (optional)
            attention_teacher: Teacher attention outputs (optional)
            mask: Attention mask (optional)
            
        Returns:
            Combined distillation loss
        """
        # Update temperature if adaptive
        if self.config.adaptive_temperature:
            self._update_adaptive_temperature()
        
        temperature = self._current_temperature
        
        # Compute distillation loss
        distill_loss = self._compute_distillation_loss(
            student_logits, teacher_logits, temperature
        )
        
        # Compute hard target loss if labels provided
        hard_loss = 0.0
        if labels is not None:
            hard_loss = self._compute_hard_target_loss(student_logits, labels)
        
        # Compute hidden matching loss
        hidden_loss = 0.0
        if (self.config.use_hidden_matching and 
            hidden_student is not None and 
            hidden_teacher is not None):
            hidden_loss = self._compute_hidden_matching_loss(
                hidden_student, hidden_teacher, mask
            )
        
        # Compute attention matching loss
        attention_loss = 0.0
        if (self.config.use_attention_matching and
            attention_student is not None and
            attention_teacher is not None):
            attention_loss = self._compute_attention_matching_loss(
                attention_student, attention_teacher, mask
            )
        
        # Combine losses
        total_loss = (
            self.config.alpha * distill_loss +
            (1 - self.config.alpha) * hard_loss +
            self.config.beta * (hidden_loss + attention_loss)
        )
        
        # Update statistics
        with self._lock:
            self._stats.total_loss = total_loss.item()
            self._stats.distillation_loss = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            self._stats.hard_target_loss = hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss
            self._stats.hidden_matching_loss = hidden_loss.item() if isinstance(hidden_loss, torch.Tensor) else hidden_loss
            self._stats.attention_matching_loss = attention_loss.item() if isinstance(attention_loss, torch.Tensor) else attention_loss
            self._stats.temperature = temperature
            self._stats.step += 1
            
            self._history.append(self._stats.to_dict())
            if len(self._history) > self._max_history:
                self._history.pop(0)
            
            self._loss_history.append(total_loss.item())
            if len(self._loss_history) > 100:
                self._loss_history.pop(0)
        
        return total_loss
    
    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute distillation loss between student and teacher.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            temperature: Softmax temperature
            
        Returns:
            Distillation loss
        """
        if self.config.loss_type == QADLossType.KL_DIVERGENCE:
            return self._kl_divergence_loss(student_logits, teacher_logits, temperature)
        elif self.config.loss_type == QADLossType.MSE:
            return self._mse_loss(student_logits, teacher_logits, temperature)
        elif self.config.loss_type == QADLossType.COSINE:
            return self._cosine_loss(student_logits, teacher_logits)
        elif self.config.loss_type == QADLossType.COMBINED:
            kl_loss = self._kl_divergence_loss(student_logits, teacher_logits, temperature)
            mse_loss = self._mse_loss(student_logits, teacher_logits, temperature)
            return 0.5 * kl_loss + 0.5 * mse_loss
        else:
            raise QADLossError(f"Unknown loss type: {self.config.loss_type}")
    
    def _kl_divergence_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute KL divergence loss.
        
        Uses temperature scaling to soften the probability distributions.
        Higher temperature makes distributions more uniform, transferring
        more "dark knowledge" from teacher to student.
        """
        # Apply temperature scaling
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        kl_div = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        )
        
        # Scale by temperature squared (as per Hinton et al.)
        return kl_div * (temperature ** 2)
    
    def _mse_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute MSE loss between softened outputs."""
        student_soft = F.softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        return F.mse_loss(student_soft, teacher_soft)
    
    def _cosine_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity loss."""
        student_norm = F.normalize(student_logits, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=-1)
        return 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=-1).mean()
    
    def _compute_hard_target_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute hard target cross-entropy loss with label smoothing.
        
        Args:
            student_logits: Student model logits
            labels: Ground truth labels
            
        Returns:
            Cross-entropy loss
        """
        if self.config.label_smoothing > 0:
            # Label smoothing cross-entropy
            num_classes = student_logits.size(-1)
            log_probs = F.log_softmax(student_logits, dim=-1)
            
            # Create smoothed labels
            with torch.no_grad():
                smoothed_labels = torch.zeros_like(log_probs)
                smoothed_labels.fill_(self.config.label_smoothing / (num_classes - 1))
                smoothed_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - self.config.label_smoothing)
            
            loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(student_logits, labels)
        
        return loss
    
    def _compute_hidden_matching_loss(
        self,
        hidden_student: torch.Tensor,
        hidden_teacher: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute hidden state matching loss.
        
        Matches intermediate layer representations to transfer
        structural knowledge from teacher to student.
        """
        # Ensure same shape
        if hidden_student.shape != hidden_teacher.shape:
            # Project if dimensions differ
            if hidden_student.size(-1) != hidden_teacher.size(-1):
                projection = nn.Linear(
                    hidden_student.size(-1),
                    hidden_teacher.size(-1),
                    bias=False
                ).to(hidden_student.device)
                hidden_student = projection(hidden_student)
        
        # Normalize
        student_norm = F.normalize(hidden_student, p=2, dim=-1)
        teacher_norm = F.normalize(hidden_teacher, p=2, dim=-1)
        
        # Compute MSE
        if mask is not None:
            # Apply mask
            mask_expanded = mask.unsqueeze(-1).expand_as(student_norm)
            diff = (student_norm - teacher_norm) * mask_expanded
            loss = (diff ** 2).sum() / mask_expanded.sum()
        else:
            loss = F.mse_loss(student_norm, teacher_norm)
        
        return loss
    
    def _compute_attention_matching_loss(
        self,
        attention_student: torch.Tensor,
        attention_teacher: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention output matching loss.
        
        Matches attention patterns to transfer attention knowledge.
        """
        # Normalize attention outputs
        student_norm = F.normalize(attention_student, p=2, dim=-1)
        teacher_norm = F.normalize(attention_teacher, p=2, dim=-1)
        
        if mask is not None:
            # Apply mask
            mask_expanded = mask.unsqueeze(-1).expand_as(student_norm)
            diff = (student_norm - teacher_norm) * mask_expanded
            loss = (diff ** 2).sum() / mask_expanded.sum()
        else:
            loss = F.mse_loss(student_norm, teacher_norm)
        
        return loss
    
    def _update_adaptive_temperature(self):
        """Update temperature based on recent loss trend."""
        if len(self._loss_history) < 10:
            return
        
        recent_loss = np.mean(self._loss_history[-10:])
        older_loss = np.mean(self._loss_history[-100:-10]) if len(self._loss_history) >= 100 else self._loss_history[0]
        
        # If loss is decreasing, increase temperature for softer targets
        # If loss is increasing, decrease temperature for harder targets
        loss_trend = recent_loss - older_loss
        
        if loss_trend < 0:  # Loss decreasing
            self._current_temperature = min(
                self._current_temperature * 1.01,
                self.config.max_temperature
            )
        else:  # Loss increasing
            self._current_temperature = max(
                self._current_temperature * 0.99,
                self.config.min_temperature
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current loss statistics."""
        with self._lock:
            return self._stats.to_dict()
    
    def get_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get loss history.
        
        Args:
            n: Number of recent entries to return (None for all)
            
        Returns:
            List of loss statistics
        """
        with self._lock:
            if n is None:
                return self._history.copy()
            return self._history[-n:]
    
    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self._stats = QADLossStats()
            self._history.clear()
            self._loss_history.clear()
    
    def set_temperature(self, temperature: float):
        """Set temperature manually (disables adaptive temperature).
        
        Args:
            temperature: New temperature value
        """
        self._current_temperature = temperature
        self.config.adaptive_temperature = False
        logger.info(f"Temperature set to {temperature}, adaptive mode disabled")


class PerLayerQADLoss(nn.Module):
    """Per-layer QAD distillation for progressive training.
    
    Computes distillation loss at each layer rather than just at the output.
    This enables layer-by-layer distillation for progressive quantization.
    """
    
    def __init__(
        self,
        config: Optional[QADLossConfig] = None,
        num_layers: int = 1,
        layer_weights: Optional[List[float]] = None
    ):
        """Initialize per-layer QAD loss.
        
        Args:
            config: QAD loss configuration
            num_layers: Number of layers to distill
            layer_weights: Weight for each layer's distillation loss
        """
        super().__init__()
        self.config = config or QADLossConfig()
        self.num_layers = num_layers
        
        if layer_weights is None:
            # Uniform weights by default
            self.layer_weights = [1.0 / num_layers] * num_layers
        else:
            assert len(layer_weights) == num_layers
            self.layer_weights = layer_weights
        
        self.base_loss = QADDistillationLoss(self.config)
    
    def forward(
        self,
        layer_outputs_student: List[torch.Tensor],
        layer_outputs_teacher: List[torch.Tensor],
        final_logits_student: torch.Tensor,
        final_logits_teacher: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute per-layer distillation loss.
        
        Args:
            layer_outputs_student: List of student layer outputs
            layer_outputs_teacher: List of teacher layer outputs
            final_logits_student: Final student logits
            final_logits_teacher: Final teacher logits
            labels: Ground truth labels
            
        Returns:
            Combined per-layer distillation loss
        """
        total_loss = 0.0
        
        # Compute loss for each layer
        for i, (student_out, teacher_out, weight) in enumerate(
            zip(layer_outputs_student, layer_outputs_teacher, self.layer_weights)
        ):
            layer_loss = self.base_loss._compute_hidden_matching_loss(
                student_out, teacher_out
            )
            total_loss += weight * layer_loss
        
        # Add final output distillation
        final_loss = self.base_loss(
            final_logits_student,
            final_logits_teacher,
            labels=labels
        )
        
        return total_loss + final_loss


# Convenience functions
def get_qad_loss_config(
    temperature: float = 1.5,
    alpha: float = 0.7,
    label_smoothing: float = 0.1,
    adaptive: bool = False
) -> QADLossConfig:
    """Get QAD loss configuration with common presets.
    
    Args:
        temperature: Temperature for softening (1.0-2.0)
        alpha: Weight for distillation vs hard target (0.0-1.0)
        label_smoothing: Label smoothing factor (0.0-0.1)
        adaptive: Enable adaptive temperature
        
    Returns:
        QADLossConfig instance
    """
    return QADLossConfig(
        temperature=temperature,
        alpha=alpha,
        label_smoothing=label_smoothing,
        adaptive_temperature=adaptive
    )


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.5
) -> torch.Tensor:
    """Convenience function for simple distillation loss.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature for softening
        
    Returns:
        KL divergence loss
    """
    config = QADLossConfig(temperature=temperature)
    loss_fn = QADDistillationLoss(config)
    return loss_fn._compute_distillation_loss(
        student_logits, teacher_logits, temperature
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing QAD Distillation Loss")
    print("=" * 60)
    
    # Create config
    config = QADLossConfig(
        temperature=1.5,
        alpha=0.7,
        label_smoothing=0.1
    )
    print(f"Config: {config.to_dict()}")
    
    # Create loss function
    qad_loss = QADDistillationLoss(config)
    
    # Create dummy data
    batch_size = 4
    num_classes = 1000
    
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    hidden_student = torch.randn(batch_size, 128, 768)
    hidden_teacher = torch.randn(batch_size, 128, 768)
    
    # Compute loss
    loss = qad_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        hidden_student=hidden_student,
        hidden_teacher=hidden_teacher
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Stats: {qad_loss.get_stats()}")
    
    # Test adaptive temperature
    print("\nTesting adaptive temperature...")
    config_adaptive = QADLossConfig(
        temperature=1.5,
        adaptive_temperature=True,
        min_temperature=1.0,
        max_temperature=2.0
    )
    qad_adaptive = QADDistillationLoss(config_adaptive)
    
    for _ in range(20):
        student_logits = torch.randn(batch_size, num_classes)
        teacher_logits = torch.randn(batch_size, num_classes)
        loss = qad_adaptive(student_logits, teacher_logits, labels=labels)
    
    print(f"Final temperature: {qad_adaptive.get_stats()['temperature']:.4f}")
    
    print("\n" + "=" * 60)

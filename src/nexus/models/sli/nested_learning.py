"""
Nested Learning Implementation for SLI

Implements nested, hierarchical learning strategies for efficient model adaptation:
- Progressive layer unfreezing
- Nested dropout schedules
- Hierarchical knowledge distillation
- Adaptive learning rate by layer depth

Based on research in "Nested Learning: Progressively Adapting Deep Models"

Author: Nexus Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class NestedStrategy(Enum):
    """Nested learning strategies."""
    PROGRESSIVE = "progressive"      # Progressive layer unfreezing
    HIERARCHICAL = "hierarchical"    # Hierarchical distillation
    ADAPTIVE = "adaptive"            # Adaptive depth-based LR
    COMBINED = "combined"            # Combined approach


@dataclass
class NestedLearningConfig:
    """Configuration for nested learning."""
    strategy: NestedStrategy = NestedStrategy.PROGRESSIVE
    
    # Progressive unfreezing
    unfreeze_schedule: List[int] = field(default_factory=lambda: [0, 1000, 2000, 3000])
    layers_per_step: int = 4
    
    # Hierarchical distillation
    num_hierarchy_levels: int = 3
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    
    # Adaptive learning rates
    base_lr: float = 1e-5
    lr_decay_factor: float = 0.9
    min_lr: float = 1e-7
    
    # Nested dropout
    dropout_schedule: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.1])
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_steps: int = 500


class LayerHierarchy:
    """
    Manages hierarchical layer grouping for nested learning.
    """
    
    def __init__(self, num_layers: int, num_levels: int = 3):
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.levels = self._create_hierarchy()
    
    def _create_hierarchy(self) -> Dict[int, List[int]]:
        """Create hierarchical grouping of layers."""
        levels = {}
        layers_per_level = self.num_layers // self.num_levels
        
        for level in range(self.num_levels):
            start = level * layers_per_level
            end = start + layers_per_level if level < self.num_levels - 1 else self.num_layers
            levels[level] = list(range(start, end))
        
        return levels
    
    def get_level_for_layer(self, layer_idx: int) -> int:
        """Get hierarchy level for a layer."""
        for level, layers in self.levels.items():
            if layer_idx in layers:
                return level
        return self.num_levels - 1
    
    def get_layers_at_level(self, level: int) -> List[int]:
        """Get all layers at a hierarchy level."""
        return self.levels.get(level, [])


class ProgressiveUnfreezer:
    """
    Manages progressive unfreezing of model layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: NestedLearningConfig
    ):
        self.model = model
        self.config = config
        self.current_step = 0
        self.unfrozen_layers = set()
        
        # Identify layers
        self.all_layers = self._identify_layers()
        self.num_layers = len(self.all_layers)
    
    def _identify_layers(self) -> List[nn.Module]:
        """Identify trainable layers in model."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.TransformerEncoderLayer)):
                layers.append(module)
        return layers
    
    def step(self, global_step: int):
        """Update layer freezing based on current step."""
        self.current_step = global_step
        
        # Determine which layers to unfreeze
        for i, step in enumerate(self.config.unfreeze_schedule):
            if global_step >= step:
                start_layer = max(0, self.num_layers - (i + 1) * self.config.layers_per_step)
                end_layer = self.num_layers
                
                for layer_idx in range(start_layer, end_layer):
                    if layer_idx < len(self.all_layers):
                        layer = self.all_layers[layer_idx]
                        for param in layer.parameters():
                            param.requires_grad = True
                        self.unfrozen_layers.add(layer_idx)
        
        logger.debug(f"Step {global_step}: Unfrozen {len(self.unfrozen_layers)}/{self.num_layers} layers")
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class HierarchicalDistiller:
    """
    Implements hierarchical knowledge distillation.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: NestedLearningConfig
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Create hierarchy
        self.hierarchy = LayerHierarchy(
            num_layers=self._count_layers(student_model),
            num_levels=config.num_hierarchy_levels
        )
    
    def _count_layers(self, model: nn.Module) -> int:
        """Count layers in model."""
        count = 0
        for _ in model.named_modules():
            count += 1
        return count
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hierarchy_level: int = 0
    ) -> torch.Tensor:
        """
        Compute hierarchical distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            hierarchy_level: Current hierarchy level
        
        Returns:
            Distillation loss
        """
        T = self.config.distillation_temperature
        
        # Soften probabilities
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)
        
        # Scale by hierarchy level (deeper layers get less weight)
        level_weight = 1.0 / (1 + hierarchy_level)
        
        return kl_loss * level_weight * self.config.distillation_alpha
    
    def get_intermediate_outputs(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Extract intermediate layer outputs."""
        outputs = {}
        hooks = []
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                outputs[layer_idx] = output.detach()
            return hook
        
        # Register hooks
        layer_idx = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.TransformerEncoderLayer)):
                hooks.append(module.register_forward_hook(hook_fn(layer_idx)))
                layer_idx += 1
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler based on layer depth.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: NestedLearningConfig
    ):
        self.model = model
        self.config = config
        self.hierarchy = LayerHierarchy(
            num_layers=self._count_trainable_layers(),
            num_levels=config.num_hierarchy_levels
        )
    
    def _count_trainable_layers(self) -> int:
        """Count trainable layers."""
        count = 0
        for module in self.model.modules():
            if any(p.requires_grad for p in module.parameters()):
                count += 1
        return count
    
    def get_lr_for_layer(self, layer_idx: int, base_lr: float) -> float:
        """
        Get learning rate for specific layer.
        
        Deeper layers (closer to output) get higher learning rates.
        """
        level = self.hierarchy.get_level_for_layer(layer_idx)
        
        # Exponential decay based on depth
        lr = base_lr * (self.config.lr_decay_factor ** (self.hierarchy.num_levels - level - 1))
        
        return max(lr, self.config.min_lr)
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with per-layer learning rates."""
        param_groups = []
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            if any(p.requires_grad for p in module.parameters()):
                lr = self.get_lr_for_layer(layer_idx, self.config.base_lr)
                param_groups.append({
                    'params': [p for p in module.parameters() if p.requires_grad],
                    'lr': lr,
                    'name': name
                })
                layer_idx += 1
        
        return torch.optim.AdamW(param_groups)


class NestedDropout:
    """
    Implements nested dropout schedule.
    """
    
    def __init__(self, config: NestedLearningConfig):
        self.config = config
        self.current_step = 0
    
    def get_dropout_rate(self, step: int) -> float:
        """Get dropout rate for current step."""
        schedule = self.config.dropout_schedule
        
        # Determine which phase we're in
        phase_size = self.config.curriculum_steps // len(schedule)
        phase = min(step // phase_size, len(schedule) - 1)
        
        return schedule[phase]
    
    def apply(self, x: torch.Tensor, step: int, training: bool = True) -> torch.Tensor:
        """Apply nested dropout."""
        if not training:
            return x
        
        rate = self.get_dropout_rate(step)
        return F.dropout(x, p=rate, training=True)


class NestedLearning:
    """
    Main interface for nested learning.
    
    Orchestrates progressive unfreezing, hierarchical distillation,
    adaptive learning rates, and nested dropout.
    
    Example:
        >>> nested = NestedLearning(student_model, teacher_model)
        >>> for step, batch in enumerate(dataloader):
        ...     loss = nested.training_step(batch, step)
        ...     loss.backward()
        ...     nested.optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        config: Optional[NestedLearningConfig] = None
    ):
        self.model = model
        self.teacher = teacher_model
        self.config = config or NestedLearningConfig()
        
        # Initialize components
        self.unfreezer = ProgressiveUnfreezer(model, self.config)
        
        if teacher_model is not None:
            self.distiller = HierarchicalDistiller(
                teacher_model, model, self.config
            )
        else:
            self.distiller = None
        
        self.lr_scheduler = AdaptiveLRScheduler(model, self.config)
        self.nested_dropout = NestedDropout(self.config)
        
        # Create optimizer
        self.optimizer = self.lr_scheduler.create_optimizer()
        
        self.global_step = 0
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            step: Current training step (auto-increments if None)
        
        Returns:
            Dictionary of losses
        """
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1
        
        # Update progressive unfreezing
        self.unfreezer.step(self.global_step)
        
        # Unpack batch
        inputs, targets = batch
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Task loss
        task_loss = F.cross_entropy(outputs, targets)
        
        losses = {'task_loss': task_loss}
        
        # Distillation loss
        if self.distiller is not None and self.teacher is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
            
            dist_loss = self.distiller.compute_distillation_loss(
                outputs, teacher_outputs
            )
            losses['distillation_loss'] = dist_loss
            total_loss = task_loss + dist_loss
        else:
            total_loss = task_loss
        
        losses['total_loss'] = total_loss
        
        # Update learning rates
        for param_group in self.optimizer.param_groups:
            layer_idx = self._get_layer_idx_from_name(param_group.get('name', ''))
            new_lr = self.lr_scheduler.get_lr_for_layer(
                layer_idx, self.config.base_lr
            )
            param_group['lr'] = new_lr
        
        return losses
    
    def _get_layer_idx_from_name(self, name: str) -> int:
        """Extract layer index from parameter name."""
        # Simple heuristic - can be customized
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part.isdigit():
                return int(part)
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'global_step': self.global_step,
            'trainable_params': self.unfreezer.get_trainable_params(),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'unfrozen_layers': len(self.unfreezer.unfrozen_layers),
            'total_layers': self.unfreezer.num_layers,
            'current_dropout': self.nested_dropout.get_dropout_rate(self.global_step),
        }


class CurriculumSampler:
    """
    Curriculum learning sampler for nested training.
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        difficulty_scores: List[float],
        curriculum_steps: int = 500
    ):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.curriculum_steps = curriculum_steps
        self.current_step = 0
    
    def get_sampler(self, step: int) -> torch.utils.data.Sampler:
        """Get appropriate sampler for current curriculum stage."""
        # Determine difficulty threshold
        progress = min(step / self.curriculum_steps, 1.0)
        threshold = progress * max(self.difficulty_scores)
        
        # Filter indices
        indices = [
            i for i, score in enumerate(self.difficulty_scores)
            if score <= threshold
        ]
        
        return torch.utils.data.SubsetRandomSampler(indices)


def apply_nested_learning(
    model: nn.Module,
    teacher_model: Optional[nn.Module] = None,
    strategy: str = "progressive",
    **kwargs
) -> NestedLearning:
    """
    Convenience function to apply nested learning.
    
    Args:
        model: Student model to train
        teacher_model: Optional teacher model for distillation
        strategy: Learning strategy ("progressive", "hierarchical", "adaptive", "combined")
        **kwargs: Additional config options
    
    Returns:
        NestedLearning instance
    """
    strategy_enum = NestedStrategy(strategy)
    config = NestedLearningConfig(strategy=strategy_enum, **kwargs)
    
    return NestedLearning(model, teacher_model, config)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Nested Learning Demo")
    print("=" * 50)
    
    # Create models
    student = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    teacher = nn.Sequential(
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Apply nested learning
    nested = NestedLearning(student, teacher)
    
    print(f"\nInitial trainable parameters: {nested.unfreezer.get_trainable_params()}")
    
    # Simulate training
    for step in range(10):
        batch = (torch.randn(4, 128), torch.randint(0, 10, (4,)))
        losses = nested.training_step(batch, step)
        
        if step % 3 == 0:
            stats = nested.get_stats()
            print(f"\nStep {step}:")
            print(f"  Trainable params: {stats['trainable_params']}")
            print(f"  Unfrozen layers: {stats['unfrozen_layers']}/{stats['total_layers']}")
            print(f"  Loss: {losses['total_loss'].item():.4f}")

"""
Comprehensive unit tests for QAD (Quantization-Aware Distillation) Loss Module.

Tests cover:
- QADDistillationLoss initialization
- KL divergence calculation
- Temperature scaling
- Label smoothing
- Adaptive temperature
- Hidden state and attention matching
- PerLayerQADLoss
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import time
from unittest.mock import MagicMock, patch

# Import the module under test
from nexus.models.sli.qad_loss import (
    QADDistillationLoss,
    QADLossConfig,
    QADLossType,
    QADLossStats,
    PerLayerQADLoss,
    QADLossError,
    get_qad_loss_config,
    compute_distillation_loss,
)
from nexus.models.sli.exceptions import SLIError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_batch_size():
    """Sample batch size for testing."""
    return 4


@pytest.fixture
def sample_num_classes():
    """Sample number of classes for testing."""
    return 1000


@pytest.fixture
def sample_hidden_dim():
    """Sample hidden dimension for testing."""
    return 768


@pytest.fixture
def sample_sequence_length():
    """Sample sequence length for testing."""
    return 128


@pytest.fixture
def sample_student_logits(sample_batch_size, sample_num_classes):
    """Create sample student logits."""
    torch.manual_seed(42)
    return torch.randn(sample_batch_size, sample_num_classes)


@pytest.fixture
def sample_teacher_logits(sample_batch_size, sample_num_classes):
    """Create sample teacher logits."""
    torch.manual_seed(43)
    return torch.randn(sample_batch_size, sample_num_classes)


@pytest.fixture
def sample_labels(sample_batch_size, sample_num_classes):
    """Create sample labels."""
    torch.manual_seed(44)
    return torch.randint(0, sample_num_classes, (sample_batch_size,))


@pytest.fixture
def sample_hidden_student(sample_batch_size, sample_sequence_length, sample_hidden_dim):
    """Create sample student hidden states."""
    torch.manual_seed(45)
    return torch.randn(sample_batch_size, sample_sequence_length, sample_hidden_dim)


@pytest.fixture
def sample_hidden_teacher(sample_batch_size, sample_sequence_length, sample_hidden_dim):
    """Create sample teacher hidden states."""
    torch.manual_seed(46)
    return torch.randn(sample_batch_size, sample_sequence_length, sample_hidden_dim)


@pytest.fixture
def sample_attention_student(sample_batch_size, sample_sequence_length):
    """Create sample student attention outputs."""
    torch.manual_seed(47)
    return torch.randn(sample_batch_size, sample_sequence_length, sample_sequence_length)


@pytest.fixture
def sample_attention_teacher(sample_batch_size, sample_sequence_length):
    """Create sample teacher attention outputs."""
    torch.manual_seed(48)
    return torch.randn(sample_batch_size, sample_sequence_length, sample_sequence_length)


@pytest.fixture
def qad_loss_default():
    """Create a QAD loss with default config."""
    return QADDistillationLoss(QADLossConfig())


@pytest.fixture
def qad_loss_kl():
    """Create a QAD loss with KL divergence type."""
    config = QADLossConfig(loss_type=QADLossType.KL_DIVERGENCE)
    return QADDistillationLoss(config)


@pytest.fixture
def qad_loss_mse():
    """Create a QAD loss with MSE type."""
    config = QADLossConfig(loss_type=QADLossType.MSE)
    return QADDistillationLoss(config)


@pytest.fixture
def qad_loss_cosine():
    """Create a QAD loss with cosine type."""
    config = QADLossConfig(loss_type=QADLossType.COSINE)
    return QADDistillationLoss(config)


@pytest.fixture
def qad_loss_combined():
    """Create a QAD loss with combined type."""
    config = QADLossConfig(loss_type=QADLossType.COMBINED)
    return QADDistillationLoss(config)


@pytest.fixture
def qad_loss_adaptive():
    """Create a QAD loss with adaptive temperature."""
    config = QADLossConfig(
        temperature=1.5,
        adaptive_temperature=True,
        min_temperature=1.0,
        max_temperature=2.0
    )
    return QADDistillationLoss(config)


# ============================================================================
# Test QADLossConfig
# ============================================================================

class TestQADLossConfig:
    """Test suite for QADLossConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QADLossConfig()
        
        assert config.temperature == 1.5
        assert config.alpha == 0.7
        assert config.beta == 0.3
        assert config.label_smoothing == 0.1
        assert config.loss_type == QADLossType.KL_DIVERGENCE
        assert config.use_attention_matching is True
        assert config.use_hidden_matching is True
        assert config.gradient_clip == 1.0
        assert config.adaptive_temperature is False
        assert config.min_temperature == 1.0
        assert config.max_temperature == 2.0

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = QADLossConfig(
            temperature=2.0,
            alpha=0.5,
            beta=0.5,
            label_smoothing=0.2,
            loss_type=QADLossType.MSE,
            use_attention_matching=False,
            use_hidden_matching=False,
            gradient_clip=2.0,
            adaptive_temperature=True,
            min_temperature=0.5,
            max_temperature=3.0
        )
        
        assert config.temperature == 2.0
        assert config.alpha == 0.5
        assert config.beta == 0.5
        assert config.label_smoothing == 0.2
        assert config.loss_type == QADLossType.MSE
        assert config.use_attention_matching is False
        assert config.use_hidden_matching is False
        assert config.gradient_clip == 2.0
        assert config.adaptive_temperature is True
        assert config.min_temperature == 0.5
        assert config.max_temperature == 3.0

    def test_config_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in \\[0, 1\]"):
            QADLossConfig(alpha=-0.1)
        
        with pytest.raises(ValueError, match="alpha must be in \\[0, 1\]"):
            QADLossConfig(alpha=1.1)

    def test_config_invalid_beta(self):
        """Test that invalid beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be in \\[0, 1\]"):
            QADLossConfig(beta=-0.1)
        
        with pytest.raises(ValueError, match="beta must be in \\[0, 1\]"):
            QADLossConfig(beta=1.1)

    def test_config_invalid_label_smoothing(self):
        """Test that invalid label_smoothing raises ValueError."""
        with pytest.raises(ValueError, match="label_smoothing must be in \\[0, 0.5\]"):
            QADLossConfig(label_smoothing=-0.1)
        
        with pytest.raises(ValueError, match="label_smoothing must be in \\[0, 0.5\]"):
            QADLossConfig(label_smoothing=0.6)

    def test_config_invalid_temperature(self):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            QADLossConfig(temperature=0.0)
        
        with pytest.raises(ValueError, match="temperature must be positive"):
            QADLossConfig(temperature=-1.0)

    def test_config_invalid_adaptive_temperature(self):
        """Test that invalid adaptive temperature range raises ValueError."""
        with pytest.raises(ValueError, match="min_temperature must be less than max_temperature"):
            QADLossConfig(
                adaptive_temperature=True,
                min_temperature=2.0,
                max_temperature=1.0
            )

    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = QADLossConfig(temperature=1.5, alpha=0.7)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['temperature'] == 1.5
        assert config_dict['alpha'] == 0.7
        assert config_dict['loss_type'] == 'kl_divergence'
        assert config_dict['use_attention_matching'] is True

    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        data = {
            'temperature': 2.0,
            'alpha': 0.5,
            'beta': 0.4,
            'label_smoothing': 0.2,
            'loss_type': 'mse',
            'use_attention_matching': False,
            'use_hidden_matching': False,
            'gradient_clip': 2.0,
            'adaptive_temperature': True,
            'min_temperature': 0.5,
            'max_temperature': 3.0
        }
        
        config = QADLossConfig.from_dict(data)
        
        assert config.temperature == 2.0
        assert config.alpha == 0.5
        assert config.beta == 0.4
        assert config.label_smoothing == 0.2
        assert config.loss_type == QADLossType.MSE
        assert config.use_attention_matching is False
        assert config.use_hidden_matching is False
        assert config.gradient_clip == 2.0
        assert config.adaptive_temperature is True
        assert config.min_temperature == 0.5
        assert config.max_temperature == 3.0

    def test_config_from_dict_defaults(self):
        """Test configuration from dict with missing values uses defaults."""
        data = {'temperature': 1.0}
        
        config = QADLossConfig.from_dict(data)
        
        assert config.temperature == 1.0
        assert config.alpha == 0.7  # Default
        assert config.loss_type == QADLossType.KL_DIVERGENCE  # Default


# ============================================================================
# Test QADLossStats
# ============================================================================

class TestQADLossStats:
    """Test suite for QADLossStats dataclass."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = QADLossStats()
        
        assert stats.total_loss == 0.0
        assert stats.distillation_loss == 0.0
        assert stats.hard_target_loss == 0.0
        assert stats.hidden_matching_loss == 0.0
        assert stats.attention_matching_loss == 0.0
        assert stats.temperature == 1.0
        assert stats.step == 0

    def test_stats_to_dict(self):
        """Test statistics serialization to dict."""
        stats = QADLossStats(
            total_loss=1.5,
            distillation_loss=0.8,
            hard_target_loss=0.7,
            step=100
        )
        stats_dict = stats.to_dict()
        
        assert isinstance(stats_dict, dict)
        assert stats_dict['total_loss'] == 1.5
        assert stats_dict['distillation_loss'] == 0.8
        assert stats_dict['hard_target_loss'] == 0.7
        assert stats_dict['step'] == 100


# ============================================================================
# Test QADDistillationLoss Initialization
# ============================================================================

class TestQADDistillationLossInitialization:
    """Test suite for QADDistillationLoss initialization."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        loss_fn = QADDistillationLoss()
        
        assert loss_fn.config is not None
        assert isinstance(loss_fn._stats, QADLossStats)
        assert isinstance(loss_fn._history, list)
        assert isinstance(loss_fn._lock, threading.RLock)
        assert loss_fn._current_temperature == loss_fn.config.temperature

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = QADLossConfig(temperature=2.0, alpha=0.5)
        loss_fn = QADDistillationLoss(config)
        
        assert loss_fn.config == config
        assert loss_fn._current_temperature == 2.0

    def test_initialization_thread_safety(self):
        """Test that initialization is thread-safe."""
        configs = [QADLossConfig(temperature=t) for t in [1.0, 1.5, 2.0]]
        loss_fns = []
        
        def create_loss(config):
            loss_fns.append(QADDistillationLoss(config))
        
        threads = [threading.Thread(target=create_loss, args=(c,)) for c in configs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(loss_fns) == 3


# ============================================================================
# Test KL Divergence Loss
# ============================================================================

class TestKLDivergenceLoss:
    """Test suite for KL divergence loss computation."""

    def test_kl_divergence_basic(self, qad_loss_kl, sample_student_logits, sample_teacher_logits):
        """Test basic KL divergence computation."""
        loss = qad_loss_kl(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # KL divergence is non-negative

    def test_kl_divergence_with_temperature(self):
        """Test KL divergence with different temperatures."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        # Lower temperature
        config_low = QADLossConfig(temperature=1.0)
        loss_low = QADDistillationLoss(config_low)(student_logits, teacher_logits)
        
        # Higher temperature
        config_high = QADLossConfig(temperature=2.0)
        loss_high = QADDistillationLoss(config_high)(student_logits, teacher_logits)
        
        # Both should be valid tensors
        assert isinstance(loss_low, torch.Tensor)
        assert isinstance(loss_high, torch.Tensor)

    def test_kl_divergence_same_logits(self, qad_loss_kl):
        """Test KL divergence with identical logits."""
        logits = torch.randn(4, 100)
        
        loss = qad_loss_kl(
            student_logits=logits,
            teacher_logits=logits
        )
        
        # Should be very small when logits are identical
        assert loss.item() < 0.1

    def test_kl_divergence_temperature_scaling(self):
        """Test that temperature scaling works correctly."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        temperature = 2.0
        
        config = QADLossConfig(temperature=temperature)
        loss_fn = QADDistillationLoss(config)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        # Manually compute expected KL divergence
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        expected_kl = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        expected_kl = expected_kl * (temperature ** 2)
        
        # Should be close
        assert torch.isclose(loss, expected_kl, rtol=1e-4)

    def test_kl_divergence_batchmean_reduction(self, qad_loss_kl):
        """Test that KL divergence uses batchmean reduction."""
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            student_logits = torch.randn(batch_size, 100)
            teacher_logits = torch.randn(batch_size, 100)
            
            loss = qad_loss_kl(student_logits, teacher_logits)
            
            # Should produce scalar
            assert loss.dim() == 0
            assert loss.item() >= 0


# ============================================================================
# Test MSE Loss
# ============================================================================

class TestMSELoss:
    """Test suite for MSE loss computation."""

    def test_mse_loss_basic(self, qad_loss_mse, sample_student_logits, sample_teacher_logits):
        """Test basic MSE loss computation."""
        loss = qad_loss_mse(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_mse_loss_same_logits(self, qad_loss_mse):
        """Test MSE loss with identical logits."""
        logits = torch.randn(4, 100)
        
        loss = qad_loss_mse(
            student_logits=logits,
            teacher_logits=logits
        )
        
        # Should be near zero
        assert loss.item() < 1e-5


# ============================================================================
# Test Cosine Loss
# ============================================================================

class TestCosineLoss:
    """Test suite for cosine similarity loss computation."""

    def test_cosine_loss_basic(self, qad_loss_cosine, sample_student_logits, sample_teacher_logits):
        """Test basic cosine loss computation."""
        loss = qad_loss_cosine(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert 0 <= loss.item() <= 2  # Cosine loss is in [0, 2]

    def test_cosine_loss_same_logits(self, qad_loss_cosine):
        """Test cosine loss with identical logits."""
        logits = torch.randn(4, 100)
        
        loss = qad_loss_cosine(
            student_logits=logits,
            teacher_logits=logits
        )
        
        # Should be near zero for identical logits
        assert loss.item() < 1e-5


# ============================================================================
# Test Combined Loss
# ============================================================================

class TestCombinedLoss:
    """Test suite for combined loss computation."""

    def test_combined_loss_basic(self, qad_loss_combined, sample_student_logits, sample_teacher_logits):
        """Test basic combined loss computation."""
        loss = qad_loss_combined(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_combined_loss_components(self, sample_student_logits, sample_teacher_logits):
        """Test that combined loss includes both KL and MSE."""
        config = QADLossConfig(loss_type=QADLossType.COMBINED)
        loss_fn = QADDistillationLoss(config)
        
        loss = loss_fn(sample_student_logits, sample_teacher_logits)
        
        # Combined loss should be average of KL and MSE
        config_kl = QADLossConfig(loss_type=QADLossType.KL_DIVERGENCE)
        config_mse = QADLossConfig(loss_type=QADLossType.MSE)
        
        loss_kl = QADDistillationLoss(config_kl)(sample_student_logits, sample_teacher_logits)
        loss_mse = QADDistillationLoss(config_mse)(sample_student_logits, sample_teacher_logits)
        
        expected = 0.5 * loss_kl + 0.5 * loss_mse
        assert torch.isclose(loss, expected, rtol=1e-4)


# ============================================================================
# Test Hard Target Loss
# ============================================================================

class TestHardTargetLoss:
    """Test suite for hard target (cross-entropy) loss."""

    def test_hard_target_loss(self, qad_loss_default, sample_student_logits, sample_labels):
        """Test hard target loss computation."""
        loss = qad_loss_default(
            student_logits=sample_student_logits,
            teacher_logits=sample_student_logits,  # Doesn't matter for hard target
            labels=sample_labels
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_hard_target_without_labels(self, qad_loss_default, sample_student_logits, sample_teacher_logits):
        """Test loss without hard target (no labels)."""
        loss = qad_loss_default(
            student_logits=sample_student_logits,
            teacher_logits=sample_teacher_logits
        )
        
        assert isinstance(loss, torch.Tensor)
        # Should still work without labels


# ============================================================================
# Test Label Smoothing
# ============================================================================

class TestLabelSmoothing:
    """Test suite for label smoothing functionality."""

    def test_label_smoothing_enabled(self, sample_student_logits, sample_labels):
        """Test label smoothing with smoothing factor."""
        config = QADLossConfig(label_smoothing=0.1)
        loss_fn = QADDistillationLoss(config)
        
        loss = loss_fn(
            student_logits=sample_student_logits,
            teacher_logits=sample_student_logits,
            labels=sample_labels
        )
        
        assert isinstance(loss, torch.Tensor)

    def test_label_smoothing_zero(self, sample_student_logits, sample_labels):
        """Test with zero label smoothing (standard cross-entropy)."""
        config = QADLossConfig(label_smoothing=0.0)
        loss_fn = QADDistillationLoss(config)
        
        loss = loss_fn(
            student_logits=sample_student_logits,
            teacher_logits=sample_student_logits,
            labels=sample_labels
        )
        
        assert isinstance(loss, torch.Tensor)

    def test_label_smoothing_distribution(self):
        """Test that label smoothing produces correct distribution."""
        config = QADLossConfig(label_smoothing=0.1)
        loss_fn = QADDistillationLoss(config)
        
        student_logits = torch.randn(1, 10)
        labels = torch.tensor([5])
        
        # Test the internal computation
        num_classes = student_logits.size(-1)
        log_probs = F.log_softmax(student_logits, dim=-1)
        
        # Create smoothed labels manually
        smoothed_labels = torch.zeros_like(log_probs)
        smoothed_labels.fill_(config.label_smoothing / (num_classes - 1))
        smoothed_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - config.label_smoothing)
        
        expected_loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()
        
        # Compute through the loss function
        hard_loss = loss_fn._compute_hard_target_loss(student_logits, labels)
        
        # Should be close
        assert torch.isclose(hard_loss, expected_loss, rtol=1e-4)


# ============================================================================
# Test Temperature Scaling
# ============================================================================

class TestTemperatureScaling:
    """Test suite for temperature scaling functionality."""

    def test_temperature_effect_on_softmax(self):
        """Test that temperature affects softmax distribution."""
        logits = torch.randn(1, 10)
        
        # Low temperature -> sharper distribution
        probs_low_temp = F.softmax(logits / 0.5, dim=-1)
        
        # High temperature -> smoother distribution
        probs_high_temp = F.softmax(logits / 2.0, dim=-1)
        
        # High temp should have higher entropy
        entropy_low = -(probs_low_temp * torch.log(probs_low_temp + 1e-10)).sum()
        entropy_high = -(probs_high_temp * torch.log(probs_high_temp + 1e-10)).sum()
        
        assert entropy_high > entropy_low

    def test_temperature_squared_scaling(self):
        """Test that KL loss is scaled by temperature squared."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        temperature = 2.0
        
        config = QADLossConfig(temperature=temperature, loss_type=QADLossType.KL_DIVERGENCE)
        loss_fn = QADDistillationLoss(config)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        # Manual computation
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kl_div = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        expected = kl_div * (temperature ** 2)
        
        assert torch.isclose(loss, expected, rtol=1e-4)


# ============================================================================
# Test Adaptive Temperature
# ============================================================================

class TestAdaptiveTemperature:
    """Test suite for adaptive temperature functionality."""

    def test_adaptive_temperature_increases_when_loss_decreasing(self, qad_loss_adaptive):
        """Test that temperature increases when loss is decreasing."""
        initial_temp = qad_loss_adaptive._current_temperature
        
        # Simulate decreasing loss
        for _ in range(20):
            student_logits = torch.randn(4, 100)
            teacher_logits = torch.randn(4, 100)
            labels = torch.randint(0, 100, (4,))
            
            loss = qad_loss_adaptive(student_logits, teacher_logits, labels=labels)
        
        # Temperature should have increased (or stayed at max)
        final_temp = qad_loss_adaptive.get_stats()['temperature']
        assert final_temp >= initial_temp

    def test_adaptive_temperature_bounds(self, qad_loss_adaptive):
        """Test that adaptive temperature stays within bounds."""
        min_temp = qad_loss_adaptive.config.min_temperature
        max_temp = qad_loss_adaptive.config.max_temperature
        
        # Run many iterations
        for _ in range(100):
            student_logits = torch.randn(4, 100)
            teacher_logits = torch.randn(4, 100)
            
            qad_loss_adaptive(student_logits, teacher_logits)
        
        final_temp = qad_loss_adaptive.get_stats()['temperature']
        assert min_temp <= final_temp <= max_temp

    def test_adaptive_temperature_disabled(self, qad_loss_default):
        """Test that adaptive temperature doesn't change when disabled."""
        initial_temp = qad_loss_default._current_temperature
        
        for _ in range(10):
            student_logits = torch.randn(4, 100)
            teacher_logits = torch.randn(4, 100)
            
            qad_loss_default(student_logits, teacher_logits)
        
        final_temp = qad_loss_default.get_stats()['temperature']
        assert final_temp == initial_temp

    def test_set_temperature_disables_adaptive(self, qad_loss_adaptive):
        """Test that setting temperature manually disables adaptive mode."""
        assert qad_loss_adaptive.config.adaptive_temperature is True
        
        qad_loss_adaptive.set_temperature(1.2)
        
        assert qad_loss_adaptive.config.adaptive_temperature is False
        assert qad_loss_adaptive._current_temperature == 1.2


# ============================================================================
# Test Hidden State Matching
# ============================================================================

class TestHiddenStateMatching:
    """Test suite for hidden state matching loss."""

    def test_hidden_matching_basic(self, qad_loss_default, sample_hidden_student, sample_hidden_teacher):
        """Test basic hidden state matching."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        loss = qad_loss_default(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hidden_student=sample_hidden_student,
            hidden_teacher=sample_hidden_teacher
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_hidden_matching_with_mask(self, qad_loss_default, sample_hidden_student, sample_hidden_teacher):
        """Test hidden state matching with attention mask."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        mask = torch.ones(4, 128)
        mask[:, 64:] = 0  # Mask out second half
        
        loss = qad_loss_default(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hidden_student=sample_hidden_student,
            hidden_teacher=sample_hidden_teacher,
            mask=mask
        )
        
        assert isinstance(loss, torch.Tensor)

    def test_hidden_matching_different_dims(self, qad_loss_default):
        """Test hidden state matching with different dimensions."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        hidden_student = torch.randn(4, 128, 512)  # Different hidden dim
        hidden_teacher = torch.randn(4, 128, 768)
        
        loss = qad_loss_default(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hidden_student=hidden_student,
            hidden_teacher=hidden_teacher
        )
        
        assert isinstance(loss, torch.Tensor)

    def test_hidden_matching_disabled(self):
        """Test that hidden matching is skipped when disabled."""
        config = QADLossConfig(use_hidden_matching=False)
        loss_fn = QADDistillationLoss(config)
        
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        hidden_student = torch.randn(4, 128, 768)
        hidden_teacher = torch.randn(4, 128, 768)
        
        loss = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hidden_student=hidden_student,
            hidden_teacher=hidden_teacher
        )
        
        # Should not include hidden matching loss
        assert isinstance(loss, torch.Tensor)


# ============================================================================
# Test Attention Matching
# ============================================================================

class TestAttentionMatching:
    """Test suite for attention matching loss."""

    def test_attention_matching_basic(self, qad_loss_default, sample_attention_student, sample_attention_teacher):
        """Test basic attention matching."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        loss = qad_loss_default(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_student=sample_attention_student,
            attention_teacher=sample_attention_teacher
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_attention_matching_with_mask(self, qad_loss_default, sample_attention_student, sample_attention_teacher):
        """Test attention matching with mask."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        mask = torch.ones(4, 128)
        
        loss = qad_loss_default(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_student=sample_attention_student,
            attention_teacher=sample_attention_teacher,
            mask=mask
        )
        
        assert isinstance(loss, torch.Tensor)

    def test_attention_matching_disabled(self):
        """Test that attention matching is skipped when disabled."""
        config = QADLossConfig(use_attention_matching=False)
        loss_fn = QADDistillationLoss(config)
        
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        attention_student = torch.randn(4, 128, 128)
        attention_teacher = torch.randn(4, 128, 128)
        
        loss = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_student=attention_student,
            attention_teacher=attention_teacher
        )
        
        assert isinstance(loss, torch.Tensor)


# ============================================================================
# Test Statistics and History
# ============================================================================

class TestStatisticsAndHistory:
    """Test suite for statistics tracking and history."""

    def test_get_stats(self, qad_loss_default, sample_student_logits, sample_teacher_logits):
        """Test getting statistics after forward pass."""
        qad_loss_default(sample_student_logits, sample_teacher_logits)
        
        stats = qad_loss_default.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_loss' in stats
        assert 'distillation_loss' in stats
        assert 'step' in stats
        assert stats['step'] == 1

    def test_get_history(self, qad_loss_default):
        """Test getting loss history."""
        # Run multiple iterations
        for _ in range(5):
            student_logits = torch.randn(4, 100)
            teacher_logits = torch.randn(4, 100)
            qad_loss_default(student_logits, teacher_logits)
        
        history = qad_loss_default.get_history()
        
        assert isinstance(history, list)
        assert len(history) == 5
        assert all(isinstance(entry, dict) for entry in history)

    def test_get_history_n(self, qad_loss_default):
        """Test getting limited history."""
        # Run multiple iterations
        for _ in range(10):
            student_logits = torch.randn(4, 100)
            teacher_logits = torch.randn(4, 100)
            qad_loss_default(student_logits, teacher_logits)
        
        history_3 = qad_loss_default.get_history(n=3)
        
        assert len(history_3) == 3

    def test_reset_stats(self, qad_loss_default, sample_student_logits, sample_teacher_logits):
        """Test resetting statistics."""
        # Run some iterations
        for _ in range(5):
            qad_loss_default(sample_student_logits, sample_teacher_logits)
        
        # Reset
        qad_loss_default.reset_stats()
        
        stats = qad_loss_default.get_stats()
        assert stats['step'] == 0
        assert stats['total_loss'] == 0.0
        assert len(qad_loss_default.get_history()) == 0

    def test_history_max_size(self):
        """Test that history is limited to max size."""
        config = QADLossConfig()
        loss_fn = QADDistillationLoss(config)
        
        # Run more iterations than max history
        for i in range(1100):
            student_logits = torch.randn(4, 100)
            teacher_logits = torch.randn(4, 100)
            loss_fn(student_logits, teacher_logits)
        
        history = loss_fn.get_history()
        assert len(history) <= 1000  # Max history size


# ============================================================================
# Test PerLayerQADLoss
# ============================================================================

class TestPerLayerQADLoss:
    """Test suite for PerLayerQADLoss."""

    def test_per_layer_initialization(self):
        """Test per-layer loss initialization."""
        loss_fn = PerLayerQADLoss(num_layers=4)
        
        assert loss_fn.num_layers == 4
        assert len(loss_fn.layer_weights) == 4
        assert sum(loss_fn.layer_weights) == 1.0

    def test_per_layer_custom_weights(self):
        """Test per-layer loss with custom weights."""
        weights = [0.4, 0.3, 0.2, 0.1]
        loss_fn = PerLayerQADLoss(num_layers=4, layer_weights=weights)
        
        assert loss_fn.layer_weights == weights

    def test_per_layer_forward(self):
        """Test per-layer forward pass."""
        loss_fn = PerLayerQADLoss(num_layers=3)
        
        layer_outputs_student = [
            torch.randn(4, 128, 768) for _ in range(3)
        ]
        layer_outputs_teacher = [
            torch.randn(4, 128, 768) for _ in range(3)
        ]
        final_logits_student = torch.randn(4, 100)
        final_logits_teacher = torch.randn(4, 100)
        labels = torch.randint(0, 100, (4,))
        
        loss = loss_fn(
            layer_outputs_student=layer_outputs_student,
            layer_outputs_teacher=layer_outputs_teacher,
            final_logits_student=final_logits_student,
            final_logits_teacher=final_logits_teacher,
            labels=labels
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_per_layer_loss_computation(self):
        """Test that per-layer losses are computed correctly."""
        weights = [0.5, 0.5]
        loss_fn = PerLayerQADLoss(num_layers=2, layer_weights=weights)
        
        layer_outputs_student = [
            torch.randn(4, 128, 768),
            torch.randn(4, 128, 768)
        ]
        layer_outputs_teacher = [
            torch.randn(4, 128, 768),
            torch.randn(4, 128, 768)
        ]
        final_logits_student = torch.randn(4, 100)
        final_logits_teacher = torch.randn(4, 100)
        
        loss = loss_fn(
            layer_outputs_student=layer_outputs_student,
            layer_outputs_teacher=layer_outputs_teacher,
            final_logits_student=final_logits_student,
            final_logits_teacher=final_logits_teacher
        )
        
        assert isinstance(loss, torch.Tensor)


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling."""

    def test_qad_loss_error(self):
        """Test QADLossError creation."""
        error = QADLossError("Test error")
        assert "Test error" in str(error)

    def test_qad_loss_error_with_layer(self):
        """Test QADLossError with layer index."""
        error = QADLossError("Test error", layer_idx=5)
        assert "layer 5" in str(error)
        assert error.layer_idx == 5

    def test_qad_loss_error_inheritance(self):
        """Test QADLossError inherits from SLIError."""
        error = QADLossError("Test")
        assert isinstance(error, SLIError)

    def test_unknown_loss_type(self):
        """Test that unknown loss type raises error."""
        config = MagicMock()
        config.loss_type = "unknown_type"
        config.alpha = 0.7
        config.beta = 0.3
        config.use_hidden_matching = False
        config.use_attention_matching = False
        config.adaptive_temperature = False
        config.temperature = 1.5
        config.label_smoothing = 0.0
        
        loss_fn = QADDistillationLoss.__new__(QADDistillationLoss)
        loss_fn.config = config
        loss_fn._stats = QADLossStats()
        loss_fn._history = []
        loss_fn._lock = threading.RLock()
        loss_fn._current_temperature = 1.5
        loss_fn._loss_history = []
        
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        with pytest.raises(QADLossError):
            loss_fn._compute_distillation_loss(student_logits, teacher_logits, 1.5)


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_get_qad_loss_config(self):
        """Test get_qad_loss_config convenience function."""
        config = get_qad_loss_config(
            temperature=1.5,
            alpha=0.7,
            label_smoothing=0.1,
            adaptive=True
        )
        
        assert config.temperature == 1.5
        assert config.alpha == 0.7
        assert config.label_smoothing == 0.1
        assert config.adaptive_temperature is True

    def test_get_qad_loss_config_defaults(self):
        """Test get_qad_loss_config with defaults."""
        config = get_qad_loss_config()
        
        assert config.temperature == 1.5
        assert config.alpha == 0.7
        assert config.label_smoothing == 0.1
        assert config.adaptive_temperature is False

    def test_compute_distillation_loss(self):
        """Test compute_distillation_loss convenience function."""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        loss = compute_distillation_loss(student_logits, teacher_logits, temperature=1.5)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


# ============================================================================
# Test Thread Safety
# ============================================================================

class TestThreadSafety:
    """Test suite for thread safety."""

    def test_concurrent_forward_passes(self, qad_loss_default):
        """Test concurrent forward passes."""
        errors = []
        
        def forward_pass():
            try:
                for _ in range(20):
                    student_logits = torch.randn(4, 100)
                    teacher_logits = torch.randn(4, 100)
                    qad_loss_default(student_logits, teacher_logits)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=forward_pass) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_stats_access(self, qad_loss_default):
        """Test concurrent stats access."""
        errors = []
        
        def worker():
            try:
                for _ in range(20):
                    student_logits = torch.randn(4, 100)
                    teacher_logits = torch.randn(4, 100)
                    qad_loss_default(student_logits, teacher_logits)
                    qad_loss_default.get_stats()
                    qad_loss_default.get_history()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_sample_batch(self, qad_loss_default):
        """Test with single sample batch."""
        student_logits = torch.randn(1, 100)
        teacher_logits = torch.randn(1, 100)
        labels = torch.tensor([5])
        
        loss = qad_loss_default(student_logits, teacher_logits, labels=labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_large_batch(self, qad_loss_default):
        """Test with large batch size."""
        student_logits = torch.randn(128, 100)
        teacher_logits = torch.randn(128, 100)
        labels = torch.randint(0, 100, (128,))
        
        loss = qad_loss_default(student_logits, teacher_logits, labels=labels)
        
        assert isinstance(loss, torch.Tensor)

    def test_single_class(self, qad_loss_default):
        """Test with single class (edge case)."""
        student_logits = torch.randn(4, 1)
        teacher_logits = torch.randn(4, 1)
        labels = torch.zeros(4, dtype=torch.long)
        
        loss = qad_loss_default(student_logits, teacher_logits, labels=labels)
        
        assert isinstance(loss, torch.Tensor)

    def test_very_high_temperature(self):
        """Test with very high temperature."""
        config = QADLossConfig(temperature=10.0)
        loss_fn = QADDistillationLoss(config)
        
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        assert isinstance(loss, torch.Tensor)

    def test_very_low_temperature(self):
        """Test with very low temperature."""
        config = QADLossConfig(temperature=0.1)
        loss_fn = QADDistillationLoss(config)
        
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        assert isinstance(loss, torch.Tensor)

    def test_identical_hidden_states(self, qad_loss_default):
        """Test hidden matching with identical states."""
        hidden = torch.randn(4, 128, 768)
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        
        loss = qad_loss_default(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hidden_student=hidden,
            hidden_teacher=hidden
        )
        
        # Hidden matching loss should be near zero
        stats = qad_loss_default.get_stats()
        assert stats['hidden_matching_loss'] < 1e-5

    def test_all_zeros_logits(self, qad_loss_kl):
        """Test with all-zero logits."""
        student_logits = torch.zeros(4, 100)
        teacher_logits = torch.zeros(4, 100)
        
        loss = qad_loss_kl(student_logits, teacher_logits)
        
        assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

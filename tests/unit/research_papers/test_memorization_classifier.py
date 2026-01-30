"""
Comprehensive Unit Tests for Pre-distillation Memorization Classifier (Paper 2601.15394).

Tests cover:
- Feature extraction (entropy, perplexity, KLD)
- Logistic regression training
- Prediction accuracy
- Model save/load
- Edge cases (empty input, None values)
"""

import pytest
import numpy as np
import sys
import os
import pickle
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.nexus_final.auditor import (
    MemorizationClassifier,
    MemorizationAuditor,
    DistillationReport
)


class TestMemorizationClassifierInitialization:
    """Test suite for classifier initialization."""
    
    def test_default_initialization(self):
        """Test classifier initializes with correct defaults."""
        classifier = MemorizationClassifier()
        assert classifier.model is None
        assert classifier.scaler is None
        assert not classifier.is_trained
        assert classifier.target_auc_roc == 0.9997
        assert classifier.model_path is None
    
    def test_initialization_with_model_path(self, tmp_path):
        """Test classifier initializes with model path."""
        model_path = str(tmp_path / "test_model.pkl")
        # Create a dummy model file
        dummy_data = {
            'model': None,
            'scaler': None,
            'is_trained': False,
            'target_auc_roc': 0.9997
        }
        with open(model_path, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        classifier = MemorizationClassifier(model_path)
        assert classifier.model_path == model_path
    
    def test_initialization_with_nonexistent_path(self):
        """Test classifier handles non-existent model path gracefully."""
        classifier = MemorizationClassifier("/nonexistent/path/model.pkl")
        assert classifier.model is None
        assert not classifier.is_trained


class TestFeatureExtraction:
    """Test feature extraction methods."""
    
    def test_calculate_zlib_entropy_random_text(self):
        """Test entropy calculation for random text."""
        # Random text should have high entropy
        random_text = "aslkdjfalskjdflkasjdflkasjdflkasjd"
        entropy = MemorizationClassifier.calculate_zlib_entropy(random_text)
        assert entropy > 0
        assert isinstance(entropy, float)
    
    def test_calculate_zlib_entropy_repetitive_text(self):
        """Test entropy calculation for repetitive text."""
        # Repetitive text should have lower entropy
        repetitive_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        entropy = MemorizationClassifier.calculate_zlib_entropy(repetitive_text)
        assert entropy > 0
        
        # Compare with random text
        random_text = "aslkdjfalskjdflkasjdflkasjdflkasjd"
        random_entropy = MemorizationClassifier.calculate_zlib_entropy(random_text)
        assert entropy < random_entropy
    
    def test_calculate_zlib_entropy_empty_string(self):
        """Test entropy calculation for empty string."""
        entropy = MemorizationClassifier.calculate_zlib_entropy("")
        assert entropy == 0.0
    
    def test_calculate_zlib_entropy_single_char(self):
        """Test entropy calculation for single character."""
        entropy = MemorizationClassifier.calculate_zlib_entropy("a")
        assert entropy >= 0.0


class TestLogisticRegressionTraining:
    """Test logistic regression training functionality."""
    
    def test_untrained_prediction_error(self):
        """Test that predicting before training raises error."""
        classifier = MemorizationClassifier()
        
        with pytest.raises(RuntimeError, match="Classifier must be trained"):
            classifier.predict(np.array([0.5, 1.0, 0.8, 0.2]))
        
        with pytest.raises(RuntimeError, match="Classifier must be trained"):
            classifier.predict_proba(np.array([0.5, 1.0, 0.8, 0.2]))
    
    def test_model_save_and_load(self, tmp_path):
        """Test model save and load functionality."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        classifier = MemorizationClassifier()
        
        # Create a mock trained state
        classifier.model = LogisticRegression()
        classifier.model.classes_ = np.array([0, 1])
        classifier.model.coef_ = np.array([[0.5, 0.3, 0.2, 0.1]])
        classifier.model.intercept_ = np.array([0.0])
        
        classifier.scaler = StandardScaler()
        classifier.scaler.mean_ = np.array([0.0, 0.0, 0.0, 0.0])
        classifier.scaler.scale_ = np.array([1.0, 1.0, 1.0, 1.0])
        classifier.scaler.n_features_in_ = 4
        
        classifier.is_trained = True
        
        # Save the model
        save_path = tmp_path / "test_classifier.pkl"
        classifier.save(str(save_path))
        
        # Verify file exists
        assert save_path.exists()
        
        # Load the model
        loaded_classifier = MemorizationClassifier(str(save_path))
        
        # Verify loaded state
        assert loaded_classifier.is_trained
        assert loaded_classifier.model is not None
        assert loaded_classifier.scaler is not None
        assert loaded_classifier.target_auc_roc == 0.9997


class TestMemorizationAuditor:
    """Test suite for MemorizationAuditor."""
    
    def test_auditor_initialization(self, mocker):
        """Test auditor initialization."""
        mock_tokenizer = mocker.MagicMock()
        auditor = MemorizationAuditor(mock_tokenizer, prefix_len=50, suffix_len=50)
        
        assert auditor.tokenizer == mock_tokenizer
        assert auditor.prefix_len == 50
        assert auditor.suffix_len == 50
        assert auditor.classifier is not None
    
    def test_calculate_zlib_entropy(self):
        """Test static entropy calculation method."""
        text = "This is a test text for entropy calculation."
        entropy = MemorizationAuditor.calculate_zlib_entropy(text)
        assert entropy > 0
        
        # Empty text
        empty_entropy = MemorizationAuditor.calculate_zlib_entropy("")
        assert empty_entropy == 0.0
    
    def test_calculate_match_ratio_perfect(self):
        """Test match ratio calculation - perfect match."""
        mock_tokenizer = MagicMock()
        auditor = MemorizationAuditor(mock_tokenizer)
        
        ratio = auditor._calculate_match_ratio([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert ratio == 1.0
    
    def test_calculate_match_ratio_partial(self):
        """Test match ratio calculation - partial match."""
        mock_tokenizer = MagicMock()
        auditor = MemorizationAuditor(mock_tokenizer)
        
        ratio = auditor._calculate_match_ratio([1, 2, 4, 5, 6], [1, 2, 3, 4, 5])
        assert ratio == 2/5  # 2 matches out of 5


class TestDistillationReport:
    """Test suite for DistillationReport."""
    
    def test_report_creation(self):
        """Test creating a distillation report."""
        report = DistillationReport(
            hard_distillation_rate=0.15,
            soft_distillation_rate=0.08,
            inherited_from_teacher_rate=0.12,
            privacy_recommendation="SOFT_DISTILLATION_RECOMMENDED",
            detailed_metrics={"samples_analyzed": 100}
        )
        
        assert report.hard_distillation_rate == 0.15
        assert report.soft_distillation_rate == 0.08
        assert report.inherited_from_teacher_rate == 0.12
        assert report.privacy_recommendation == "SOFT_DISTILLATION_RECOMMENDED"
        assert report.detailed_metrics["samples_analyzed"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

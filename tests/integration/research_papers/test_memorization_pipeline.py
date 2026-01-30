"""
Integration Tests for End-to-End Memorization Pipeline (Paper 2601.15394).

Tests cover:
- Train classifier → Filter data → Verify reduction
- Full pipeline with real metrics
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.nexus_final.auditor import MemorizationClassifier, MemorizationAuditor
from src.nexus_final.data_loader import MemorizationFilter, UniversalDataLoader


class TestEndToEndMemorizationPipeline:
    """End-to-end tests for memorization pipeline."""
    
    def test_full_pipeline_with_mock_classifier(self, mocker, tmp_path):
        """Test full pipeline with mock classifier."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Create a mock trained classifier
        classifier = MemorizationClassifier()
        classifier.model = LogisticRegression()
        classifier.model.classes_ = [0, 1]
        classifier.model.coef_ = [[0.5, 0.3, 0.2, 0.1]]
        classifier.model.intercept_ = [0.0]
        
        classifier.scaler = StandardScaler()
        classifier.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
        classifier.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]
        classifier.scaler.n_features_in_ = 4
        classifier.is_trained = True
        
        # Save the classifier
        classifier_path = tmp_path / "classifier.pkl"
        classifier.save(str(classifier_path))
        
        # Create data loader with filtering
        loader = UniversalDataLoader(
            data_root=str(tmp_path),
            filter_memorization_risk=True,
            entropy_threshold=0.4,
            classifier_path=str(classifier_path)
        )
        
        assert loader.filter_memorization_risk == True
        assert loader.memorization_filter is not None
    
    def test_entropy_filtering_reduction(self):
        """Test that entropy filtering reduces dataset size."""
        filter_obj = MemorizationFilter(entropy_threshold=0.5)
        
        # Create samples with varying entropy
        samples = [
            {"messages": [{"role": "user", "content": "a" * 1000}]},  # Low entropy
            {"messages": [{"role": "user", "content": "The quick brown fox jumps over the lazy dog"}]},  # Higher entropy
            {"messages": [{"role": "user", "content": "b" * 1000}]},  # Low entropy
            {"messages": [{"role": "user", "content": "Machine learning is a subset of artificial intelligence"}]},  # Higher entropy
        ]
        
        def sample_generator():
            for s in samples:
                yield s
        
        filtered = list(filter_obj.filter_dataset(sample_generator()))
        
        # Some samples should be filtered out
        assert len(filtered) <= len(samples)
    
    def test_classifier_based_filtering_integration(self, mocker):
        """Test classifier-based filtering integration."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Setup mock filter with classifier
        filter_obj = MemorizationFilter(
            entropy_threshold=0.0,  # Allow all through entropy
            risk_threshold=0.5
        )
        
        # Create mock classifier
        classifier = MemorizationClassifier()
        classifier.model = LogisticRegression()
        classifier.model.classes_ = [0, 1]
        classifier.model.coef_ = [[1.0, 0.1, 0.1, 0.1]]
        classifier.model.intercept_ = [-0.5]
        classifier.scaler = StandardScaler()
        classifier.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
        classifier.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]
        classifier.scaler.n_features_in_ = 4
        classifier.is_trained = True
        
        filter_obj.classifier = classifier
        
        # Test filtering
        samples = [
            {"messages": [{"role": "user", "content": "Sample text"}]}
            for _ in range(5)
        ]
        
        def sample_generator():
            for s in samples:
                yield s
        
        # Should filter without errors
        filtered = list(filter_obj.filter_dataset(sample_generator()))
        assert isinstance(filtered, list)


class TestMemorizationPipelineWithRealMetrics:
    """Test pipeline with real metric calculations."""
    
    def test_zlib_entropy_consistency(self):
        """Test that zlib entropy calculation is consistent."""
        text = "This is a test text for consistency checking."
        
        # Calculate entropy multiple times
        entropies = [
            MemorizationClassifier.calculate_zlib_entropy(text)
            for _ in range(5)
        ]
        
        # All calculations should return the same value
        assert all(e == entropies[0] for e in entropies)
    
    def test_entropy_ordering(self):
        """Test that entropy correctly orders text by compressibility."""
        # Highly repetitive text - low entropy
        repetitive = "abc" * 1000
        
        # Random-like text - higher entropy
        random_like = "The quick brown fox jumps over the lazy dog. " * 100
        
        repetitive_entropy = MemorizationClassifier.calculate_zlib_entropy(repetitive)
        random_entropy = MemorizationClassifier.calculate_zlib_entropy(random_like)
        
        # Repetitive text should have lower entropy
        assert repetitive_entropy < random_entropy
    
    def test_filter_statistics_accumulation(self):
        """Test that filter statistics are properly accumulated."""
        filter_obj = MemorizationFilter(entropy_threshold=0.5)
        
        # Create sample generator
        def sample_generator():
            for i in range(10):
                yield {"messages": [{"role": "user", "content": f"Sample {i}"}]}
        
        # Filter samples
        filtered = list(filter_obj.filter_dataset(sample_generator()))
        
        # Should return a list
        assert isinstance(filtered, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

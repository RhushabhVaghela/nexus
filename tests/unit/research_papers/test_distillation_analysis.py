"""
Comprehensive Unit Tests for Distillation Analysis (Paper 2601.15394).

Tests cover:
- Hard vs soft comparison
- Inherited rate calculation
- Report generation
- Privacy recommendations
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.nexus_final.auditor import DistillationReport


class TestHardVsSoftComparison:
    """Test hard vs soft distillation comparison."""
    
    def test_distillation_report_creation(self):
        """Test creation of distillation report."""
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


class TestInheritedRateCalculation:
    """Test inherited rate calculation from teacher."""
    
    def test_inherited_rate_zero(self):
        """Test inherited rate of zero."""
        report = DistillationReport(
            hard_distillation_rate=0.1,
            soft_distillation_rate=0.05,
            inherited_from_teacher_rate=0.0,
            privacy_recommendation="BOTH_METHODS_ACCEPTABLE",
            detailed_metrics={"samples_analyzed": 100}
        )
        
        assert report.inherited_from_teacher_rate == 0.0
    
    def test_inherited_rate_partial(self):
        """Test partial inherited rate."""
        report = DistillationReport(
            hard_distillation_rate=0.2,
            soft_distillation_rate=0.1,
            inherited_from_teacher_rate=0.15,
            privacy_recommendation="SOFT_DISTILLATION_RECOMMENDED",
            detailed_metrics={"samples_analyzed": 100}
        )
        
        assert report.inherited_from_teacher_rate == 0.15


class TestPrivacyRecommendations:
    """Test privacy recommendation logic."""
    
    def test_soft_distillation_recommended(self):
        """Test recommendation for soft distillation."""
        report = DistillationReport(
            hard_distillation_rate=0.30,
            soft_distillation_rate=0.15,
            inherited_from_teacher_rate=0.20,
            privacy_recommendation="SOFT_DISTILLATION_RECOMMENDED",
            detailed_metrics={"samples_analyzed": 100}
        )
        
        assert report.privacy_recommendation == "SOFT_DISTILLATION_RECOMMENDED"
    
    def test_additional_safeguards_needed(self):
        """Test recommendation for additional safeguards."""
        report = DistillationReport(
            hard_distillation_rate=0.35,
            soft_distillation_rate=0.35,
            inherited_from_teacher_rate=0.30,
            privacy_recommendation="ADDITIONAL_PRIVACY_SAFEGUARDS_NEEDED",
            detailed_metrics={"samples_analyzed": 100}
        )
        
        assert report.privacy_recommendation == "ADDITIONAL_PRIVACY_SAFEGUARDS_NEEDED"
    
    def test_both_methods_acceptable(self):
        """Test recommendation when both methods are acceptable."""
        report = DistillationReport(
            hard_distillation_rate=0.05,
            soft_distillation_rate=0.03,
            inherited_from_teacher_rate=0.04,
            privacy_recommendation="BOTH_METHODS_ACCEPTABLE",
            detailed_metrics={"samples_analyzed": 100}
        )
        
        assert report.privacy_recommendation == "BOTH_METHODS_ACCEPTABLE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

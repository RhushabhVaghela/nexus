"""
Unit tests for quality_metrics.py
"""

import pytest
from src.utils.quality_metrics import QualityMetrics, QualityScoringPipeline

class TestQualityMetrics:
    
    def test_syntax_validity_python(self):
        sample = {
            "messages": [
                {"role": "assistant", "content": "```python\ndef hello():\n    return 'world'\n```"}
            ]
        }
        score = QualityMetrics.syntax_validity(sample)
        assert score == 1.0

    def test_syntax_validity_invalid(self):
        sample = {
            "messages": [
                {"role": "assistant", "content": "```python\ndef hello( return 'world'\n```"} # Syntax error
            ]
        }
        # Should detect "def" and give 0.5 partial credit
        score = QualityMetrics.syntax_validity(sample)
        assert score == 0.5

    def test_content_length_score(self):
        # Good length (500-10000)
        sample = {"messages": [{"content": "a" * 1000}]}
        assert QualityMetrics.content_length_score(sample) == 1.0
        
        # Too short
        sample_short = {"messages": [{"content": "a" * 50}]}
        assert QualityMetrics.content_length_score(sample_short) == 0.3

    def test_complexity_score(self):
        # Simple code
        code = "console.log('hi')"
        sample = {"messages": [{"role": "assistant", "content": code}]}
        # < 2 complexity points -> 0.4
        assert QualityMetrics.complexity_score(sample) <= 0.6
        
        # Complex code
        complex_code = """
        function test() {
            if (true) {
                try {
                    items.map(i => i + 1);
                } catch (e) {}
            }
        }
        """
        sample_complex = {"messages": [{"role": "assistant", "content": complex_code}]}
        # functions(1) + if(1) + try(1) + map(1) = 4 -> 0.7 (Range 2-5)
        # Wait, metrics logic: 2 <= complexity < 5: return 0.7
        assert QualityMetrics.complexity_score(sample_complex) >= 0.7

    def test_pipeline_filtering(self):
        pipeline = QualityScoringPipeline(threshold=0.5)
        
        good_sample = {
            "messages": [{"role": "assistant", "content": "```python\ndef good(): pass\n```" + "a"*600}]
        }
        # Syntax 1.0, Length 1.0, Complexity ~0.4 -> Avg > 0.5
        
        bad_sample = {
            "messages": [{"role": "assistant", "content": "short"}]
        }
        # Syntax 0.5, Length 0.3, Complexity 0.4 -> Avg < 0.5
        
        batch = [good_sample, bad_sample]
        filtered = pipeline.filter_batch(batch)
        
        assert len(filtered) == 1
        assert filtered[0] == good_sample

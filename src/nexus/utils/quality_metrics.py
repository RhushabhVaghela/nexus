#!/usr/bin/env python3
"""
utils/quality_metrics.py
Shared quality evaluation for all generators.
Based on documentation recommendations.
"""

import ast
import re
import hashlib
from typing import Dict, List, Any, Optional
from collections import defaultdict
import math

# ═══════════════════════════════════════════════════════════════
# QUALITY METRICS CLASS
# ═══════════════════════════════════════════════════════════════

class QualityMetrics:
    """Shared quality evaluation for all generators."""
    
    @staticmethod
    def syntax_validity(sample: Dict) -> float:
        """Check if code in sample is syntactically valid. Returns 0-1 score."""
        messages = sample.get("messages", [])
        
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # Extract code blocks
                code_blocks = re.findall(r'```(?:python|javascript|typescript|dart)?\n(.*?)```', 
                                        content, re.DOTALL)
                
                if code_blocks:
                    valid = 0
                    for code in code_blocks:
                        try:
                            # Try parsing as Python (most common)
                            ast.parse(code)
                            valid += 1
                        except SyntaxError:
                            # Not Python or invalid
                            # Check for basic structure (functions, classes)
                            if re.search(r'(function|class|def|const|let|var)\s+\w+', code):
                                valid += 0.5
                    
                    return valid / len(code_blocks) if code_blocks else 0.5
        
        return 0.5  # Default for non-code samples
    
    @staticmethod
    def content_length_score(sample: Dict) -> float:
        """Score based on content length (too short or too long is bad)."""
        messages = sample.get("messages", [])
        
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        
        # Optimal range: 500-10000 characters
        if 500 <= total_length <= 10000:
            return 1.0
        elif 200 <= total_length < 500:
            return 0.7
        elif 10000 < total_length <= 20000:
            return 0.8
        elif total_length < 200:
            return 0.3
        else:  # > 20000
            return 0.5
    
    @staticmethod
    def diversity_score(sample: Dict, corpus_hashes: set) -> float:
        """Check novelty against existing corpus."""
        content = str(sample.get("messages", []))
        sample_hash = hashlib.md5(content.encode()).hexdigest()
        
        if sample_hash in corpus_hashes:
            return 0.0  # Duplicate
        
        # Check for near-duplicates (first 100 chars)
        short_hash = hashlib.md5(content[:100].encode()).hexdigest()
        if short_hash in corpus_hashes:
            return 0.5  # Near-duplicate
        
        return 1.0
    
    @staticmethod
    def complexity_score(sample: Dict) -> float:
        """Measure code complexity (moderate complexity is good)."""
        messages = sample.get("messages", [])
        
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # Count complexity indicators
                indicators = {
                    'functions': len(re.findall(r'(def |function |const \w+ = \()', content)),
                    'classes': len(re.findall(r'class \w+', content)),
                    'loops': len(re.findall(r'(for |while |\.map\(|\.forEach\()', content)),
                    'conditionals': len(re.findall(r'(if |elif |else:|switch |case )', content)),
                    'try_catch': len(re.findall(r'(try:|try \{|catch)', content)),
                }
                
                complexity = sum(indicators.values())
                
                # Optimal: 5-20 complexity points
                if 5 <= complexity <= 20:
                    return 1.0
                elif 2 <= complexity < 5:
                    return 0.7
                elif 20 < complexity <= 40:
                    return 0.8
                elif complexity < 2:
                    return 0.4
                else:
                    return 0.6
        
        return 0.5
    
    @staticmethod  
    def compute_composite_score(sample: Dict, corpus_hashes: set = None) -> float:
        """Compute weighted composite quality score."""
        scores = {
            'syntactic': QualityMetrics.syntax_validity(sample),
            'length': QualityMetrics.content_length_score(sample),
            'complexity': QualityMetrics.complexity_score(sample),
        }
        
        if corpus_hashes:
            scores['diversity'] = QualityMetrics.diversity_score(sample, corpus_hashes)
        
        # Weighted average
        weights = {
            'syntactic': 0.30,
            'length': 0.20,
            'complexity': 0.25,
            'diversity': 0.25 if corpus_hashes else 0,
        }
        
        total_weight = sum(weights.values())
        composite = sum(scores.get(k, 0) * v for k, v in weights.items()) / total_weight
        
        return composite


# ═══════════════════════════════════════════════════════════════
# QUALITY SCORING PIPELINE
# ═══════════════════════════════════════════════════════════════

class QualityScoringPipeline:
    """Multi-stage quality filtering pipeline."""
    
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.corpus_hashes = set()
        self.metrics = QualityMetrics()
    
    def score_sample(self, sample: Dict) -> float:
        """Score a single sample."""
        return QualityMetrics.compute_composite_score(sample, self.corpus_hashes)
    
    def filter_batch(self, batch: List[Dict]) -> List[Dict]:
        """Keep only samples above quality threshold."""
        scored = [(s, self.score_sample(s)) for s in batch]
        
        # Add accepted samples to corpus
        accepted = []
        for sample, score in scored:
            if score >= self.threshold:
                content = str(sample.get("messages", []))
                self.corpus_hashes.add(hashlib.md5(content.encode()).hexdigest())
                accepted.append(sample)
        
        return accepted
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "threshold": self.threshold,
            "corpus_size": len(self.corpus_hashes),
        }


# ═══════════════════════════════════════════════════════════════
# VERIFIED GENERATOR WRAPPER
# ═══════════════════════════════════════════════════════════════

class VerifiedGenerator:
    """Wrapper that adds verification to any generator."""
    
    def __init__(self, generator_func, max_attempts: int = 5):
        self.generator_func = generator_func
        self.max_attempts = max_attempts
        self.pipeline = QualityScoringPipeline(threshold=0.60)
    
    def generate_with_verification(self, *args, **kwargs) -> Optional[Dict]:
        """Generate with quality verification."""
        for attempt in range(self.max_attempts):
            sample = self.generator_func(*args, **kwargs)
            
            if sample is None:
                continue
            
            score = self.pipeline.score_sample(sample)
            
            if score >= self.pipeline.threshold:
                return sample
        
        return None  # Failed after max_attempts


# ═══════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality metrics for generated data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--threshold", type=float, default=0.75, help="Quality threshold")
    parser.add_argument("--output", help="Output filtered JSONL file")
    args = parser.parse_args()
    
    pipeline = QualityScoringPipeline(threshold=args.threshold)
    
    total = 0
    passed = 0
    
    with open(args.input, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    filtered = pipeline.filter_batch(samples)
    
    total = len(samples)
    passed = len(filtered)
    
    print(f"Total samples: {total}")
    print(f"Passed quality gate: {passed} ({passed/total*100:.1f}%)")
    print(f"Rejected: {total - passed} ({(total-passed)/total*100:.1f}%)")
    
    if args.output:
        with open(args.output, 'w') as f:
            for sample in filtered:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {passed} samples to {args.output}")

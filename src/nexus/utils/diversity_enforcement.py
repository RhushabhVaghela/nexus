#!/usr/bin/env python3
"""
utils/diversity_enforcement.py
Zipfian sampling and diversity enforcement for generators.
Based on documentation recommendations.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# DIVERSITY ENFORCEMENT CONFIG
# ═══════════════════════════════════════════════════════════════

DIVERSITY_ENFORCEMENT = {
    "min_unique_blueprints": 40,    # Minimum unique templates
    "template_reuse_limit": 5000,   # Max times to reuse same template
    "entropy_threshold": 0.85,      # Minimum entropy vs. real data
    "tail_coverage": 0.20           # Force 20% of samples from rare variants
}

# ═══════════════════════════════════════════════════════════════
# ZIPFIAN DISTRIBUTION (Realistic long-tail)
# ═══════════════════════════════════════════════════════════════

def create_zipfian_weights(blueprints: List[str], alpha: float = 1.0) -> Dict[str, float]:
    """
    Create Zipfian distribution weights for blueprints.
    
    Args:
        blueprints: List of blueprint names
        alpha: Zipf exponent (higher = more skewed)
    
    Returns:
        Dictionary of blueprint -> weight
    """
    n = len(blueprints)
    ranks = np.arange(1, n + 1)
    
    # Zipf's law: probability ~ 1/rank^alpha
    raw_weights = 1.0 / np.power(ranks, alpha)
    
    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()
    
    return dict(zip(blueprints, weights))


# Example weights for a typical generator
BLUEPRINT_WEIGHTS = {
    # Common (80% of samples)
    "Simple CRUD": 0.25,
    "Dashboard": 0.20,
    "Form App": 0.15,
    "Blog": 0.10,
    "Landing Page": 0.10,
    
    # Moderate (15%)
    "Real-time Chat": 0.05,
    "E-commerce": 0.04,
    "API Gateway": 0.03,
    "Video Streaming": 0.02,
    "Social Feed": 0.01,
    
    # Rare/Hard (5%) - CRITICAL FOR PREVENTING COLLAPSE
    "Blockchain Explorer": 0.01,
    "Multiplayer Game": 0.01,
    "WebRTC Conference": 0.01,
    "Compiler": 0.01,
    "ML Pipeline": 0.01
}


def select_blueprint_zipfian(weights: Dict[str, float] = None) -> str:
    """Sample with realistic frequency distribution."""
    if weights is None:
        weights = BLUEPRINT_WEIGHTS
    
    blueprints = list(weights.keys())
    probs = list(weights.values())
    
    # Normalize if needed
    total = sum(probs)
    probs = [p / total for p in probs]
    
    return np.random.choice(blueprints, p=probs)


# ═══════════════════════════════════════════════════════════════
# ENTROPY TRACKER
# ═══════════════════════════════════════════════════════════════

class EntropyTracker:
    """Track entropy of generated samples to detect collapse."""
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.template_counts = defaultdict(int)
        self.total_samples = 0
    
    def record(self, template: str):
        """Record a template usage."""
        self.template_counts[template] += 1
        self.total_samples += 1
    
    def compute_entropy(self) -> float:
        """Compute Shannon entropy of template distribution."""
        if self.total_samples == 0:
            return 1.0
        
        entropy = 0.0
        for count in self.template_counts.values():
            if count > 0:
                p = count / self.total_samples
                entropy -= p * np.log2(p)
        
        # Normalize to 0-1 (max entropy = log2(num_templates))
        max_entropy = np.log2(len(self.template_counts)) if self.template_counts else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def is_healthy(self, threshold: float = 0.85) -> bool:
        """Check if entropy is above threshold."""
        return self.compute_entropy() >= threshold


# ═══════════════════════════════════════════════════════════════
# ENHANCED GENERATOR WITH DIVERSITY
# ═══════════════════════════════════════════════════════════════

class DiversityEnforcedGenerator:
    """Generator wrapper that enforces diversity constraints."""
    
    def __init__(self, blueprint_weights: Dict[str, float] = None):
        self.blueprint_weights = blueprint_weights or BLUEPRINT_WEIGHTS
        self.template_usage = defaultdict(int)
        self.entropy_tracker = EntropyTracker()
        self.randomization_factor = 1.0
    
    def select_with_diversity_bias(self) -> str:
        """Select blueprint with inverse frequency bias."""
        # Apply inverse frequency weighting to underused templates
        adjusted_weights = {}
        
        for template, base_weight in self.blueprint_weights.items():
            usage = self.template_usage[template]
            reuse_limit = DIVERSITY_ENFORCEMENT["template_reuse_limit"]
            
            if usage >= reuse_limit:
                adjusted_weights[template] = 0  # Exhausted
            else:
                # Boost underused templates
                usage_ratio = usage / max(sum(self.template_usage.values()), 1)
                boost = max(0.1, 1 - usage_ratio * self.randomization_factor)
                adjusted_weights[template] = base_weight * boost
        
        # Fallback if all exhausted
        total = sum(adjusted_weights.values())
        if total == 0:
            # Reset and continue
            self.template_usage.clear()
            return random.choice(list(self.blueprint_weights.keys()))
        
        return select_blueprint_zipfian(adjusted_weights)
    
    def should_increase_randomization(self) -> bool:
        """Check if we need more diversity."""
        return self.entropy_tracker.compute_entropy() < DIVERSITY_ENFORCEMENT["entropy_threshold"]
    
    def increase_randomization(self):
        """Increase randomization to combat entropy decay."""
        self.randomization_factor = min(2.0, self.randomization_factor * 1.1)
    
    def record_generation(self, template: str):
        """Record a successful generation."""
        self.template_usage[template] += 1
        self.entropy_tracker.record(template)
        
        # Periodic entropy check
        if self.entropy_tracker.total_samples % 100000 == 0:
            if self.should_increase_randomization():
                self.increase_randomization()
    
    def get_stats(self) -> Dict:
        """Get diversity statistics."""
        return {
            "total_generated": sum(self.template_usage.values()),
            "unique_templates": len(self.template_usage),
            "entropy": self.entropy_tracker.compute_entropy(),
            "randomization_factor": self.randomization_factor,
            "template_usage": dict(self.template_usage)
        }


# ═══════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demo
    generator = DiversityEnforcedGenerator()
    
    print("Simulating 10000 generations...")
    for i in range(10000):
        template = generator.select_with_diversity_bias()
        generator.record_generation(template)
    
    stats = generator.get_stats()
    print(f"\nDiversity Stats:")
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Unique templates: {stats['unique_templates']}")
    print(f"  Entropy: {stats['entropy']:.3f}")
    print(f"  Randomization: {stats['randomization_factor']:.2f}")
    print(f"\nTop 5 templates:")
    sorted_usage = sorted(stats['template_usage'].items(), key=lambda x: -x[1])
    for template, count in sorted_usage[:5]:
        print(f"  {template}: {count}")

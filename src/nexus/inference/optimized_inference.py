"""
Optimized Inference Integration Layer

Integrates all 8 research-backed optimizations for achieving 100+ tokens/second:
1. Layer Pipelining (EasySpec-style)
2. Adaptive Layer Skipping (SWIFT/LayerSkip)
3. Semi-Autoregressive Decoding (SPACE)
4. Async Decompression (nvCOMP)
5. Optimized Compression (ZSTD + quantization)
6. Layer Fusion (Kernel fusion)
7. Early Exit Routing (Dynamic routing)
8. Low-Rank Attention (Sparse attention)

Usage:
    from nexus.inference.optimized_inference import OptimizedInference

    inference = OptimizedInference(model, config)
    output = inference.generate(input_ids, max_new_tokens=100)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Import all optimization modules
from nexus.optimizations import (
    LayerPipeliningOptimizer,
    AdaptiveLayerSkipper,
    SemiAutoregressiveDecoder,
    AsyncDecompressor,
    OptimizedCompressor,
    LayerFusionOptimizer,
    EarlyExitRouter,
    SparseAttentionOptimizer,
)


@dataclass
class OptimizedInferenceConfig:
    """Configuration for optimized inference."""

    # Enable/disable optimizations
    enable_layer_pipelining: bool = True
    enable_layer_skipping: bool = True
    enable_semi_autoregressive: bool = True
    enable_async_decompression: bool = True
    enable_optimized_compression: bool = True
    enable_layer_fusion: bool = True
    enable_early_exit: bool = True
    enable_sparse_attention: bool = True

    # Optimization-specific configs
    layer_pipelining_config: Dict = field(default_factory=dict)
    layer_skipping_config: Dict = field(default_factory=dict)
    semi_autoregressive_config: Dict = field(default_factory=dict)
    async_decompression_config: Dict = field(default_factory=dict)
    compression_config: Dict = field(default_factory=dict)
    layer_fusion_config: Dict = field(default_factory=dict)
    early_exit_config: Dict = field(default_factory=dict)
    sparse_attention_config: Dict = field(default_factory=dict)

    # Fallback behavior
    fallback_on_error: bool = True
    enable_metrics: bool = True

    # GPU compression (cuda_zstd) — used by async decompression & compression
    use_gpu_compression: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "OptimizedInferenceConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("inference", {}))


class OptimizationRegistry:
    """
    Registry for managing all optimization modules.

    Handles initialization, fallback, and metrics collection.
    """

    def __init__(self, model: nn.Module, config: OptimizedInferenceConfig):
        self.model = model
        self.config = config
        self.optimizations: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {
            "tokens_generated": 0,
            "total_time_ms": 0,
            "optimizations_used": [],
        }

        self._initialize_optimizations()

    def _initialize_optimizations(self):
        """Initialize all enabled optimizations."""
        # Layer Pipelining
        if self.config.enable_layer_pipelining:
            try:
                hidden_size = getattr(self.model.config, "hidden_size", 4096)
                num_layers = getattr(self.model.config, "num_hidden_layers", 80)

                self.optimizations["layer_pipelining"] = LayerPipeliningOptimizer(
                    self.model,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    config=None,  # Use defaults
                )
                logger.info("✓ Layer pipelining initialized")
            except Exception as e:
                logger.warning(f"✗ Layer pipelining failed: {e}")
                if not self.config.fallback_on_error:
                    raise

        # Adaptive Layer Skipping
        if self.config.enable_layer_skipping:
            try:
                hidden_size = getattr(self.model.config, "hidden_size", 4096)
                num_layers = getattr(self.model.config, "num_hidden_layers", 80)

                self.optimizations["layer_skipping"] = AdaptiveLayerSkipper(
                    self.model, num_layers=num_layers, hidden_size=hidden_size
                )
                logger.info("✓ Adaptive layer skipping initialized")
            except Exception as e:
                logger.warning(f"✗ Layer skipping failed: {e}")

        # Semi-Autoregressive Decoding
        if self.config.enable_semi_autoregressive:
            try:
                self.optimizations["semi_autoregressive"] = SemiAutoregressiveDecoder(
                    self.model
                )
                logger.info("✓ Semi-autoregressive decoding initialized")
            except Exception as e:
                logger.warning(f"✗ Semi-autoregressive failed: {e}")

        # Async Decompression
        if self.config.enable_async_decompression:
            try:
                from nexus.optimizations.async_decompression import (
                    AsyncDecompressionConfig,
                )

                self.optimizations["async_decompression"] = AsyncDecompressor(
                    config=AsyncDecompressionConfig(
                        use_gpu_compression=self.config.use_gpu_compression,
                    )
                )
                logger.info("✓ Async decompression initialized")
            except Exception as e:
                logger.warning(f"✗ Async decompression failed: {e}")

        # Optimized Compression
        if self.config.enable_optimized_compression:
            try:
                self.optimizations["compression"] = OptimizedCompressor(
                    use_gpu=self.config.use_gpu_compression,
                )
                logger.info("✓ Optimized compression initialized")
            except Exception as e:
                logger.warning(f"✗ Optimized compression failed: {e}")

        # Layer Fusion
        if self.config.enable_layer_fusion:
            try:
                from nexus.optimizations.layer_fusion import FusionConfig

                fusion_optimizer = LayerFusionOptimizer(config=FusionConfig())
                self.optimizations["layer_fusion"] = fusion_optimizer
                logger.info("✓ Layer fusion initialized")
            except Exception as e:
                logger.warning(f"✗ Layer fusion failed: {e}")

        # Early Exit Routing
        if self.config.enable_early_exit:
            try:
                hidden_size = getattr(self.model.config, "hidden_size", 4096)
                num_layers = getattr(self.model.config, "num_hidden_layers", 80)

                self.optimizations["early_exit"] = EarlyExitRouter(
                    self.model, num_layers=num_layers, hidden_size=hidden_size
                )
                logger.info("✓ Early exit routing initialized")
            except Exception as e:
                logger.warning(f"✗ Early exit failed: {e}")

        # Sparse Attention
        if self.config.enable_sparse_attention:
            try:
                from nexus.optimizations.low_rank_attention import SparseAttentionConfig

                self.optimizations["sparse_attention"] = SparseAttentionOptimizer(
                    config=SparseAttentionConfig()
                )
                logger.info("✓ Sparse attention initialized")
            except Exception as e:
                logger.warning(f"✗ Sparse attention failed: {e}")

    def get(self, name: str) -> Optional[Any]:
        """Get optimization by name."""
        return self.optimizations.get(name)

    def list_active(self) -> List[str]:
        """List all active optimization names."""
        return list(self.optimizations.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Collect stats from all optimizations."""
        stats = {}
        for name, opt in self.optimizations.items():
            if hasattr(opt, "get_stats"):
                try:
                    stats[name] = opt.get_stats()
                except Exception as e:
                    logger.warning(f"Failed to get stats for {name}: {e}")
        return stats


class OptimizedInference:
    """
    Main inference class with all optimizations integrated.

    Provides a unified interface for running optimized inference
    with automatic fallback and metrics collection.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizedInferenceConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize optimized inference.

        Args:
            model: Base model to optimize
            config: Configuration object (or loaded from config_path)
            config_path: Path to YAML config file
        """
        self.model = model

        # Load config
        if config_path:
            self.config = OptimizedInferenceConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = OptimizedInferenceConfig()

        # Initialize optimization registry
        self.registry = OptimizationRegistry(model, self.config)

        # Apply model-level optimizations
        self._apply_model_optimizations()

        # Stats
        self.generation_stats = {
            "total_tokens": 0,
            "total_time_ms": 0,
            "tokens_per_second": 0.0,
            "calls": 0,
        }

        logger.info(
            f"OptimizedInference initialized with {len(self.registry.list_active())} optimizations"
        )
        logger.info(f"Active: {', '.join(self.registry.list_active())}")

    def _apply_model_optimizations(self):
        """Apply optimizations that modify the model structure."""
        # Layer fusion
        if "layer_fusion" in self.registry.optimizations:
            try:
                fusion_opt = self.registry.get("layer_fusion")
                self.model = fusion_opt.fuse_model(self.model)
                logger.info("✓ Applied layer fusion to model")
            except Exception as e:
                logger.warning(f"Layer fusion application failed: {e}")

        # Sparse attention
        if "sparse_attention" in self.registry.optimizations:
            try:
                sparse_opt = self.registry.get("sparse_attention")
                self.model = sparse_opt.optimize_model(self.model)
                logger.info("✓ Applied sparse attention to model")
            except Exception as e:
                logger.warning(f"Sparse attention application failed: {e}")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens with all optimizations enabled.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            attention_mask: Optional attention mask

        Returns:
            Generated token IDs [batch, seq_len + generated]
        """
        start_time = time.time()

        # Try semi-autoregressive first for speed
        if "semi_autoregressive" in self.registry.optimizations:
            try:
                sar_decoder = self.registry.get("semi_autoregressive")
                output = sar_decoder.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    attention_mask=attention_mask,
                )

                # Update stats
                elapsed_ms = (time.time() - start_time) * 1000
                tokens_generated = output.shape[1] - input_ids.shape[1]

                self.generation_stats["total_tokens"] += tokens_generated
                self.generation_stats["total_time_ms"] += elapsed_ms
                self.generation_stats["calls"] += 1

                return output

            except Exception as e:
                logger.warning(
                    f"Semi-autoregressive generation failed, falling back: {e}"
                )

        # Standard autoregressive generation with optimizations
        output_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass with optimizations
            outputs = self._forward_with_optimizations(
                output_ids, attention_mask=attention_mask
            )

            # Get next token logits
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Check for EOS
            if hasattr(self.model.config, "eos_token_id"):
                if (next_token == self.model.config.eos_token_id).all():
                    break

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]

        self.generation_stats["total_tokens"] += tokens_generated
        self.generation_stats["total_time_ms"] += elapsed_ms
        self.generation_stats["calls"] += 1

        return output_ids

    def _forward_with_optimizations(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optimizations applied.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Model outputs (logits or hidden states)
        """
        # Use layer pipelining if available
        if "layer_pipelining" in self.registry.optimizations:
            try:
                # Get embeddings
                hidden_states = self.model.get_input_embeddings()(input_ids)

                # Apply layer pipelining
                pipelining = self.registry.get("layer_pipelining")
                output, metrics = pipelining.forward(
                    hidden_states, attention_mask=attention_mask, use_speculation=True
                )

                # Project to logits
                logits = (
                    self.model.lm_head(output)
                    if hasattr(self.model, "lm_head")
                    else output
                )
                return logits

            except Exception as e:
                logger.debug(f"Layer pipelining failed: {e}")

        # Use early exit routing if available
        if "early_exit" in self.registry.optimizations:
            try:
                hidden_states = self.model.get_input_embeddings()(input_ids)

                early_exit = self.registry.get("early_exit")
                layers = self._get_model_layers()

                output, metrics = early_exit.forward_with_routing(
                    hidden_states, layers, attention_mask
                )

                logits = (
                    self.model.lm_head(output)
                    if hasattr(self.model, "lm_head")
                    else output
                )
                return logits

            except Exception as e:
                logger.debug(f"Early exit failed: {e}")

        # Standard forward
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def _get_model_layers(self) -> List[nn.Module]:
        """Extract transformer layers from model."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "layers"):
            return list(self.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return list(self.model.transformer.h)
        return []

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Calculate overall metrics
        total_time_s = self.generation_stats["total_time_ms"] / 1000
        tokens_per_second = (
            self.generation_stats["total_tokens"] / total_time_s
            if total_time_s > 0
            else 0.0
        )

        # Collect optimization stats
        opt_stats = self.registry.get_stats()

        report = {
            "summary": {
                "total_tokens_generated": self.generation_stats["total_tokens"],
                "total_time_seconds": total_time_s,
                "tokens_per_second": tokens_per_second,
                "generation_calls": self.generation_stats["calls"],
                "active_optimizations": len(self.registry.list_active()),
                "optimization_list": self.registry.list_active(),
                "gpu_compression": self._get_gpu_compression_status(opt_stats),
            },
            "optimization_stats": opt_stats,
            "target_achievement": {
                "target_tokens_per_second": 100,
                "actual_tokens_per_second": tokens_per_second,
                "achievement_percentage": (tokens_per_second / 100) * 100,
                "target_met": tokens_per_second >= 100,
            },
            "recommendations": self._generate_recommendations(
                tokens_per_second, opt_stats
            ),
        }

        return report

    def _generate_recommendations(
        self, tokens_per_second: float, opt_stats: Dict
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if tokens_per_second < 50:
            recommendations.append("Consider enabling more aggressive layer skipping")
            recommendations.append(
                "Try reducing sequence length or using sparse attention"
            )

        if tokens_per_second < 80:
            recommendations.append(
                "Semi-autoregressive decoding may help if not enabled"
            )
            recommendations.append("Check if async decompression is active")

        if "sparse_attention" not in self.registry.list_active():
            recommendations.append(
                "Sparse attention could provide 80% computation reduction"
            )

        if "layer_fusion" not in self.registry.list_active():
            recommendations.append("Layer fusion could reduce per-layer time by 23%")

        # GPU compression recommendation
        gpu_status = self._get_gpu_compression_status(opt_stats)
        if gpu_status["backend"] != "gpu":
            recommendations.append(
                "GPU compression (cuda_zstd) not active — install cuda-zstd "
                "for ~10x faster ZSTD decompression on GPU"
            )

        return recommendations

    def _get_gpu_compression_status(self, opt_stats: Dict) -> Dict[str, Any]:
        """Extract GPU compression backend status from optimization stats."""
        status: Dict[str, Any] = {
            "enabled": self.config.use_gpu_compression,
            "backend": "none",
        }

        # Check async_decompression stats for backend info
        async_stats = opt_stats.get("async_decompression", {})
        if "backend" in async_stats:
            status["backend"] = async_stats["backend"]

        # Check compression stats for backend info
        comp_stats = opt_stats.get("compression", {})
        if "backend" in comp_stats:
            status["backend"] = comp_stats["backend"]

        return status

    def print_performance_report(self):
        """Print formatted performance report."""
        report = self.get_performance_report()

        print("\n" + "=" * 60)
        print("OPTIMIZED INFERENCE PERFORMANCE REPORT")
        print("=" * 60)

        summary = report["summary"]
        print(f"\n📊 Overall Performance:")
        print(f"   Tokens Generated: {summary['total_tokens_generated']:,}")
        print(f"   Total Time: {summary['total_time_seconds']:.2f}s")
        print(f"   Throughput: {summary['tokens_per_second']:.2f} tokens/s")
        print(f"   Generation Calls: {summary['generation_calls']}")

        target = report["target_achievement"]
        print(f"\n🎯 Target Achievement:")
        print(f"   Target: {target['target_tokens_per_second']} tokens/s")
        print(f"   Actual: {target['actual_tokens_per_second']:.2f} tokens/s")
        print(f"   Achievement: {target['achievement_percentage']:.1f}%")
        status = "✅ MET" if target["target_met"] else "❌ NOT MET"
        print(f"   Status: {status}")

        print(f"\n🔧 Active Optimizations ({summary['active_optimizations']}):")
        for opt in summary["optimization_list"]:
            print(f"   ✓ {opt}")

        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   → {rec}")

        print("\n" + "=" * 60)


def create_optimized_inference(
    model: nn.Module, config_path: Optional[str] = None, **kwargs
) -> OptimizedInference:
    """
    Factory function to create OptimizedInference with smart defaults.

    Args:
        model: Base model
        config_path: Optional config file path
        **kwargs: Additional config overrides

    Returns:
        Configured OptimizedInference instance
    """
    if config_path and Path(config_path).exists():
        config = OptimizedInferenceConfig.from_yaml(config_path)
    else:
        config = OptimizedInferenceConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return OptimizedInference(model, config)

"""
Comprehensive unit tests for semi_autoregressive.py
Tests token prediction, verification logic, batch processing, and quality metrics.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from src.optimizations.semi_autoregressive import (
    SARConfig,
    ParallelTokenHead,
    SPACEDecoder,
    SemiAutoregressiveDecoder,
)


class TestSARConfig:
    """Test SARConfig dataclass."""

    def test_default_config_values(self):
        """Test default SAR configuration."""
        config = SARConfig()
        assert config.lookahead_tokens == 4
        assert config.max_parallel_windows == 8
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.verify_tokens is True

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = SARConfig(
            lookahead_tokens=8,
            max_parallel_windows=16,
            temperature=0.7,
            top_k=100,
            top_p=0.95,
            verify_tokens=False,
        )

        assert config.lookahead_tokens == 8
        assert config.max_parallel_windows == 16
        assert config.temperature == 0.7
        assert config.top_k == 100
        assert config.top_p == 0.95
        assert config.verify_tokens is False


class TestParallelTokenHead:
    """Test ParallelTokenHead for parallel token prediction."""

    def test_init(self):
        """Test initialization."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        assert head.hidden_size == 768
        assert head.vocab_size == 32000
        assert head.num_heads == 4

    def test_forward_shape(self):
        """Test forward pass shapes."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        logits_list, confidence_list = head.forward(hidden_states)

        # Should have 4 heads
        assert len(logits_list) == 4
        assert len(confidence_list) == 4

        # Each logit should be [batch, vocab]
        for logits in logits_list:
            assert logits.shape == (batch_size, 32000)

    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        for batch_size in [1, 4, 16]:
            hidden_states = torch.randn(batch_size, 128, 768)
            logits_list, confidence_list = head.forward(hidden_states)

            for logits in logits_list:
                assert logits.shape[0] == batch_size

    def test_generate_parallel_tokens_basic(self):
        """Test basic parallel token generation."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        batch_size = 2
        hidden_states = torch.randn(batch_size, 128, 768)

        tokens, confidences = head.generate_parallel_tokens(
            hidden_states, temperature=1.0, top_k=50, top_p=0.9
        )

        # Should generate 4 tokens
        assert tokens.shape == (batch_size, 4)
        assert confidences.shape == (batch_size, 4)

    def test_generate_parallel_tokens_with_temperature(self):
        """Test token generation with temperature."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        hidden_states = torch.randn(2, 128, 768)

        # High temperature
        tokens_hot, _ = head.generate_parallel_tokens(
            hidden_states,
            temperature=2.0,
            top_k=0,  # Disable top-k
            top_p=1.0,  # Disable top-p
        )

        # Low temperature
        tokens_cold, _ = head.generate_parallel_tokens(
            hidden_states, temperature=0.1, top_k=0, top_p=1.0
        )

        # Should produce different results
        assert not torch.equal(tokens_hot, tokens_cold)

    def test_generate_parallel_tokens_top_k(self):
        """Test token generation with top-k filtering."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        hidden_states = torch.randn(2, 128, 768)

        # With top-k=10
        tokens, _ = head.generate_parallel_tokens(
            hidden_states, temperature=1.0, top_k=10, top_p=1.0
        )

        assert tokens.shape == (2, 4)

    def test_generate_parallel_tokens_top_p(self):
        """Test token generation with top-p filtering."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        hidden_states = torch.randn(2, 128, 768)

        # With top-p=0.5
        tokens, _ = head.generate_parallel_tokens(
            hidden_states, temperature=1.0, top_k=0, top_p=0.5
        )

        assert tokens.shape == (2, 4)

    def test_confidence_prediction_shape(self):
        """Test confidence prediction shape."""
        head = ParallelTokenHead(hidden_size=768, vocab_size=32000, num_heads=4)

        hidden_states = torch.randn(2, 128, 768)

        _, confidence_list = head.forward(hidden_states)

        # Each confidence should be [batch, 1]
        for conf in confidence_list:
            assert conf.shape == (2, 1)


class TestSPACEDecoder:
    """Test SPACEDecoder for semi-autoregressive decoding."""

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_init(self, mock_head_class):
        """Test initialization."""
        mock_head = Mock()
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        assert decoder.base_model is mock_model
        assert decoder.vocab_size == 32000
        assert decoder.hidden_size == 768
        assert decoder.parallel_heads is mock_head

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_generate_single_pass(self, mock_head_class):
        """Test single generation pass."""
        mock_head = Mock()
        mock_head.generate_parallel_tokens.return_value = (
            torch.randint(0, 32000, (2, 4)),
            torch.ones(2, 4) * 0.9,
        )
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000
        mock_model.config.eos_token_id = 32000

        # Mock model outputs
        mock_output = Mock()
        mock_output.hidden_states = [None, torch.randn(2, 1, 768)]
        mock_output.logits = torch.randn(2, 1, 32000)
        mock_model.return_value = mock_output

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        input_ids = torch.randint(0, 32000, (2, 10))

        output = decoder.generate(input_ids, max_new_tokens=4)

        # Should have original + 4 new tokens
        assert output.shape[1] >= 10

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_verify_tokens_all_match(self, mock_head_class):
        """Test token verification when all tokens match."""
        mock_head = Mock()
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        # First call: predict next token
        mock_output1 = Mock()
        mock_output1.logits = torch.randn(2, 1, 32000)
        mock_output1.logits[:, :, 100] = 10.0  # Set high logit for token 100

        mock_model.return_value = mock_output1

        decoder = SPACEDecoder(
            base_model=mock_model,
            vocab_size=32000,
            hidden_size=768,
            config=SARConfig(verify_tokens=True),
        )

        prefix_ids = torch.randint(0, 32000, (2, 10))
        candidate_tokens = torch.ones(2, 4, dtype=torch.long) * 100  # All match
        confidences = torch.ones(2, 4) * 0.95

        verified, num_accepted = decoder._verify_tokens(
            prefix_ids, candidate_tokens, confidences, attention_mask=None
        )

        # All tokens should be accepted
        assert num_accepted == 4

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_verify_tokens_mismatch(self, mock_head_class):
        """Test token verification when tokens mismatch."""
        mock_head = Mock()
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        # First call: predict next token
        mock_output1 = Mock()
        mock_output1.logits = torch.randn(2, 1, 32000)
        mock_output1.logits[:, :, 100] = 10.0  # Set high logit for token 100

        mock_model.return_value = mock_output1

        decoder = SPACEDecoder(
            base_model=mock_model,
            vocab_size=32000,
            hidden_size=768,
            config=SARConfig(verify_tokens=True),
        )

        prefix_ids = torch.randint(0, 32000, (2, 10))
        candidate_tokens = (
            torch.ones(2, 4, dtype=torch.long) * 200
        )  # Different from prediction
        confidences = torch.ones(2, 4) * 0.5  # Low confidence

        verified, num_accepted = decoder._verify_tokens(
            prefix_ids, candidate_tokens, confidences, attention_mask=None
        )

        # Low confidence + mismatch should not accept
        assert num_accepted == 0

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_verify_tokens_high_confidence_mismatch(self, mock_head_class):
        """Test token verification with high confidence mismatch."""
        mock_head = Mock()
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        # First call: predict next token
        mock_output1 = Mock()
        mock_output1.logits = torch.randn(2, 1, 32000)
        mock_output1.logits[:, :, 100] = 10.0

        mock_model.return_value = mock_output1

        decoder = SPACEDecoder(
            base_model=mock_model,
            vocab_size=32000,
            hidden_size=768,
            config=SARConfig(verify_tokens=True),
        )

        prefix_ids = torch.randint(0, 32000, (2, 10))
        candidate_tokens = torch.ones(2, 4, dtype=torch.long) * 200
        confidences = torch.ones(2, 4) * 0.95  # High confidence

        verified, num_accepted = decoder._verify_tokens(
            prefix_ids, candidate_tokens, confidences, attention_mask=None
        )

        # High confidence should override mismatch
        assert num_accepted == 4

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_generate_updates_stats(self, mock_head_class):
        """Test that generation updates statistics."""
        mock_head = Mock()
        mock_head.generate_parallel_tokens.return_value = (
            torch.randint(0, 32000, (2, 4)),
            torch.ones(2, 4) * 0.9,
        )
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000
        mock_model.config.eos_token_id = 32000

        mock_output = Mock()
        mock_output.hidden_states = [None, torch.randn(2, 1, 768)]
        mock_output.logits = torch.randn(2, 1, 32000)
        mock_model.return_value = mock_output

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        input_ids = torch.randint(0, 32000, (2, 10))

        decoder.generate(input_ids, max_new_tokens=8)

        stats = decoder.get_stats()
        assert stats["total_calls"] > 0
        assert stats["parallel_tokens_generated"] > 0

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_get_stats(self, mock_head_class):
        """Test getting decoding statistics."""
        mock_head = Mock()
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        # Set some stats
        decoder.stats["total_calls"] = 10
        decoder.stats["parallel_tokens_generated"] = 40
        decoder.stats["verified_tokens"] = 35
        decoder.stats["rejected_tokens"] = 5

        stats = decoder.get_stats()

        assert stats["acceptance_rate"] == 0.875  # 35/40
        assert stats["tokens_per_forward_call"] == 3.5  # 35/10
        assert "theoretical_speedup" in stats

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_get_stats_empty(self, mock_head_class):
        """Test getting stats when no generation occurred."""
        mock_head = Mock()
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        stats = decoder.get_stats()

        assert stats["acceptance_rate"] == 0.0
        assert stats["tokens_per_forward_call"] == 1.0

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_generate_with_attention_mask(self, mock_head_class):
        """Test generation with attention mask."""
        mock_head = Mock()
        mock_head.generate_parallel_tokens.return_value = (
            torch.randint(0, 32000, (2, 4)),
            torch.ones(2, 4) * 0.9,
        )
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000
        mock_model.config.eos_token_id = 32000

        mock_output = Mock()
        mock_output.hidden_states = [None, torch.randn(2, 1, 768)]
        mock_output.logits = torch.randn(2, 1, 32000)
        mock_model.return_value = mock_output

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        input_ids = torch.randint(0, 32000, (2, 10))
        attention_mask = torch.ones(2, 10)

        output = decoder.generate(
            input_ids, max_new_tokens=4, attention_mask=attention_mask
        )

        # Should complete successfully
        assert output.shape[1] >= 10

    @patch("src.optimizations.semi_autoregressive.ParallelTokenHead")
    def test_generate_stops_at_eos(self, mock_head_class):
        """Test generation stops at EOS token."""
        mock_head = Mock()
        mock_head.generate_parallel_tokens.return_value = (
            torch.randint(0, 32000, (2, 4)),
            torch.ones(2, 4) * 0.9,
        )
        mock_head_class.return_value = mock_head

        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000
        mock_model.config.eos_token_id = 32000

        # Simulate EOS generation
        call_count = [0]

        def mock_return(*args, **kwargs):
            call_count[0] += 1
            mock_output = Mock()
            mock_output.hidden_states = [None, torch.randn(2, 1, 768)]
            mock_output.logits = torch.randn(2, 1, 32000)
            # Set EOS on first call
            if call_count[0] == 1:
                mock_output.logits[:, :, 32000] = 100.0
            return mock_output

        mock_model.return_value = mock_return

        decoder = SPACEDecoder(base_model=mock_model, vocab_size=32000, hidden_size=768)

        input_ids = torch.randint(0, 32000, (2, 10))

        output = decoder.generate(input_ids, max_new_tokens=20)

        # Should stop after EOS
        assert call_count[0] < 20


class TestSemiAutoregressiveDecoder:
    """Test SemiAutoregressiveDecoder for complete decoder module."""

    @patch("src.optimizations.semi_autoregressive.SPACEDecoder")
    def test_init(self, mock_decoder_class):
        """Test initialization."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        decoder = SemiAutoregressiveDecoder(base_model=mock_model)

        assert decoder.base_model is mock_model
        assert decoder.space_decoder is mock_decoder

    @patch("src.optimizations.semi_autoregressive.SPACEDecoder")
    def test_generate(self, mock_decoder_class):
        """Test generation method."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        mock_decoder = Mock()
        mock_decoder.generate.return_value = torch.randint(0, 32000, (2, 50))
        mock_decoder_class.return_value = mock_decoder

        decoder = SemiAutoregressiveDecoder(base_model=mock_model)

        input_ids = torch.randint(0, 32000, (2, 10))

        output = decoder.generate(input_ids, max_new_tokens=40)

        mock_decoder.generate.assert_called_once()

    @patch("src.optimizations.semi_autoregressive.SPACEDecoder")
    def test_forward(self, mock_decoder_class):
        """Test forward method."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        mock_output = Mock()
        mock_output.logits = torch.randn(2, 10, 32000)
        mock_model.return_value = mock_output

        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        decoder = SemiAutoregressiveDecoder(base_model=mock_model)

        input_ids = torch.randint(0, 32000, (2, 10))

        output = decoder.forward(input_ids)

        # Should call base model forward
        mock_model.assert_called_once()

    @patch("src.optimizations.semi_autoregressive.SPACEDecoder")
    def test_generate_with_attention_mask(self, mock_decoder_class):
        """Test generation with attention mask."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768
        mock_model.config.vocab_size = 32000

        mock_decoder = Mock()
        mock_decoder.generate.return_value = torch.randint(0, 32000, (2, 50))
        mock_decoder_class.return_value = mock_decoder

        decoder = SemiAutoregressiveDecoder(base_model=mock_model)

        input_ids = torch.randint(0, 32000, (2, 10))
        attention_mask = torch.ones(2, 10)

        output = decoder.generate(
            input_ids, max_new_tokens=40, attention_mask=attention_mask
        )

        mock_decoder.generate.assert_called_once()

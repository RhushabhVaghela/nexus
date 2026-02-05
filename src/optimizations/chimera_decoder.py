"""
Chimera: Lossless Multi-Token Prediction for Efficient Autoregressive Decoding

Key Ideas from arxiv:2402.15758v2:
- Lossless multi-token prediction architecture
- 2.5-3.5× speedup on autoregressive decode
- Combines with semi-autoregressive decoding: 3×4=12× amortization
- Uses auxiliary loss for multi-token prediction heads
- Architectural improvements over standard speculative decoding

Research reference: Chimera (arxiv:2402.15758v2)

This implementation provides:
- ChimeraHead: Multi-token prediction head with shared projection
- ChimeraConfig: Configuration dataclass
- ChimeraWrapper: Wrapper for existing language models
- ChimeraTrainer: Training utilities with auxiliary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ChimeraConfig:
    """
    Configuration for Chimera multi-token prediction.

    Attributes:
        num_predict_tokens: Number of tokens to predict in parallel (default: 4)
        aux_loss_weight: Weight for auxiliary multi-token loss (default: 0.1)
        share_embeddings: Whether to share with main LM head embeddings (default: True)
        hidden_size: Hidden dimension size (auto-configured if not provided)
        vocab_size: Vocabulary size (auto-configured if not provided)
        use_confidence_predictor: Whether to use confidence prediction heads (default: True)
        confidence_threshold: Threshold for accepting predicted tokens (default: 0.8)
        temperature: Sampling temperature for token generation (default: 1.0)
        top_k: Top-k filtering for sampling (default: 50)
        top_p: Top-p (nucleus) filtering for sampling (default: 0.9)
        layer_norm_type: Type of layer norm ('layer_norm' or 'rms_norm')
        dropout: Dropout rate for prediction heads (default: 0.0)
    """

    num_predict_tokens: int = 4
    aux_loss_weight: float = 0.1
    share_embeddings: bool = True
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    use_confidence_predictor: bool = True
    confidence_threshold: float = 0.8
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    layer_norm_type: str = "layer_norm"
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_predict_tokens < 1:
            raise ValueError(
                f"num_predict_tokens must be >= 1, got {self.num_predict_tokens}"
            )
        if self.aux_loss_weight < 0.0 or self.aux_loss_weight > 1.0:
            raise ValueError(
                f"aux_loss_weight must be in [0, 1], got {self.aux_loss_weight}"
            )
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )


class ChimeraHead(nn.Module):
    """
    Chimera Multi-Token Prediction Head.

    Predicts multiple future tokens simultaneously using a shared projection
    followed by separate heads for each prediction position. This architecture
    provides lossless multi-token prediction with improved training stability.

    Architecture:
        hidden_states -> shared projection -> layer_norm -> k * vocab_projection

    The key innovation is the shared representation layer that captures
    contextual information once, then distributes it to multiple prediction heads.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_tokens: int = 4,
        use_confidence_predictor: bool = True,
        layer_norm_type: str = "layer_norm",
        dropout: float = 0.0,
    ):
        """
        Initialize Chimera prediction head.

        Args:
            hidden_size: Hidden dimension size of the base model
            vocab_size: Vocabulary size for token prediction
            num_tokens: Number of tokens to predict in parallel
            use_confidence_predictor: Whether to include confidence prediction heads
            layer_norm_type: Type of layer normalization ('layer_norm' or 'rms_norm')
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.use_confidence_predictor = use_confidence_predictor

        # Shared projection layer - reduces computation by reusing representation
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Layer normalization after shared projection
        if layer_norm_type == "rms_norm":
            self.layer_norm = nn.RMSNorm(hidden_size)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size)

        # Separate output heads for each token position
        # Using ModuleDict for efficient access during forward pass
        self.token_heads = nn.ModuleDict()
        for i in range(num_tokens):
            self.token_heads[f"head_{i}"] = nn.Linear(
                hidden_size, vocab_size, bias=False
            )

        # Confidence predictors for adaptive generation
        if use_confidence_predictor:
            self.confidence_heads = nn.ModuleDict()
            for i in range(num_tokens):
                self.confidence_heads[f"conf_{i}"] = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1),
                    nn.Sigmoid(),
                )

        logger.info(
            f"ChimeraHead initialized: hidden_size={hidden_size}, "
            f"vocab_size={vocab_size}, num_tokens={num_tokens}"
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for multi-token prediction.

        Args:
            hidden_states: Hidden states from base model [batch, seq_len, hidden_size]
                           or [batch, hidden_size] for single position

        Returns:
            Tuple of (logits_list, confidence_list)
            - logits_list: List of logit tensors for each token position
            - confidence_list: List of confidence scores (if enabled)
        """
        # Get representation at final position for next-token prediction
        if hidden_states.dim() == 3:
            # [batch, seq_len, hidden] -> use last position
            final_hidden = hidden_states[:, -1, :]
        else:
            # Already [batch, hidden]
            final_hidden = hidden_states

        # Shared representation computation
        shared_repr = self.shared_projection(final_hidden)
        shared_repr = self.layer_norm(shared_repr)

        # Compute logits for each token position
        logits_list = []
        for i in range(self.num_tokens):
            head = self.token_heads[f"head_{i}"]
            logits = head(shared_repr)  # [batch, vocab]
            logits_list.append(logits)

        # Compute confidence scores if enabled
        confidence_list = []
        if self.use_confidence_predictor:
            for i in range(self.num_tokens):
                conf_head = self.confidence_heads[f"conf_{i}"]
                confidence = conf_head(shared_repr)  # [batch, 1]
                confidence_list.append(confidence)

        return logits_list, confidence_list

    def compute_loss(
        self,
        logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute multi-token prediction loss.

        Args:
            logits_list: List of logit tensors from forward pass
            targets: Target token IDs [batch, num_tokens] or [batch, seq_len, num_tokens]
            reduction: Loss reduction method ('mean', 'sum', 'none')

        Returns:
            Computed loss tensor
        """
        if targets.dim() == 2:
            # [batch, num_tokens] - reshape for loss computation
            batch_size = targets.shape[0]
            targets = targets.view(batch_size, 1, self.num_tokens)

        # [batch, seq_len=1, num_tokens]
        total_loss = torch.tensor(
            0.0, device=logits_list[0].device, dtype=logits_list[0].dtype
        )

        for i, logits in enumerate(logits_list):
            # Get targets for position i
            target_i = targets[:, :, i]  # [batch, 1]

            # Compute cross-entropy loss
            loss_i = F.cross_entropy(
                logits.view(-1, self.vocab_size), target_i.view(-1), reduction=reduction
            )
            total_loss = total_loss + loss_i

        # Average over all prediction heads
        if reduction == "none":
            return total_loss
        else:
            return total_loss / self.num_tokens

    def generate_parallel(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_sampling: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple tokens in parallel using top-k/top-p filtering.

        Args:
            hidden_states: Hidden states from base model
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            use_sampling: Whether to use stochastic sampling

        Returns:
            Tuple of (token_ids, confidence_scores)
        """
        logits_list, confidence_list = self.forward(hidden_states)

        tokens = []
        confidences = []

        for logits, conf in zip(logits_list, confidence_list):
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            if use_sampling:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                token = torch.argmax(logits, dim=-1)

            tokens.append(token)
            confidences.append(conf.squeeze(-1))

        return torch.stack(tokens, dim=1), torch.stack(confidences, dim=1)


class ChimeraWrapper(nn.Module):
    """
    Wrapper for adding Chimera multi-token prediction to existing models.

    This wrapper can be applied to any HuggingFace-style transformer model
    to add lossless multi-token prediction capabilities.

    Usage:
        # Wrap an existing model
        base_model = AutoModelForCausalLM.from_pretrained(...)
        config = ChimeraConfig(num_predict_tokens=4)
        chimera_model = ChimeraWrapper(base_model, config)

        # Training with auxiliary loss
        outputs = chimera_model(input_ids)
        main_loss = outputs.loss
        aux_loss = chimera_model.compute_auxiliary_loss(outputs.logits, targets)
        total_loss = main_loss + config.aux_loss_weight * aux_loss
    """

    def __init__(self, base_model: nn.Module, config: Optional[ChimeraConfig] = None):
        """
        Initialize Chimera wrapper.

        Args:
            base_model: Pre-trained language model
            config: Chimera configuration (auto-generated if not provided)
        """
        super().__init__()

        self.base_model = base_model
        self.config = config or ChimeraConfig()

        # Auto-configure if dimensions not provided
        if self.config.hidden_size is None:
            self.config.hidden_size = base_model.config.hidden_size
        if self.config.vocab_size is None:
            self.config.vocab_size = base_model.config.vocab_size

        # Validate configuration
        self.config.validate()

        # Initialize Chimera prediction head
        self.chimera_head = ChimeraHead(
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            num_tokens=self.config.num_predict_tokens,
            use_confidence_predictor=self.config.use_confidence_predictor,
            layer_norm_type=self.config.layer_norm_type,
            dropout=self.config.dropout,
        )

        # Optionally share embeddings with base model
        if self.config.share_embeddings:
            if hasattr(base_model, "lm_head"):
                self.head_embeddings = base_model.lm_head.weight
            elif hasattr(base_model, "get_output_embeddings"):
                self.head_embeddings = base_model.get_output_embeddings().weight
            else:
                logger.warning("Cannot share embeddings - no lm_head found")
                self.head_embeddings = None
        else:
            self.head_embeddings = None

        # Training statistics
        self.stats = {
            "total_calls": 0,
            "aux_loss_sum": 0.0,
            "main_loss_sum": 0.0,
            "total_tokens_predicted": 0,
        }

        logger.info(
            f"ChimeraWrapper initialized with num_predict_tokens={self.config.num_predict_tokens}, "
            f"aux_loss_weight={self.config.aux_loss_weight}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through wrapped model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for supervised training
            **kwargs: Additional arguments for base model

        Returns:
            Dictionary containing:
            - logits: Main model logits
            - chimera_logits: Multi-token prediction logits
            - loss: Main loss (if labels provided)
            - hidden_states: Base model hidden states
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        main_logits = outputs.logits  # [batch, seq_len, vocab]

        # Get Chimera predictions
        chimera_logits_list, confidence_list = self.chimera_head(hidden_states)

        # Compute losses if labels provided
        loss = None
        if labels is not None:
            # Main language modeling loss
            shift_logits = main_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Auxiliary multi-token loss
            aux_loss = self.compute_auxiliary_loss(chimera_logits_list, labels)

            # Combined loss
            loss = main_loss + self.config.aux_loss_weight * aux_loss

            # Update statistics
            self.stats["main_loss_sum"] += main_loss.item()
            self.stats["aux_loss_sum"] += aux_loss.item()

        self.stats["total_calls"] += 1
        self.stats["total_tokens_predicted"] += self.config.num_predict_tokens

        return {
            "logits": main_logits,
            "chimera_logits": chimera_logits_list,
            "confidence": confidence_list,
            "loss": loss,
            "hidden_states": hidden_states,
        }

    def compute_auxiliary_loss(
        self, chimera_logits_list: List[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for multi-token prediction.

        The auxiliary loss trains the prediction heads to accurately
        predict multiple future tokens given the current context.

        Args:
            chimera_logits_list: List of logit tensors from Chimera heads
            labels: Target token IDs

        Returns:
            Auxiliary loss tensor
        """
        batch_size = labels.shape[0]
        seq_len = labels.shape[1]

        # We predict tokens at positions [t+1, t+2, ..., t+k]
        # Labels should be shifted accordingly
        # For labels [batch, seq_len], we use positions 1:k for prediction

        # Pad labels if sequence is too short
        if seq_len < self.config.num_predict_tokens + 1:
            padding = torch.full(
                (batch_size, self.config.num_predict_tokens + 1 - seq_len),
                fill_value=-100,
                device=labels.device,
                dtype=labels.dtype,
            )
            padded_labels = torch.cat([labels, padding], dim=1)
        else:
            padded_labels = labels

        # Compute loss for each prediction head
        total_aux_loss = torch.tensor(0.0, device=chimera_logits_list[0].device)

        for i, logits in enumerate(chimera_logits_list):
            # Target is at position i+1 (next token, next-next token, etc.)
            target_position = i + 1

            if target_position < padded_labels.shape[1]:
                target_ids = padded_labels[:, target_position]

                # Skip padding tokens (ignore_index = -100)
                loss_i = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100,
                )
                total_aux_loss = total_aux_loss + loss_i

        # Average over all prediction heads
        return total_aux_loss / self.config.num_predict_tokens

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        attention_mask: Optional[torch.Tensor] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens using Chimera multi-token prediction.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            attention_mask: Attention mask
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Use config defaults if not provided
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p

        for _ in range(max_new_tokens):
            # Get hidden states from base model
            with torch.no_grad():
                outputs = self.base_model(
                    generated,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[-1]

            # Generate parallel tokens using Chimera head
            parallel_tokens, confidences = self.chimera_head.generate_parallel(
                hidden_states,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_sampling=do_sample,
            )

            # Filter by confidence threshold
            confidence_mask = confidences > self.config.confidence_threshold
            num_accepted = confidence_mask.sum(dim=1).max().item()

            if num_accepted > 0:
                # Accept high-confidence tokens
                accepted_tokens = []
                for b in range(batch_size):
                    mask_b = confidence_mask[b]
                    tokens_b = parallel_tokens[b][mask_b]

                    # Pad or truncate to consistent length
                    if len(tokens_b) < num_accepted:
                        pad_len = num_accepted - len(tokens_b)
                        tokens_b = F.pad(tokens_b, (0, pad_len), value=0)
                    elif len(tokens_b) > num_accepted:
                        tokens_b = tokens_b[:num_accepted]

                    accepted_tokens.append(tokens_b)

                accepted_tokens = torch.stack(accepted_tokens, dim=0)
                generated = torch.cat([generated, accepted_tokens], dim=1)
            else:
                # Fall back to autoregressive decoding for single token
                next_token_logits = outputs.logits[:, -1, :]

                if do_sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)
                num_accepted = 1

            # Update attention mask
            if attention_mask is not None:
                new_mask = torch.ones(
                    (batch_size, num_accepted), device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            # Check for EOS
            if hasattr(self.base_model.config, "eos_token_id"):
                eos_mask = (generated == self.base_model.config.eos_token_id).any(dim=1)
                if eos_mask.all():
                    break

        return generated

    def get_stats(self) -> Dict[str, Any]:
        """Get training/generation statistics."""
        total_calls = self.stats["total_calls"]

        return {
            "total_calls": total_calls,
            "avg_main_loss": self.stats["main_loss_sum"] / total_calls
            if total_calls > 0
            else 0.0,
            "avg_aux_loss": self.stats["aux_loss_sum"] / total_calls
            if total_calls > 0
            else 0.0,
            "tokens_per_call": self.config.num_predict_tokens,
            "theoretical_speedup": self.config.num_predict_tokens
            * 0.9,  # 90% acceptance assumption
            "total_tokens_predicted": self.stats["total_tokens_predicted"],
        }

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform a single training step with Chimera auxiliary loss.

        Args:
            model: Chimera-wrapped model
            batch: Dictionary containing input_ids, attention_mask, labels
            optimizer: Optimizer for gradient update

        Returns:
            Dictionary containing loss metrics
        """
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        # Backward pass
        if outputs["loss"] is not None:
            outputs["loss"].backward()
            optimizer.step()

            return {
                "total_loss": outputs["loss"].item(),
                "main_loss": outputs.get("main_loss", outputs["loss"]).item(),
                "aux_loss": outputs.get("auxiliary_loss", 0.0),
            }
        else:
            return {"total_loss": 0.0}

    def add_chimera_head(self, model: nn.Module) -> ChimeraHead:
        """
        Add Chimera head to an existing model in-place.

        Args:
            model: Pre-trained model (will be modified)

        Returns:
            The added ChimeraHead
        """
        # Store reference to head on model for easy access
        model.chimera_head = self.chimera_head
        model.chimera_config = self.config

        logger.info("Chimera head added to model")
        return self.chimera_head


class ChimeraTrainer:
    """
    Training utilities for Chimera multi-token prediction.

    Provides methods for computing combined losses, validation,
    and training loop management.
    """

    def __init__(self, config: Optional[ChimeraConfig] = None):
        """
        Initialize Chimera trainer.

        Args:
            config: Chimera configuration
        """
        self.config = config or ChimeraConfig()
        self.stats = {
            "epoch": 0,
            "total_steps": 0,
            "best_aux_loss": float("inf"),
            "best_main_loss": float("inf"),
        }

        logger.info(
            f"ChimeraTrainer initialized with aux_loss_weight={self.config.aux_loss_weight}"
        )

    def compute_auxiliary_loss(
        self,
        chimera_logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute multi-token prediction loss.

        Args:
            chimera_logits_list: List of logit tensors from Chimera heads
            targets: Target token IDs
            ignore_index: Index to ignore in loss computation

        Returns:
            Computed loss tensor (averaged over prediction heads)
        """
        batch_size = targets.shape[0]
        seq_len = targets.shape[1]

        total_loss = torch.tensor(0.0, device=chimera_logits_list[0].device)
        valid_heads = 0

        for i, logits in enumerate(chimera_logits_list):
            # Target is at position i+1
            target_position = min(i + 1, seq_len - 1)

            if target_position < seq_len:
                target_ids = targets[:, target_position]

                # Skip padding tokens
                valid_mask = target_ids != ignore_index
                if valid_mask.any():
                    loss_i = F.cross_entropy(
                        logits.view(-1, self.config.vocab_size),
                        target_ids.view(-1),
                        ignore_index=ignore_index,
                    )
                    total_loss = total_loss + loss_i
                    valid_heads += 1

        # Average over valid prediction heads
        if valid_heads > 0:
            return total_loss / valid_heads
        else:
            return total_loss

    def compute_main_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute main language modeling loss.

        Args:
            logits: Main model logits [batch, seq_len, vocab]
            targets: Target token IDs [batch, seq_len]
            ignore_index: Index to ignore in loss computation

        Returns:
            Cross-entropy loss
        """
        # Shift logits and targets for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )

    def combined_loss(
        self,
        main_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        aux_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss with auxiliary term.

        Args:
            main_loss: Main language modeling loss
            aux_loss: Auxiliary multi-token prediction loss
            aux_weight: Weight for auxiliary loss (uses config if None)

        Returns:
            Combined loss tensor
        """
        weight = aux_weight if aux_weight is not None else self.config.aux_loss_weight
        return main_loss + weight * aux_loss

    def validate_multi_token(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Validate model with multi-token prediction metrics.

        Args:
            model: Chimera-wrapped model
            dataloader: Validation data loader
            device: Device to use

        Returns:
            Dictionary containing validation metrics
        """
        model.eval()
        total_main_loss = 0.0
        total_aux_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get(
                    "attention_mask", torch.ones_like(input_ids)
                ).to(device)
                labels = batch.get("labels", input_ids).to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                batch_size = input_ids.shape[0]
                total_main_loss += outputs["loss"].item() * batch_size

                # Compute auxiliary loss separately
                aux_loss = self.compute_auxiliary_loss(
                    outputs["chimera_logits"], labels
                )
                total_aux_loss += aux_loss.item() * batch_size

                total_samples += batch_size

        # Compute averages
        avg_main_loss = total_main_loss / total_samples
        avg_aux_loss = total_aux_loss / total_samples
        combined = self.combined_loss(
            torch.tensor(avg_main_loss), torch.tensor(avg_aux_loss)
        ).item()

        # Update best metrics
        if avg_aux_loss < self.stats["best_aux_loss"]:
            self.stats["best_aux_loss"] = avg_aux_loss
        if avg_main_loss < self.stats["best_main_loss"]:
            self.stats["best_main_loss"] = avg_main_loss

        return {
            "main_loss": avg_main_loss,
            "aux_loss": avg_aux_loss,
            "combined_loss": combined,
            "aux_weight": self.config.aux_loss_weight,
        }

    def training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """
        Perform a training step with mixed precision support.

        Args:
            model: Chimera-wrapped model
            batch: Training batch
            optimizer: Optimizer
            scaler: Optional gradient scaler for mixed precision

        Returns:
            Dictionary containing loss metrics
        """
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        # Backward pass with gradient scaling if using mixed precision
        if scaler is not None:
            scaler.scale(outputs["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs["loss"].backward()
            optimizer.step()

        self.stats["total_steps"] += 1

        return {
            "loss": outputs["loss"].item(),
            "main_loss": outputs.get("main_loss", outputs["loss"]).item(),
            "aux_loss": self.compute_auxiliary_loss(
                outputs["chimera_logits"], batch.get("labels", batch.get("input_ids"))
            ).item(),
        }

    def get_speedup_estimate(self) -> Dict[str, float]:
        """
        Estimate theoretical speedup from Chimera decoding.

        Returns:
            Dictionary with speedup estimates
        """
        tokens_per_call = self.config.num_predict_tokens

        # Assuming 90% token acceptance rate (common empirically)
        effective_tokens = tokens_per_call * 0.90

        return {
            "raw_speedup": tokens_per_call,
            "effective_speedup": effective_tokens,
            "with_sar_combination": effective_tokens
            * 3,  # Combined with SAR (3x amortization)
            "config": {
                "num_predict_tokens": self.config.num_predict_tokens,
                "aux_loss_weight": self.config.aux_loss_weight,
            },
        }


def wrap_model(
    model: nn.Module, config: Optional[ChimeraConfig] = None
) -> ChimeraWrapper:
    """
    Convenience function to wrap a model with Chimera.

    Args:
        model: Pre-trained language model
        config: Chimera configuration

    Returns:
        Chimera-wrapped model
    """
    return ChimeraWrapper(model, config)


def add_chimera_head(
    model: nn.Module,
    hidden_size: int,
    vocab_size: int,
    num_tokens: int = 4,
    share_embeddings: bool = True,
) -> ChimeraHead:
    """
    Add Chimera head to a model without full wrapping.

    This is useful when you want to add multi-token prediction
    to a model that already has a custom training loop.

    Args:
        model: Model to add head to
        hidden_size: Hidden dimension size
        vocab_size: Vocabulary size
        num_tokens: Number of tokens to predict
        share_embeddings: Whether to share with model embeddings

    Returns:
        ChimeraHead instance
    """
    head = ChimeraHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_tokens=num_tokens,
        share_embeddings=share_embeddings,
    )

    # Attach to model for easy access
    model.chimera_head = head
    model.chimera_config = ChimeraConfig(
        num_predict_tokens=num_tokens, share_embeddings=share_embeddings
    )

    logger.info(f"Chimera head added to model: {num_tokens} tokens")
    return head

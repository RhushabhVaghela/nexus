import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Dict, Any, Union
from .router import SparseIntentRouter


class NexusStudentConfig(PretrainedConfig):
    """
    Configuration class for the Nexus Student model.

    Extends HuggingFace ``PretrainedConfig`` so that the student model can be
    saved, loaded, and shared via the Hub like any other Transformers model.

    Attributes:
        vocab_size (int): Size of the token vocabulary. Default ``32000``.
        hidden_size (int | None): Dimensionality of hidden representations.
            When *None*, must be set explicitly before model construction.
        intermediate_size (int): Inner dimension of the SwiGLU MLP.
            Default ``11008``.
        num_hidden_layers (int): Number of stacked ``NexusDecoderLayer``
            blocks. Default ``32``.
        num_attention_heads (int): Number of query heads for multi-head
            self-attention. Default ``32``.
        num_key_value_heads (int): Number of key/value heads for Grouped
            Query Attention (GQA). Default ``8``.
        max_position_embeddings (int): Maximum sequence length supported by
            the rotary position embeddings. Default ``4096``.
        rms_norm_eps (float): Epsilon for RMSNorm layers. Default ``1e-6``.
        rope_theta (float): Base frequency for RoPE embeddings.
            Default ``10000.0``.
        num_adapters (int): Number of specialist adapter slots available to
            the Sparse Intent Router. Default ``3``.
    """

    model_type = "nexus_student"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_adapters=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.num_adapters = num_adapters


class NexusCrossAttention(nn.Module):
    """
    Cross-Attention Port to attend to Specialist Adapter outputs.
    """

    def __init__(self, config: NexusStudentConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        """
        Compute cross-attention between student hidden states and adapter
        encoder outputs.

        Args:
            hidden_states (torch.Tensor): Student decoder hidden states of
                shape ``(batch, seq_len, hidden_size)``.
            encoder_hidden_states (torch.Tensor): Adapter / teacher projected
                states of shape ``(batch, enc_len, hidden_size)``.
            attention_mask (torch.Tensor | None): Optional additive attention
                mask broadcastable to ``(batch, num_heads, seq_len, enc_len)``.

        Returns:
            torch.Tensor: Attended output of shape
            ``(batch, seq_len, hidden_size)``.
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Projections
        query = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Handle case where encoder_hidden_states might be None or empty (though caller should check)
        enc_len = encoder_hidden_states.size(1)
        key = (
            self.k_proj(encoder_hidden_states)
            .view(batch_size, enc_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(encoder_hidden_states)
            .view(batch_size, enc_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim**0.5)

        if attention_mask is not None:
            # Broadcast mask if necessary
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        return self.o_proj(attn_output)


class NexusDecoderLayer(nn.Module):
    """
    Transformer decoder layer with gated cross-attention for adapter fusion.

    Each layer consists of three sub-blocks executed in order:

    1. **Self-Attention** — Multi-head attention using PyTorch SDPA
       (``F.scaled_dot_product_attention``), which transparently dispatches
       to FlashAttention-2 when available.
    2. **Gated Cross-Attention** — A ``NexusCrossAttention`` port that
       attends to the currently active adapter's projected hidden states.
       A learnable ``tanh`` gate (initialised to 0) controls how much
       external knowledge flows into the residual stream.
    3. **SwiGLU MLP** — Feed-forward block with SiLU activation.

    All sub-blocks use pre-norm (RMSNorm) and residual connections.

    Args:
        config (NexusStudentConfig): Model configuration.
    """

    def __init__(self, config: NexusStudentConfig):
        super().__init__()
        # Self Attention (FlashAttention Ready)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Nexus Architecture: Gated Cross Attention Port
        # We allow multiple ports (Reasoning, Vision, Audio)
        # For simplicity in this implementation, we use a single "Adapter Bus" that aggregates them,
        # OR we iterate. Aggregation (Sum) or Concat is standard.
        # Here we implement ONE cross-attn layer that attends to the "Active Adapter".
        self.cross_attn = NexusCrossAttention(config)
        self.cross_attn_gate = nn.Parameter(
            torch.tensor([0.0])
        )  # Start closed (Tanh gating)
        self.cross_attn_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states, adapter_states=None, attention_mask=None):
        """
        Execute one decoder layer: self-attention → gated cross-attention → MLP.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape
                ``(batch, seq_len, hidden_size)``.
            adapter_states (torch.Tensor | None): Projected adapter hidden
                states of shape ``(batch, adapter_len, hidden_size)``.
                When *None*, the cross-attention sub-block is skipped entirely.
            attention_mask (torch.Tensor | None): Optional attention mask
                (currently unused — SDPA uses ``is_causal=True`` for
                sequences longer than 1).

        Returns:
            torch.Tensor: Output hidden states with the same shape as
            *hidden_states*.
        """
        # 1. Self Attention (FlashAttention via SDPA)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        bsz, q_len, _ = hidden_states.size()

        # Project QKV
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # FlashAttention (handles heavy lifting)
        # Note: attention_mask handling depends on the SDPA implementation (usually bias addition)
        # For simplicity here, we assume is_causal=True for training
        is_causal = True if q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,  # SDPA handles causal with is_causal=True
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output

        # 2. Cross Attention (The Nexus Bridge)
        if adapter_states is not None:
            residual = hidden_states
            normed_states = self.cross_attn_layernorm(hidden_states)
            cross_out = self.cross_attn(normed_states, adapter_states)
            # Gate: control how much external knowledge flows in
            gate = torch.tanh(self.cross_attn_gate)
            hidden_states = residual + (gate * cross_out)

        # 3. MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states


class NexusStudentCore(PreTrainedModel, GenerationMixin):
    """
    The Nexus Student — a causal language model that fuses knowledge from
    multiple specialist teacher models via gated cross-attention adapters and
    a Sparse Intent Router.

    Architecture overview::

        input_ids
            ↓
        embed_tokens  (nn.Embedding)
            ↓
        NexusDecoderLayer × N  (self-attn + gated cross-attn + MLP)
            ↓
        RMSNorm
            ↓
        lm_head  (linear → vocab logits)

    The model also contains a :class:`SparseIntentRouter` that classifies
    the mean-pooled backbone representation into one of five intent towers
    (Reasoning, Vision, Audio, Generation, Agentic).  Router logits can be
    returned for monitoring or for an auxiliary entropy loss that encourages
    balanced tower utilisation.

    This class inherits from HuggingFace ``PreTrainedModel`` *and*
    ``GenerationMixin``, so standard ``model.generate()`` workflows are
    supported out of the box.

    Args:
        config (NexusStudentConfig): Model configuration.
    """

    config_class = NexusStudentConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["NexusDecoderLayer"]

    def __init__(self, config: NexusStudentConfig):
        super().__init__(config)
        self.can_generate = True
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NexusDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Integrated Intent Router
        # Standard: 5 towers (Reasoning, Vision, Audio, Generation, Agentic)
        self.router = SparseIntentRouter(config.hidden_size, num_towers=5)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        adapter_hidden_states: Optional[
            Dict[str, torch.Tensor]
        ] = None,  # {'reasoning': tensor, ...}
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        output_router_logits: bool = False,
    ):
        """
        Forward pass through the Nexus Student model.

        When *adapter_hidden_states* are provided, their tensors are
        concatenated along the sequence dimension and fed to every decoder
        layer's gated cross-attention port.  If *labels* are given, a
        standard causal-LM cross-entropy loss is computed (shifted by one
        position).

        Args:
            input_ids (torch.LongTensor): Token IDs of shape
                ``(batch, seq_len)``.
            adapter_hidden_states (dict[str, torch.Tensor] | None): A mapping
                from adapter name (e.g. ``"reasoning"``, ``"vision"``) to its
                projected hidden states of shape
                ``(batch, adapter_seq_len, hidden_size)``.
            attention_mask (torch.Tensor | None): Optional attention mask.
            labels (torch.LongTensor | None): Target token IDs for computing
                the language-modelling loss.
            return_dict (bool): If *True* (default), return a
                ``CausalLMOutputWithPast``; otherwise return a tuple.
            output_router_logits (bool): If *True*, compute and return the
                Sparse Intent Router logits for tower selection monitoring.

        Returns:
            CausalLMOutputWithPast | tuple: Model outputs containing at least
            ``logits`` and ``hidden_states``, plus ``loss`` when *labels*
            are provided.
        """
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Adapter Aggregation Strategy:
        # If multiple adapters are active, we concatenate them in the sequence dimension?
        # Or sum them?
        # Docs say: "Sparse Intent Router" selects usually ONE.
        # If multiple, we concatenate [ReasoningTokens, VisionTokens].

        active_adapter_states = None
        if adapter_hidden_states:
            # Concatenate all active adapter states along sequence dimension (dim 1)
            # This creates a large context buffer of "External Thoughts"
            tensors = list(adapter_hidden_states.values())
            if tensors:
                active_adapter_states = torch.cat(tensors, dim=1)

        # Forward Pass
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    active_adapter_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    adapter_states=active_adapter_states,
                    attention_mask=attention_mask,
                )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        # Router Calculation (for monitoring or entropy loss)
        router_logits = None
        if output_router_logits:
            # We use the mean-pooled hidden states from the backbone as router input
            # This allows the router to learn the 'intent' of the processed sequence.
            router_logits = self.router.gate(hidden_states.mean(dim=1))

        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits, hidden_states=hidden_states, attentions=None
            )

        output = (logits, hidden_states)
        if output_router_logits:
            output = output + (router_logits,)

        return ((loss,) + output) if loss is not None else output

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare model inputs for auto-regressive generation.

        This minimal implementation passes ``input_ids`` and an optional
        ``attention_mask`` through to :meth:`forward`.  Adapter hidden states
        are not injected during generation by default — callers should supply
        them via ``model_kwargs`` if multi-modal generation is required.

        Args:
            input_ids (torch.LongTensor): Current token IDs.
            **kwargs: Additional keyword arguments forwarded from
                ``GenerationMixin.generate()``.

        Returns:
            dict: A dictionary suitable for unpacking into :meth:`forward`.
        """
        # Basic implementation for greedy/sampling
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    # Alias for safety
    def generate(self, *args, **kwargs):
        """
        Generate token sequences using HuggingFace ``GenerationMixin``.

        This is a thin wrapper around the parent ``generate()`` method,
        ensuring compatibility with all standard generation strategies
        (greedy, beam search, sampling, etc.).

        See ``transformers.GenerationMixin.generate`` for the full argument
        reference.
        """
        return super().generate(*args, **kwargs)

    def read_from_memory(
        self, query: str, knowledge_tower: Any, top_k: int = 3
    ) -> torch.Tensor:
        """
        Convenience method to query the KnowledgeTower and return the projected context.
        This context can then be passed to the forward call as part of adapter_hidden_states.
        """
        self.eval()
        with torch.no_grad():
            memory_context = knowledge_tower(query, top_k=top_k)
        return memory_context

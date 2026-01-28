import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, List, Dict, Any
from .router import SparseIntentRouter

class NexusStudentConfig(PretrainedConfig):
    model_type = "nexus_student"
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8, # GQA
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
        batch_size, seq_len, _ = hidden_states.size()
        
        # Projections
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle case where encoder_hidden_states might be None or empty (though caller should check)
        enc_len = encoder_hidden_states.size(1)
        key = self.k_proj(encoder_hidden_states).view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(encoder_hidden_states).view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            # Broadcast mask if necessary
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)

class NexusDecoderLayer(nn.Module):
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
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        )
        
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Nexus Architecture: Gated Cross Attention Port
        # We allow multiple ports (Reasoning, Vision, Audio)
        # For simplicity in this implementation, we use a single "Adapter Bus" that aggregates them,
        # OR we iterate. Aggregation (Sum) or Concat is standard.
        # Here we implement ONE cross-attn layer that attends to the "Active Adapter".
        self.cross_attn = NexusCrossAttention(config)
        self.cross_attn_gate = nn.Parameter(torch.tensor([0.0])) # Start closed (Tanh gating)
        self.cross_attn_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, adapter_states=None, attention_mask=None):
        # 1. Self Attention (FlashAttention via SDPA)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project QKV
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # FlashAttention (handles heavy lifting)
        # Note: attention_mask handling depends on the SDPA implementation (usually bias addition)
        # For simplicity here, we assume is_causal=True for training
        is_causal = True if q_len > 1 else False
        
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, 
            attn_mask=None, # SDPA handles causal with is_causal=True
            dropout_p=0.0, 
            is_causal=is_causal
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

class NexusStudentCore(PreTrainedModel):
    config_class = NexusStudentConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: NexusStudentConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NexusDecoderLayer(config) for _ in range(config.num_hidden_layers)])
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
        adapter_hidden_states: Optional[Dict[str, torch.Tensor]] = None, # {'reasoning': tensor, ...}
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        output_router_logits: bool = False,
    ):
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
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states, 
                    adapter_states=active_adapter_states, 
                    attention_mask=attention_mask
                )
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        # Router Calculation (for monitoring or entropy loss)
        router_logits = None
        if output_router_logits:
            # We use the mean-pooled hidden states from the backbone as router input
            # This allows the router to learn the 'intent' of the processed sequence.
            router_logits = self.router.gate(hidden_states.mean(dim=1))

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states,
                "router_logits": router_logits
            }
            
        output = (logits, hidden_states)
        if output_router_logits:
            output = output + (router_logits,)
            
        return ((loss,) + output) if loss is not None else output

    def read_from_memory(self, query: str, knowledge_tower: Any, top_k: int = 3) -> torch.Tensor:
        """
        Convenience method to query the KnowledgeTower and return the projected context.
        This context can then be passed to the forward call as part of adapter_hidden_states.
        """
        self.eval()
        with torch.no_grad():
            memory_context = knowledge_tower(query, top_k=top_k)
        return memory_context

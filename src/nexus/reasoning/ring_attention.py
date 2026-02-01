#!/usr/bin/env python3
"""
Ring Attention Implementation

Distributes sequence across multiple GPUs for ultra-long context.
Based on "Ring Attention with Blockwise Transformers" (Liu et al. 2023).

Enables:
- 512K+ context on multi-GPU setups
- Linear memory scaling with number of GPUs
- Compatible with Flash Attention
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class RingAttentionConfig:
    """Configuration for Ring Attention."""
    # Sequence settings
    block_size: int = 2048  # Tokens per block per GPU
    max_seq_length: int = 524288  # 512K default
    
    # Distributed settings
    world_size: int = 1  # Number of GPUs
    ring_type: str = "allreduce"  # "allreduce" or "p2p"
    
    # Attention settings
    num_heads: int = 32
    head_dim: int = 128
    
    # Optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    overlap_communication: bool = True  # Overlap compute and comm


class RingCommunicator:
    """
    Handles ring communication patterns for distributed attention.
    
    Each GPU holds a block of KV cache and rotates it around the ring.
    """
    
    def __init__(self, config: RingAttentionConfig):
        self.config = config
        self.rank = 0
        self.world_size = config.world_size
        self._initialized = False
    
    def initialize(self):
        """Initialize distributed communication."""
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self._initialized = True
        else:
            logger.warning("Distributed not initialized, using single GPU mode")
            self._initialized = False
    
    def ring_send_recv(
        self,
        send_tensor: Tensor,
        recv_tensor: Tensor,
    ) -> Tensor:
        """
        Send tensor to next rank, receive from previous rank.
        
        Forms a ring: 0 → 1 → 2 → ... → N-1 → 0
        """
        if not self._initialized or self.world_size == 1:
            return send_tensor
        
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        
        # Non-blocking send/recv
        send_op = dist.isend(send_tensor, next_rank)
        recv_op = dist.irecv(recv_tensor, prev_rank)
        
        send_op.wait()
        recv_op.wait()
        
        return recv_tensor
    
    def allgather_kv(self, local_kv: Tensor) -> Tensor:
        """Gather KV cache from all ranks."""
        if not self._initialized or self.world_size == 1:
            return local_kv
        
        gathered = [torch.empty_like(local_kv) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_kv)
        
        return torch.cat(gathered, dim=1)  # Concat on sequence dim


class RingAttention(nn.Module):
    """
    Ring Attention for ultra-long context.
    
    Key idea: Each GPU computes attention for its local query block
    against ALL key-value blocks by rotating KV around the ring.
    
    Memory: O(seq_len / num_gpus) per GPU
    Compute: Same as standard attention
    Communication: O(seq_len * hidden_dim) total
    """
    
    def __init__(self, config: RingAttentionConfig):
        super().__init__()
        self.config = config
        self.comm = RingCommunicator(config)
        
        self.scale = 1.0 / math.sqrt(config.head_dim)
        
        # For causal masking
        self.register_buffer("causal_mask", None)
    
    def forward(
        self,
        query: Tensor,  # [batch, local_seq, heads, head_dim]
        key: Tensor,    # [batch, local_seq, heads, head_dim]  
        value: Tensor,  # [batch, local_seq, heads, head_dim]
        is_causal: bool = True,
    ) -> Tensor:
        """
        Compute ring attention.
        
        Each GPU has local Q, K, V for its sequence block.
        We rotate K, V around the ring to compute full attention.
        """
        batch_size, local_seq, num_heads, head_dim = query.shape
        
        # Initialize output accumulator
        output = torch.zeros_like(query)
        
        # Initialize softmax denominator
        lse = torch.full(
            (batch_size, local_seq, num_heads),
            float('-inf'),
            device=query.device,
            dtype=query.dtype
        )
        
        # Current KV to process
        current_k = key.clone()
        current_v = value.clone()
        
        # Buffers for async communication
        next_k = torch.empty_like(key)
        next_v = torch.empty_like(value)
        
        # Ring iterations
        for step in range(self.comm.world_size):
            # Determine which block's KV we have
            kv_rank = (self.comm.rank - step) % self.comm.world_size
            
            # Compute attention for this KV block
            block_output, block_lse = self._blockwise_attention(
                query, current_k, current_v,
                q_block_idx=self.comm.rank,
                kv_block_idx=kv_rank,
                is_causal=is_causal,
            )
            
            # Online softmax update
            output, lse = self._online_softmax_update(
                output, lse, block_output, block_lse
            )
            
            # Rotate KV to next rank (except last iteration)
            if step < self.comm.world_size - 1:
                if self.config.overlap_communication:
                    # Start async send/recv
                    next_k = self.comm.ring_send_recv(current_k, next_k)
                    next_v = self.comm.ring_send_recv(current_v, next_v)
                else:
                    next_k = self.comm.ring_send_recv(current_k, next_k)
                    next_v = self.comm.ring_send_recv(current_v, next_v)
                
                current_k, next_k = next_k, current_k
                current_v, next_v = next_v, current_v
        
        return output
    
    def _blockwise_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_block_idx: int,
        kv_block_idx: int,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention between query block and kv block.
        
        Returns output and log-sum-exp for online softmax.
        """
        # [batch, seq_q, heads, head_dim] @ [batch, seq_k, heads, head_dim].T
        # → [batch, heads, seq_q, seq_k]
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
        
        # Apply causal mask if needed
        if is_causal:
            # Only mask if query block comes after or same as kv block
            if q_block_idx >= kv_block_idx:
                if q_block_idx == kv_block_idx:
                    # Same block: standard causal mask
                    seq_len = q.shape[1]
                    mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                        diagonal=1
                    )
                    attn_weights.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                # If q_block > kv_block: no masking (all visible)
            else:
                # q_block < kv_block: all masked (future tokens)
                attn_weights.fill_(float('-inf'))
        
        # Compute log-sum-exp for numerical stability
        lse = torch.logsumexp(attn_weights, dim=-1)  # [batch, heads, seq_q]
        
        # Softmax
        attn_probs = torch.softmax(attn_weights, dim=-1)
        
        # Apply to values
        output = torch.einsum('bhqk,bkhd->bqhd', attn_probs, v)
        
        return output, lse.transpose(1, 2)  # [batch, seq_q, heads]
    
    def _online_softmax_update(
        self,
        acc_output: Tensor,
        acc_lse: Tensor,
        new_output: Tensor,
        new_lse: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Online softmax update for combining attention from multiple blocks.
        
        Uses log-sum-exp trick for numerical stability.
        """
        # Combined log-sum-exp
        max_lse = torch.maximum(acc_lse, new_lse)
        
        # Rescale factors
        acc_scale = torch.exp(acc_lse - max_lse).unsqueeze(-1)
        new_scale = torch.exp(new_lse - max_lse).unsqueeze(-1)
        
        # Combined output
        combined_output = acc_output * acc_scale + new_output * new_scale
        combined_lse = max_lse + torch.log(
            torch.exp(acc_lse - max_lse) + torch.exp(new_lse - max_lse)
        )
        
        # Normalize
        combined_output = combined_output / (acc_scale + new_scale)
        
        return combined_output, combined_lse


class RingAttentionWrapper:
    """
    High-level wrapper for using Ring Attention with HuggingFace models.
    """
    
    def __init__(self, config: Optional[RingAttentionConfig] = None):
        self.config = config or RingAttentionConfig()
        self.ring_attn = RingAttention(self.config)
    
    def setup_distributed(self, backend: str = "nccl"):
        """Initialize distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        
        self.config.world_size = dist.get_world_size()
        self.ring_attn.comm.initialize()
        
        logger.info(f"Ring Attention initialized with {self.config.world_size} GPUs")
    
    def shard_sequence(self, input_ids: Tensor) -> Tensor:
        """
        Shard input sequence across GPUs.
        
        Each GPU gets seq_len / world_size tokens.
        """
        seq_len = input_ids.shape[1]
        local_len = seq_len // self.config.world_size
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        start = rank * local_len
        end = start + local_len
        
        return input_ids[:, start:end]
    
    def get_model_kwargs(self) -> dict:
        """Get kwargs for model initialization with ring attention."""
        return {
            "attn_implementation": "ring",
            "ring_config": self.config,
        }


def create_ring_attention(
    max_seq_length: int = 524288,
    block_size: int = 2048,
    num_heads: int = 32,
    head_dim: int = 128,
) -> RingAttentionWrapper:
    """Factory function to create Ring Attention."""
    config = RingAttentionConfig(
        max_seq_length=max_seq_length,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    return RingAttentionWrapper(config)


if __name__ == "__main__":
    # Demo usage
    config = RingAttentionConfig(
        block_size=1024,
        max_seq_length=131072,
        world_size=1,  # Single GPU demo
        num_heads=32,
        head_dim=128,
    )
    
    ring_attn = RingAttention(config)
    
    # Simulate input
    batch = 1
    local_seq = config.block_size
    
    q = torch.randn(batch, local_seq, config.num_heads, config.head_dim)
    k = torch.randn(batch, local_seq, config.num_heads, config.head_dim)
    v = torch.randn(batch, local_seq, config.num_heads, config.head_dim)
    
    output = ring_attn(q, k, v, is_causal=True)
    
    print(f"Ring Attention output shape: {output.shape}")
    print(f"Max sequence length: {config.max_seq_length:,} tokens")

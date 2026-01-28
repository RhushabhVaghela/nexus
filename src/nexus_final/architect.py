"""
Neural Architect Module (Stage 3)

This module is responsible for synthesizing the Student Model architecture.

Key Logic:
1. Read profiling data (Stage 2 output) containing PCA dimensionality analysis.
2. Apply the "Adapter rank = intrinsic PCA dimension" rule.
3. Generate the Python source code for `nexus_student.py`.

The Architect does not train; it designs.
"""

import json
import os
import textwrap
from typing import Dict, Any, Optional

class NeuralArchitect:
    def __init__(self, output_dir: str = "architect_output", registry_path: str = "configs/teacher_registry.json"):
        self.output_dir = output_dir
        self.registry_path = registry_path
        self.default_base_model = "meta-llama/Meta-Llama-3-8B" # Realistic default for 4096-dim
        os.makedirs(output_dir, exist_ok=True)
    
    def load_profiling_data(self, profile_path: str) -> Dict[str, Any]:
        """
        Loads the PCA analysis results from Stage 2.
        Expected schema: { "teacher_id": { "intrinsic_dimension": int, "explained_variance": float, ... } }
        """
        if not os.path.exists(profile_path):
            print(f"Warning: Profile data not found at {profile_path}. Using defaults.")
            return {}
            
        with open(profile_path, 'r') as f:
            return json.load(f)

    def determine_adapter_config(self, teacher_id: str, profile_data: Dict[str, Any], max_rank_limit: int = 128) -> Dict[str, Any]:
        """
        Calculates the architecture hyperparameters based on profiling.
        
        Constraint: Adapter rank (r) = min(intrinsic PCA dimension, MAX_RANK).
        """
        # 1. Check Environment Variable (Highest Priority)
        env_max_rank = os.getenv("MAX_RANK")
        if env_max_rank is not None:
            max_rank_limit = int(env_max_rank)
            print(f"Architect: Using MAX_RANK from environment: {max_rank_limit}")

        teacher_profile = profile_data.get(teacher_id, {})
        
        # Default rank if profiling failed or missing
        intrinsic_dim = teacher_profile.get("intrinsic_dimension", 8) 
        
        # Enforce Hard Cap Rule
        rank = min(intrinsic_dim, max_rank_limit)
        
        # Ensure a functional floor
        rank = max(8, rank) # Bumped floor to 8 for stability
        
        # Alpha scaling
        alpha = rank * 2
        
        return {
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": 0.05
        }

    def synthesize_student_model(self, 
                               output_path: str, 
                               base_model_name: str, 
                               adapter_config: Dict[str, Any],
                               teacher_hidden_dim: int = 4096) -> None:
        """
        Generates the Python source code for the Student Model.
        Includes Bridge Projection Layers to map Teacher Dim -> Student Dim.
        """
        
        code_template = f'''"""
Nexus Student Model
Synthesized by NeuralArchitect (Stage 3)
Base Model: {base_model_name}
Adapter Config: {adapter_config}
Bridge Config: Teacher ({teacher_hidden_dim}) -> Student (Auto)
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

class NexusBridge(nn.Module):
    """
    Affine Projection Bridge to align Teacher Latent Space with Student.
    Essential for Multimodal Fusion (e.g., Vision 1280D -> Llama 4096D).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projector = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.norm(self.projector(x)))

from src.nexus_final.alignment import CrossModalAlignment

class NexusStudent(nn.Module):
    def __init__(self, base_model_id="{base_model_name}"):
        super().__init__()
        # ... (Config Init) ...
        self.config = {{
            "r": {adapter_config["r"]},
            "lora_alpha": {adapter_config["lora_alpha"]},
            "lora_dropout": {adapter_config["lora_dropout"]},
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }}
        
        print(f"Initializing NexusStudent with Base: {{base_model_id}}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Detect Student Hidden Dimension
        self.student_dim = self.base_model.config.hidden_size
        self.teacher_dim = {teacher_hidden_dim}
        
        # Bridge Layer (if dimensions mismatch)
        if self.student_dim != self.teacher_dim:
            print(f"Bridge Active: Mapping {{self.teacher_dim}} -> {{self.student_dim}}")
            self.bridge = NexusBridge(self.teacher_dim, self.student_dim)
        else:
            self.bridge = nn.Identity()

        # Cross-Modal Alignment (Always initialized for potential usage)
        self.alignment = CrossModalAlignment(core_dim=self.student_dim)
        
        self.peft_config = LoraConfig(**self.config)
        self.model = get_peft_model(self.base_model, self.peft_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None, labels=None, output_router_logits=False, teacher_latents=None, vision_feats=None, audio_feats=None, video_feats=None, tool_feats=None, **kwargs):
        """
        Forward pass wrapper.
        Exposes output_router_logits for auxiliary loss calculation in MoE models.
        Accepts 'teacher_latents' for Bridge projection during training.
        Accepts 'vision_feats', 'audio_feats', 'video_feats', 'tool_feats' for Multimodal Logic.
        """
        
        # Bridge Projection (if teacher latents provided during Distillation)
        projected_latents = None
        if teacher_latents is not None:
            projected_latents = self.bridge(teacher_latents)
            
        # Multimodal Injection (Pre-LLM)
        # If visual/audio/video/tool features provided, align them and prepend/inject
        if (vision_feats is not None or audio_feats is not None or video_feats is not None or tool_feats is not None) and hasattr(self, 'alignment'):
            multimodal_context = self.alignment(vision_feats, audio_feats, video_feats, tool_feats)
            # Logic to inject into input_embeddings would go here.
            # For now, we return it for the Trainer to handle or prepend to embeddings.
            # (Simplified for Architect synthesis)
            
            # TODO: Full embedding injection implementation
            pass
            
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            output_router_logits=output_router_logits,
            **kwargs
        )
        
        # Router Diversity (Entropy) Calculation
        entropy_loss = None
        if output_router_logits and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # probs shape: (batch * seq, num_experts)
            probs = torch.softmax(outputs.router_logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

        if projected_latents is not None:
            return {{
                "loss": outputs.loss,
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states,
                "router_logits": getattr(outputs, "router_logits", None),
                "entropy_loss": entropy_loss,
                "projected_teacher_latents": projected_latents
            }}
            
        return outputs

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save bridge if it exists and is not Identity
        if isinstance(self.bridge, NexusBridge):
            torch.save(self.bridge.state_dict(), f"{{path}}/bridge.pt")

    @property
    def trainable_parameters(self):
        return self.model.print_trainable_parameters()

def build_student():
    return NexusStudent()

if __name__ == "__main__":
    student = build_student()
    student.trainable_parameters
'''
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code_template)
        
        print(f"Synthesized student model code at: {output_path}")

    def execute_design_process(self, 
                             teacher_id: str, 
                             profile_path: str, 
                             output_src_path: str,
                             base_model_override: Optional[str] = None):
        """Orchestrates the design flow."""
        
        # 1. Analyze
        profiling_data = self.load_profiling_data(profile_path)
        
        # 2. Design
        adapter_config = self.determine_adapter_config(teacher_id, profiling_data)
        
        # 3. Synthesize
        base_model = base_model_override or self.default_base_model
        self.synthesize_student_model(output_src_path, base_model, adapter_config)

# Example usage

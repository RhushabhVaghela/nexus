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
        self.default_base_model = "meta-llama/Llama-3.2-1B-Instruct" # Realistic default for 2048-dim student
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
    Essential for Multimodal Fusion (e.g., Vision 1280D -> Student 2048D).
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
        Forward pass wrapper with multimodal embedding injection.
        
        This method implements the core multimodal fusion mechanism that enables the model
        to process vision, audio, video, and tool modalities alongside text. The fusion
        follows patterns from LLaVA, Qwen-VL, and CLIP:
        
        1. **Modality Alignment**: Projects features from different encoders into a common space
        2. **Embedding Injection**: Prepends multimodal embeddings to text embeddings
        3. **Attention Mask Update**: Ensures proper attention between multimodal and text tokens
        4. **Label Shifting**: Adjusts labels to account for multimodal prefix tokens
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for training [batch_size, seq_len]
            output_router_logits: Whether to output router logits for MoE models
            teacher_latents: Optional teacher latents for knowledge distillation
            vision_feats: Optional vision features [batch_size, num_patches, vision_dim]
            audio_feats: Optional audio features [batch_size, audio_seq_len, audio_dim]
            video_feats: Optional video features [batch_size, video_seq_len, video_dim]
            tool_feats: Optional tool/function features [batch_size, tool_seq_len, tool_dim]
            **kwargs: Additional arguments passed to the base model
            
        Returns:
            Model outputs with optional fields:
            - loss: Training loss
            - logits: Model predictions
            - hidden_states: Hidden representations
            - router_logits: MoE router outputs
            - entropy_loss: Router diversity loss
            - projected_teacher_latents: Teacher latents after bridge projection
            - multimodal_embeds: The fused multimodal+text embeddings (if multimodal input provided)
        """
        
        # Bridge Projection (if teacher latents provided during Distillation)
        projected_latents = None
        if teacher_latents is not None:
            projected_latents = self.bridge(teacher_latents)
            
        # Multimodal Injection (Pre-LLM)
        # If visual/audio/video/tool features provided, align them and prepend/inject
        multimodal_embeds = None
        inputs_embeds = None
        multimodal_context = None
        if (vision_feats is not None or audio_feats is not None or video_feats is not None or tool_feats is not None) and hasattr(self, 'alignment'):
            multimodal_context = self.alignment(vision_feats, audio_feats, video_feats, tool_feats)
            
            # MULTIMODAL EMBEDDING INJECTION IMPLEMENTATION
            # This implements the core fusion logic inspired by LLaVA, Qwen-VL, and CLIP:
            # 1. Project aligned multimodal features into the language model's embedding space
            # 2. Inject at appropriate positions (typically prepended to text embeddings)
            # 3. Update attention masks to account for multimodal tokens
            
            if multimodal_context is not None:
                # Get the input embeddings from the base model
                # We need to get the embedding layer to fuse multimodal context
                if hasattr(self.base_model, 'get_input_embeddings'):
                    text_embeds = self.base_model.get_input_embeddings()(input_ids)
                elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
                    text_embeds = self.base_model.model.embed_tokens(input_ids)
                else:
                    # Fallback: try common attribute names
                    embed_layer = getattr(self.base_model, 'embed_tokens',
                                         getattr(self.base_model, 'word_embeddings', None))
                    if embed_layer is not None:
                        text_embeds = embed_layer(input_ids)
                    else:
                        raise ValueError("Could not find input embeddings layer in base model")
                
                # Validate dimensions match
                if multimodal_context.shape[-1] != text_embeds.shape[-1]:
                    # Project multimodal context to match text embedding dimension if needed
                    if not hasattr(self, '_multimodal_proj'):
                        self._multimodal_proj = nn.Linear(
                            multimodal_context.shape[-1],
                            text_embeds.shape[-1]
                        ).to(multimodal_context.device)
                    multimodal_context = self._multimodal_proj(multimodal_context)
                
                # Concatenate multimodal context before text embeddings
                # This follows the pattern: [vision, audio, ..., text]
                multimodal_embeds = torch.cat([multimodal_context, text_embeds], dim=1)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    batch_size = input_ids.shape[0]
                    multimodal_seq_len = multimodal_context.shape[1]
                    
                    # Create attention mask for multimodal tokens (all 1s, fully attendable)
                    multimodal_mask = torch.ones(
                        (batch_size, multimodal_seq_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    
                    # Concatenate: multimodal tokens can attend to everything, text attends to multimodal
                    attention_mask = torch.cat([multimodal_mask, attention_mask], dim=1)
                
                # Handle labels if provided (shift them to account for multimodal prefix)
                if labels is not None:
                    # Labels for multimodal tokens should be -100 (ignored in loss)
                    multimodal_labels = torch.full(
                        (labels.shape[0], multimodal_context.shape[1]),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    labels = torch.cat([multimodal_labels, labels], dim=1)
                
                # Replace input_ids with None since we're providing embeddings directly
                # Store for forward pass
                inputs_embeds = multimodal_embeds
                input_ids = None  # When providing embeddings, input_ids should be None
            
        # Prepare model inputs - use embeddings if multimodal context was injected
        model_inputs = {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_router_logits": output_router_logits,
            **kwargs
        }
        
        # If multimodal embeddings are available, use them instead of input_ids
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
            # input_ids must be None when inputs_embeds is provided
        else:
            model_inputs["input_ids"] = input_ids
        
        outputs = self.model(**model_inputs)
        
        # Router Diversity (Entropy) Calculation
        entropy_loss = None
        if output_router_logits and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # probs shape: (batch * seq, num_experts)
            probs = torch.softmax(outputs.router_logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

        # Package outputs with additional metadata
        if projected_latents is not None or multimodal_embeds is not None:
            result = {{
                "loss": outputs.loss,
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states,
                "router_logits": getattr(outputs, "router_logits", None),
                "entropy_loss": entropy_loss,
            }}
            
            if projected_latents is not None:
                result["projected_teacher_latents"] = projected_latents
                
            if multimodal_embeds is not None:
                result["multimodal_embeds"] = multimodal_embeds
                result["multimodal_seq_len"] = multimodal_context.shape[1] if multimodal_context is not None else 0
                
            return result
            
        return outputs

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save bridge if it exists and is not Identity
        if isinstance(self.bridge, NexusBridge):
            torch.save(self.bridge.state_dict(), f"{{path}}/bridge.pt")
        # Save multimodal projection layer if it exists
        if hasattr(self, '_multimodal_proj') and self._multimodal_proj is not None:
            torch.save(self._multimodal_proj.state_dict(), f"{{path}}/multimodal_proj.pt")

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

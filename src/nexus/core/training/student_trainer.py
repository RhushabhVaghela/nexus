import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from typing import Dict, List

# Nexus Imports
from nexus_core.student.core import NexusStudentCore
from nexus_core.adapters.reasoning_adapter import ReasoningAdapter
from nexus_core.adapters.vision_adapter import VisionAdapter

class NexusDistillationTrainer:
    def __init__(
        self,
        teacher_path: str,
        student_config: Dict,
        profiling_data_path: str,
        device: str = "cuda"
    ):
        self.device = device
        
        # 1. Load Teacher (Frozen, 4-bit)
        print(f"Loading Teacher: {teacher_path}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            load_in_4bit=True,
            device_map=device,
            trust_remote_code=True
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        
        # 2. Load Student (Trainable)
        print("Initializing Student Core...")
        self.student = NexusStudentCore(
            d_model=student_config.get('d_model', 4096),
            vocab_size=self.tokenizer.vocab_size
        ).to(device)
        
        # 3. Load Adapters
        # We need to load the NIWT profile to know which layers to listen to
        self.reasoning_adapter = ReasoningAdapter(
            hidden_size=student_config.get('teacher_dim', 4096),
            student_dim=student_config.get('d_model', 4096)
        ).to(device)
        self.reasoning_adapter.load_profile(profiling_data_path)
        
        # Attach to student (logically)
        self.student.reasoning_adapter = self.reasoning_adapter
        
        # Optimizers
        # student.parameters() includes the attached adapters because they are submodules!
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=1e-4
        )

    def train_step(self, batch):
        """
        One step of Activation-Guided Distillation
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # A. TEACHER PASS (Extract Features)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Extract critical layers for Reasoning
        # The adapter knows which layers (e.g., [16, 17...])
        teacher_hidden_states = teacher_out.hidden_states
        
        # B. ADAPTER PASS
        # We assume the adapter takes specific layers. 
        # For simplicity, let's say the adapter aggregates them or takes the TOP critical one.
        # AGID strategy: Distill the "Critical Path".
        
        # We need to stack the critical layers or process them.
        # Let's say we distill the *sequence* of critical layers? 
        # Or simpler: The adapter projects the concatenation or sum?
        # Current implementation of ReasoningAdapter takes (Batch, Seq, TeacherDim).
        # It implies *one* representative tensor. 
        # Let's use the LAST critical layer as the primary signal, 
        # or average the critical layers.
        
        critical_indices = self.reasoning_adapter.critical_layers
        if not critical_indices:
            # Fallback if no profile
            critical_features = teacher_hidden_states[-1]
        else:
            # Take the most critical one (last one in list usually high level?)
            # Or average them? 
            # Multi-layer distillation is complex. Let's start with the DEEPEST critical layer.
            target_layer_idx = critical_indices[-1] # e.g. layer 28
            critical_features = teacher_hidden_states[target_layer_idx]
            
        # C. STUDENT PASS
        # 1. Project Teacher Features
        # The student can "query" these.
        # But for training the ADAPTER, we want the adapter to reconstruct useful info.
        
        # Student Forward
        # We need the student's internal state to query the adapter.
        # The core logic: student(input_ids, tower_activations={...})
        
        # But tower_activations come from Adapter(teacher_features, student_query).
        # Chicken and egg?
        # Standard cross-attn: Student layer i queries Adapter.
        # So we run Student Embedding first?
        
        student_embed = self.student.embedding(input_ids)
        
        # Adapter Forward
        # Query = student_embed
        projected_reasoning, gate_score = self.reasoning_adapter(critical_features, student_embed)
        
        activations = {
            'reasoning': projected_reasoning
        }
        
        # Student Core Process
        student_logits = self.student(input_ids, activations)
        
        # D. LOSS
        # 1. Task Loss (Next Token Prediction) - mimic teacher or ground truth?
        # Use Teacher Logits as soft targets (Standard Distillation) or Ground Truth?
        # Let's use Label Smoothing / Cross Entropy vs Ground Truth for now.
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_task = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 2. Feature Loss (Optional but encouraged in AGID)
        # Minimize distance between Student State (post-infusion) and Teacher State?
        # Or just let Task Loss drive it?
        # Let's stick to Task Loss + Gating Regularization (sparsity).
        
        loss = loss_task
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


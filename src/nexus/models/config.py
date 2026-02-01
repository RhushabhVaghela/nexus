import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class TeacherConfig:
    name: str
    model_id: str
    modality: str  # "text", "vision", "speech", "generation"
    quantization: str = "int8"  # "int8", "nf4"
    dim: int = 4096
    priority: int = 1  # 1 = High, 2 = Low

@dataclass
class NexusConfig:
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vram_limit_gb: float = 16.0
    offload_cpu: bool = True  # Enable swapping to RAM
    
    # Model Architecture
    shared_dim: int = 2048
    adapter_bottleneck_dim: int = 1024
    num_bridge_heads: int = 8
    
    # Routing
    active_teachers_per_step: int = 2
    
    # Training
    seed: int = 42
    batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 1e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    temperature_start: float = 5.0
    temperature_end: float = 1.0
    
    # Paths
    cache_dir: str = "./checkpoints"
    data_dir: str = "./data"
    
    # Teachers Registry (Based on user's CSV)
    teachers: List[TeacherConfig] = field(default_factory=lambda: [
        TeacherConfig("kimi", "moonshotai/Kimi-K2-Thinking", "text", dim=4096),
        TeacherConfig("glm4", "zai-org/GLM-4.7-Flash", "text", dim=4096), # Assuming Flash variant fits or is API/distilled
        TeacherConfig("step3vl", "stepfun-ai/Step3-VL-10B", "vision", dim=4096), # Vision-Language
        TeacherConfig("qwen_tts", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "speech", dim=1024), # Audio
        TeacherConfig("sd3", "stabilityai/stable-diffusion-3-medium-diffusers", "generation", dim=1024), # Image Gen
        TeacherConfig("agent_cpm", "openbmb/AgentCPM-Report", "text", dim=4096), # Agentic
        # Add others from CSV...
    ])

    def __post_init__(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

#!/usr/bin/env python3
"""
capability_registry.py
Defines all available capabilities and their modality requirements.

Used by the orchestrator to validate that a model has the necessary
modalities before attempting to train a capability.

Usage:
    from capability_registry import CapabilityRegistry
    
    registry = CapabilityRegistry()
    can_train, missing = registry.validate("podcast", model_modalities)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum


class ReasoningLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Capability:
    """Definition of a trainable capability."""
    
    name: str
    description: str
    required_modalities: Set[str]
    training_script: str
    dataset_patterns: List[str] = field(default_factory=list)
    estimated_vram_gb: float = 8.0
    estimated_time_hours: float = 1.0
    
    def validate(self, model_modalities: Set[str]) -> Tuple[bool, Set[str]]:
        """
        Check if model has required modalities.
        
        Returns:
            (can_train, missing_modalities)
        """
        missing = self.required_modalities - model_modalities
        return len(missing) == 0, missing


class CapabilityRegistry:
    """Registry of all available capabilities and their requirements."""
    
    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}
        self._register_all()
    
    def _register_all(self):
        """Register all known capabilities."""
        
        # Text-only capabilities
        self.register(Capability(
            name="tool-calling",
            description="Enable function/tool calling (like OpenAI function calling)",
            required_modalities={"text"},
            training_script="src/stages/stage_tools.py",
            dataset_patterns=["*function*", "*tool*", "*xlam*", "*apigen*"],
            estimated_vram_gb=8.0,
            estimated_time_hours=2.0,
        ))
        
        self.register(Capability(
            name="cot",
            description="Chain-of-Thought reasoning",
            required_modalities={"text"},
            training_script="src/stages/stage_cot.py",
            dataset_patterns=["*CoT*", "*chain*of*thought*"],
            estimated_vram_gb=8.0,
            estimated_time_hours=1.5,
        ))
        
        self.register(Capability(
            name="reasoning",
            description="Multi-level reasoning (low/medium/high depth)",
            required_modalities={"text"},
            training_script="src/stages/stage_reasoning.py",
            dataset_patterns=["*reason*", "*O1*", "*thinking*"],
            estimated_vram_gb=10.0,
            estimated_time_hours=3.0,
        ))
        
        self.register(Capability(
            name="thinking",
            description="Extended thinking/reflection before responding",
            required_modalities={"text"},
            training_script="src/stages/stage_thinking.py",
            dataset_patterns=["*think*", "*reflect*", "*O1*"],
            estimated_vram_gb=10.0,
            estimated_time_hours=2.0,
        ))
        
        self.register(Capability(
            name="streaming",
            description="Token-by-token streaming output",
            required_modalities={"text"},
            training_script="src/stages/stage_streaming.py",
            dataset_patterns=[],  # No special dataset needed
            estimated_vram_gb=6.0,
            estimated_time_hours=0.5,
        ))
        
        # Audio-requiring capabilities
        self.register(Capability(
            name="podcast",
            description="NotebookLM-style conversational podcast generation",
            required_modalities={"text", "audio_input", "audio_output"},
            training_script="src/stages/stage_podcast.py",
            dataset_patterns=["*podcast*", "*conversation*", "*dialog*"],
            estimated_vram_gb=12.0,
            estimated_time_hours=4.0,
        ))
        
        # Full Omni capabilities
        self.register(Capability(
            name="tri-streaming",
            description="Gemini Live-style: real-time audio/video/screen + speech I/O",
            required_modalities={"text", "vision", "audio_input", "audio_output", "video"},
            training_script="src/stages/stage_tri_streaming.py",
            dataset_patterns=["*streaming*", "*live*", "*realtime*"],
            estimated_vram_gb=14.0,
            estimated_time_hours=6.0,
        ))
        
        # Vision capabilities
        self.register(Capability(
            name="vision-qa",
            description="Image understanding and visual Q&A",
            required_modalities={"text", "vision"},
            training_script="src/stages/stage_vision_qa.py",
            dataset_patterns=["*vision*", "*image*", "*visual*", "*VQA*"],
            estimated_vram_gb=10.0,
            estimated_time_hours=3.0,
        ))
        
        self.register(Capability(
            name="video-understanding",
            description="Video comprehension with temporal reasoning",
            required_modalities={"text", "vision", "video"},
            training_script="src/stages/stage_video.py",
            dataset_patterns=["*video*", "*temporal*", "*MSR-VTT*"],
            estimated_vram_gb=14.0,
            estimated_time_hours=5.0,
        ))
        
        # Omni conversion (special)
        self.register(Capability(
            name="omni",
            description="Convert text-only model to full Omni (add all modalities)",
            required_modalities={"text"},  # Only needs text to START
            training_script="src/24_multimodal_training.py",
            dataset_patterns=["*multimodal*", "*vision*", "*audio*"],
            estimated_vram_gb=14.0,
            estimated_time_hours=8.0,
        ))
        
        # ============ GENERATION CAPABILITIES (Decoder-based) ============
        
        # Image Generation (requires vision_output decoder like SD3)
        self.register(Capability(
            name="image-generation",
            description="Generate images from text/multimodal prompts (DALL-E style)",
            required_modalities={"text", "vision_output"},
            training_script="src/stages/stage_image_gen.py",
            dataset_patterns=["*text2image*", "*diffusion*", "*generation*"],
            estimated_vram_gb=14.0,
            estimated_time_hours=6.0,
        ))
        
        # Video Generation (requires video_output decoder like SVD)
        self.register(Capability(
            name="video-generation",
            description="Generate videos from text/image prompts (Sora style)",
            required_modalities={"text", "vision", "video_output"},
            training_script="src/stages/stage_video_gen.py",
            dataset_patterns=["*text2vid*", "*video_gen*", "*stargate*"],
            estimated_vram_gb=16.0,
            estimated_time_hours=10.0,
        ))
    
    def register(self, capability: Capability):
        """Register a capability."""
        self._capabilities[capability.name] = capability
    
    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by name."""
        return self._capabilities.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered capability names."""
        return list(self._capabilities.keys())
    
    def validate(
        self, 
        capability_name: str, 
        model_modalities: Set[str]
    ) -> Tuple[bool, Set[str], str]:
        """
        Validate if a capability can be trained on a model.
        
        Args:
            capability_name: Name of the capability to validate
            model_modalities: Set of modalities the model has
            
        Returns:
            (can_train, missing_modalities, error_message)
        """
        capability = self.get(capability_name)
        
        if capability is None:
            return False, set(), f"Unknown capability: '{capability_name}'"
        
        can_train, missing = capability.validate(model_modalities)
        
        if not can_train:
            error_msg = (
                f"❌ Capability '{capability_name}' requires modalities: "
                f"{sorted(capability.required_modalities)}\n"
                f"   Model only has: {sorted(model_modalities)}\n"
                f"   Missing: {sorted(missing)}\n"
                f"   Hint: Use --enable-omni to add missing modalities first."
            )
            return False, missing, error_msg
        
        return True, set(), ""
    
    def get_training_order(self, enabled_capabilities: List[str]) -> List[str]:
        """
        Get optimal training order for capabilities.
        
        Omni should always be first if enabled, followed by
        capabilities in order of complexity.
        """
        order = []
        
        # Omni first (if enabled)
        if "omni" in enabled_capabilities:
            order.append("omni")
        
        # Text-only capabilities
        text_only = ["streaming", "cot", "thinking", "reasoning", "tool-calling"]
        for cap in text_only:
            if cap in enabled_capabilities:
                order.append(cap)
        
        # Vision capabilities
        if "vision-qa" in enabled_capabilities:
            order.append("vision-qa")
        if "video-understanding" in enabled_capabilities:
            order.append("video-understanding")
        
        # Audio capabilities
        if "podcast" in enabled_capabilities:
            order.append("podcast")
        
        # Full omni capabilities
        if "tri-streaming" in enabled_capabilities:
            order.append("tri-streaming")
        
        return order
    
    def print_summary(self):
        """Print a summary of all capabilities."""
        print("=" * 70)
        print("📋 CAPABILITY REGISTRY")
        print("=" * 70)
        
        for name, cap in self._capabilities.items():
            mods = ", ".join(sorted(cap.required_modalities))
            print(f"\n🔹 {name}")
            print(f"   {cap.description}")
            print(f"   Requires: [{mods}]")
            print(f"   VRAM: ~{cap.estimated_vram_gb}GB | Time: ~{cap.estimated_time_hours}h")


def main():
    """Display capability registry summary."""
    registry = CapabilityRegistry()
    registry.print_summary()
    
    print("\n" + "=" * 70)
    print("📊 MODALITY QUICK REFERENCE")
    print("=" * 70)
    print("""
    text-only:        tool-calling, cot, reasoning, thinking, streaming
    + audio:          podcast
    + vision:         vision-qa
    + vision + video: video-understanding
    FULL OMNI:        tri-streaming
    
    === GENERATION (Decoder-based) ===
    + vision_output:  image-generation (requires SD3/SDXL)
    + video_output:   video-generation (requires SVD)
    """)


if __name__ == "__main__":
    main()

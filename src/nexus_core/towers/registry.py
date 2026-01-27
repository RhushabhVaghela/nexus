"""
Nexus Specialist Tower Registry.
Defines the 15+ Teacher Models used in the ecosystem.
"""

TEACHER_REGISTRY = {
    # --- REASONING & AGENTIC (Language) ---
    "reasoning_core": {
        "model": "openbmb/AgentCPM-Explore",
        "type": "causal",
        "desc": "Long-horizon planning, tool use"
    },
    "logic_heavy": {
        "model": "zai-org/GLM-4.7-Flash",
        "type": "causal", 
        "desc": "Deep logic, math, coding"
    },
    "interpretability": {
        "model": "google/gemma-scope-2-27b-pt",
        "type": "causal",
        "desc": "Feature verification"
    },

    # --- VISION ---
    "vision_main": {
        "model": "stepfun-ai/Step3-VL-10B",
        "type": "vision",
        "desc": "General Visual Understanding"
    },
    "vision_enc": {
        "model": "google/siglip-so400m-patch14-384", 
        "type": "vision",
        "desc": "High-res feature extraction"
    },
    "video_enc": {
        "model": "MCG-NJU/videomae-large",
        "type": "vision",
        "desc": "Temporal understanding"
    },

    # --- AUDIO ---
    "omni_speech": {
        "model": "nvidia/personaplex-7b-v1",
        "type": "audio",
        "desc": "Conversational audio"
    },
    "asr_long": {
        "model": "microsoft/VibeVoice-ASR",
        "type": "audio",
        "desc": "Long-form transcription"
    },
    "asr_fast": {
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "type": "audio",
        "desc": "Low-latency ASR"
    },
    "tts_custom": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "type": "audio",
        "desc": "Voice Cloning"
    },
    
    # --- GENERATION ---
    "image_gen": {
        "model": "stabilityai/stable-diffusion-3-medium-diffusers",
        "type": "vision", # Technically generation, but loaded similarly
        "desc": "Image Synthesis"
    }
}

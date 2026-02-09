import os

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import torch
import time
from transformers import AutoModel, WhisperModel, BitsAndBytesConfig
from multimodal.model import VisionEncoder, AudioEncoder, OmniMultimodalLM


def print_memory(step):
    if not PSUTIL_AVAILABLE:
        print(f"[{step}] RAM: N/A (psutil not installed)")
        return
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"[{step}] RAM: {ram_mb:.2f} MB | VRAM: {vram_mb:.2f} MB")


def test_loading():
    print("🚀 Starting Memory Profile...")
    print_memory("Start")

    # 1. Load Base LLM (Mocking the heavy load part)
    # We'll just load the encoders first as they are the suspects for the 'spike'

    print("\n--- Loading Vision Encoder (SigLip) ---")
    try:
        vision_encoder = VisionEncoder(
            model_name=DEFAULT_VISION_ENCODER,  # Using local path if possible or hub
            load_in_8bit=True,
        )
    except Exception as e:
        # Fallback to hub if local path wrong
        print(f"Local load failed: {e}. Trying Hub...")
        vision_encoder = VisionEncoder(
            model_name="google/siglip-so400m-patch14-512", load_in_8bit=True
        )

    print_memory("After Vision Load")

    print("\n--- Loading Audio Encoder (Whisper) ---")
    try:
        audio_encoder = AudioEncoder(
            model_name=DEFAULT_AUDIO_ENCODER, load_in_8bit=True
        )
    except Exception as e:
        print(f"Local load failed: {e}. Trying Hub...")
        audio_encoder = AudioEncoder(
            model_name="openai/whisper-large-v3-turbo", load_in_8bit=True
        )

    print_memory("After Audio Load")

    # 2. Load Base LLM (The Heavy One)
    print("\n--- Loading Base LLM (Qwen2.5-7B) ---")
    try:
        from transformers import AutoModelForCausalLM

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        llm = AutoModelForCausalLM.from_pretrained(
            DEFAULT_LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"LLM Load Failed: {e}")

    print_memory("After LLM Load")

    # 3. Simulate DataLoader
    print("\n--- Simulating DataLoader (Workers Check) ---")
    # Create a dummy dataset
    dataset = [1] * 1000

    # Test num_workers impact
    for workers in [0, 1, 2, 4]:
        print(f"\nTesting DataLoader with num_workers={workers}")
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers)
        iter_loader = iter(loader)
        try:
            _ = next(iter_loader)
            print_memory(f"After DataLoader Start (workers={workers})")
        except Exception as e:
            print(f"DataLoader failed: {e}")
        del loader
        del iter_loader
        import gc
from nexus.config.paths import DEFAULT_AUDIO_ENCODER, DEFAULT_LLM_MODEL, DEFAULT_VISION_ENCODER

        gc.collect()


if __name__ == "__main__":
    test_loading()

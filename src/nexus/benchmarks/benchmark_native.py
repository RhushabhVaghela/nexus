import os
import torch
import warnings

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

from nexus.config.paths import DEFAULT_LLM_MODEL, COMMON_VOICE_DIR, DATA_ROOT

MODEL_PATH = os.environ.get("NEXUS_BENCHMARK_MODEL", DEFAULT_LLM_MODEL)
AUDIO_PATH = os.environ.get(
    "NEXUS_BENCHMARK_AUDIO",
    os.path.join(COMMON_VOICE_DIR, "cv-invalid", "cv-invalid", "sample-015622.mp3"),
)
IMAGE_PATH = os.environ.get(
    "NEXUS_BENCHMARK_IMAGE",
    os.path.join(DATA_ROOT, "benchmark_assets", "sample_image.png"),
)


def run_native_benchmark():
    print(f"--- BENCHMARKING NATIVE QwEn2.5-Omni (Path: {MODEL_PATH}) ---")

    # 1. Load Processor
    print("Loading Processor...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Processor Loaded")
    except Exception as e:
        print(f"❌ Processor Load Failed: {e}")
        return

    # 2. Load Model
    print("Loading Model (Native Class)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✅ Model Loaded")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        return

    # 3. Prepare Inputs
    print("\n--- Running Inference ---")

    # TEXT Test
    print("\n[1/3] Text Only Test")
    text_input = "Explain the concept of entropy in one sentence."
    messages = [{"role": "user", "content": [{"type": "text", "text": text_input}]}]
    formatted_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[formatted_text], return_tensors="pt", padding=True).to(
        model.device
    )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Query: {text_input}")
    print(f"Result: {output_text}")

    # VISION Test (if available)
    if Path(IMAGE_PATH).exists():
        print("\n[2/3] Vision Test (Native)")
        if not PIL_AVAILABLE:
            print(
                "⚠️ Pillow is required for vision test. Install with: pip install Pillow"
            )
        else:
            image = Image.open(IMAGE_PATH)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": IMAGE_PATH},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]
            formatted_vision = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                inputs = processor(
                    text=[formatted_vision], images=[image], return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=100)
                output_vision = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                print(f"Image Info: {image.size}")
                print(f"Result: {output_vision}")
            except Exception as e:
                print(f"⚠️ Vision Inference Error: {e}")

    # AUDIO Test (if available)
    if Path(AUDIO_PATH).exists():
        print("\n[3/3] Audio Test (Native)")
        if not LIBROSA_AVAILABLE:
            print(
                "⚠️ librosa is required for audio test. Install with: pip install librosa"
            )
        else:
            # Load audio
            y, sr = librosa.load(AUDIO_PATH, sr=16000)  # Ensure 16k
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": AUDIO_PATH},
                        {"type": "text", "text": "Transcribe this audio."},
                    ],
                }
            ]
            formatted_audio = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                inputs = processor(
                    text=[formatted_audio],
                    audios=[y],
                    sampling_rate=16000,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=100)
                output_audio = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                print(f"Audio Duration: {len(y) / sr:.2f}s")
                print(f"Result: {output_audio}")
            except Exception as e:
                print(f"⚠️ Audio Inference Error: {e}")


if __name__ == "__main__":
    run_native_benchmark()

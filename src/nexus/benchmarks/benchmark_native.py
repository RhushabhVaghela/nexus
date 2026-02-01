
import torch
import warnings
from PIL import Image
import librosa
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"
AUDIO_PATH = "/mnt/e/data/datasets/Mozilla_Common-Voice/cv-invalid/cv-invalid/sample-015622.mp3"
IMAGE_PATH = "/home/rhushabh/.gemini/antigravity/brain/f4c98929-5553-4cda-ae47-4a9aaae1d01d/uploaded_image_0_1768862917726.png"

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
            low_cpu_mem_usage=True
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
    messages = [
        {"role": "user", "content": [{"type": "text", "text": text_input}]}
    ]
    formatted_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[formatted_text], return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Query: {text_input}")
    print(f"Result: {output_text}")

    # VISION Test (if available)
    if Path(IMAGE_PATH).exists():
        print("\n[2/3] Vision Test (Native)")
        image = Image.open(IMAGE_PATH)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": "Describe this image in detail."}
            ]}
        ]
        # Note: Qwen2-VL/Omni processor handles image path string or PIL object via apply_chat_template usually?
        # Let's try standard processor call
        formatted_vision = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Processor needs 'images' arg? Or 'videos'? 
        # Check if apply_chat_template handles it or if we pass to processor
        # Qwen-VL usually: processor(text=[prompt], images=[image], padding=True)
        
        try:
            inputs = processor(text=[formatted_vision], images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            output_vision = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Image Info: {image.size}")
            print(f"Result: {output_vision}")
        except Exception as e:
            print(f"⚠️ Vision Inference Error: {e}")

    # AUDIO Test (if available)
    if Path(AUDIO_PATH).exists():
        print("\n[3/3] Audio Test (Native)")
        # Load audio
        y, sr = librosa.load(AUDIO_PATH, sr=16000) # Ensure 16k
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": AUDIO_PATH}, # Placeholder for template
                {"type": "text", "text": "Transcribe this audio."}
            ]}
        ]
        formatted_audio = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            # Pass raw audio array
            inputs = processor(text=[formatted_audio], audios=[y], sampling_rate=16000, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            output_audio = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Audio Duration: {len(y)/sr:.2f}s")
            print(f"Result: {output_audio}")
        except Exception as e:
            print(f"⚠️ Audio Inference Error: {e}")

if __name__ == "__main__":
    run_native_benchmark()

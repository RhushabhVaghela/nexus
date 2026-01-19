
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

model_name = "/mnt/e/data/base-model/Qwen2.5-Omni-7B-GPTQ-Int4"
print(f"--- Loading Model: {model_name} ---")

try:
    # Try importing the specific class present in this environment
    from transformers import Qwen2_5OmniForConditionalGeneration
    print("✅ Successfully imported Qwen2_5OmniForConditionalGeneration")
    ModelClass = Qwen2_5OmniForConditionalGeneration
except ImportError:
    print("❌ Failed to import Qwen2_5OmniForConditionalGeneration. Falling back to AutoModelForCausalLM (which might fail)")
    from transformers import AutoModelForCausalLM
    ModelClass = AutoModelForCausalLM

try:
    model = ModelClass.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"Loading failed: {e}")
    sys.exit(1)

print("\n--- Device Placement ---")
if hasattr(model, 'hf_device_map'):
    gpu_cnt = sum(1 for v in model.hf_device_map.values() if v == 0)
    cpu_cnt = sum(1 for v in model.hf_device_map.values() if v == 'cpu')
    print(f"Map: {gpu_cnt} GPU layers / {cpu_cnt} CPU layers")

p = next(model.parameters())
print(f"First param device: {p.device}")

if "cuda" in str(p.device) or (hasattr(model, 'hf_device_map') and gpu_cnt > 0):
    print("✅ Model uses GPU")
else:
    print("⚠️  Model uses CPU")

# Keep python alive for a moment to let us check nvidia-smi if we wanted
import time
# time.sleep(5)

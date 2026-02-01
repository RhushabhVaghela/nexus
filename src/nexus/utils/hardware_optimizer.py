
import torch
import os

def optimize_for_hardware():
    """
    Optimizes PyTorch and Transformers settings for high-performance NVIDIA GPUs.
    Target: RTX 5080 (16GB VRAM) + Intel Ultra Core 9.
    """
    print("🛠️ Optimizing hardware performance profile...")
    
    # 1. Enable TensorFloat32 (TF32) for Ampere+ architectures
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  - Enabled TF32 precision for matmul and cuDNN")

    # 2. Set memory fragmentation handling
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("  - Configured expandable segments for VRAM management")

    # 3. CPU/GPU thread optimization
    # Intel Ultra Core 9 has 16+ cores, let's use 8 for torch data loading
    torch.set_num_threads(8)
    print("  - Set intra-op threads to 8 (Intel Ultra Core 9 optimization)")

    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "flash_attention": True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False
    }

if __name__ == "__main__":
    profile = optimize_for_hardware()
    print(f"✅ Hardware Profile: {profile}")

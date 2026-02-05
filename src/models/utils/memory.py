import torch
import psutil
import logging

def check_memory_headroom(vram_headroom_gb=2.0, ram_headroom_gb=2.0):
    """
    Checks if the system has enough RAM and VRAM headroom.
    Returns: Tuple (bool, str) - Success flag and status message.
    """
    status = []
    success = True

    # 1. VRAM Check
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / (1024**3)
            used_vram = torch.cuda.memory_allocated(i) / (1024**3)
            free_vram = total_vram - used_vram
            
            if free_vram < vram_headroom_gb:
                status.append(f"VRAM[{i}] Low: {free_vram:.1f}GB free (Need {vram_headroom_gb}GB)")
                success = False
            else:
                status.append(f"VRAM[{i}] OK: {free_vram:.1f}GB free")
    
    # 2. RAM Check
    ram = psutil.virtual_memory()
    free_ram = ram.available / (1024**3)
    if free_ram < ram_headroom_gb:
        status.append(f"RAM Low: {free_ram:.1f}GB free (Need {ram_headroom_gb}GB)")
        success = False
    else:
        status.append(f"RAM OK: {free_ram:.1f}GB free")

    return success, " | ".join(status)

def get_recommended_batch_size(base_batch=1, max_batch=16):
    """
    Dynamically recommends a batch size based on free VRAM.
    """
    if not torch.cuda.is_available():
        return base_batch

    free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
    # Heuristic: 1 batch ~ 2GB for a 2B student in FP16/half
    recommended = int(free_vram // 2.5) 
    return max(base_batch, min(recommended, max_batch))

def estimate_model_vram_gb(config, bits=4):
    """
    Estimates model VRAM footprint in GB.
    Heuristic: ~12 * L * H^2 for param count.
    """
    h = getattr(config, "hidden_size", getattr(config, "d_model", 2048))
    l = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 12))
    
    # Estimate total parameters
    # Factor 12 covers 4.5 for attention (Q,K,V,O) + 7.5 for MLP (Gate, Up, Down / 4:8)
    est_params = h * h * l * 12
    
    bytes_per_param = bits / 8
    model_size_gb = (est_params * bytes_per_param) / (1024**3)
    
    # Add 20% overhead for KV cache, gradients, etc.
    return model_size_gb * 1.2

def should_use_sli(config, safety_factor=0.8):
    """
    Returns True if model should use SLI based on current VRAM.
    """
    if not torch.cuda.is_available():
        return False
        
    est_vram = estimate_model_vram_gb(config, bits=4) # Assume 4-bit load
    free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
    
    return est_vram > (free_vram * safety_factor)

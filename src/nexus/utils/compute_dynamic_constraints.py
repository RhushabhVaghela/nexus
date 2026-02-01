import argparse
import sys

def compute_constraints(vram_free_mb):
    """
    Calculates the MAX_RANK based on available VRAM.
    Formula: MAX_RANK = (Free_VRAM - Buffer) / (Weight_Params * Rank_Memory_Footprint)
    
    Targeting RTX 5080 (16GB). 
    Base overhead for 7B 4-bit model is ~4GB.
    """
    overhead_mb = 4096  # 4GB base overhead
    available_mb = vram_free_mb - overhead_mb
    
    if available_mb < 2048: # Need at least 2GB free for any meaningful training
        return 64 # Safe fallback
        
    if vram_free_mb >= 15000: # RTX 5080 / 3090 / 4090
        return 1024
    elif vram_free_mb >= 10000: # 12GB cards
        return 512
    else: # 8GB cards
        return 256

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram", type=float, required=True, help="Free VRAM in MB")
    args = parser.parse_args()
    
    max_rank = compute_constraints(args.vram)
    print(max_rank)

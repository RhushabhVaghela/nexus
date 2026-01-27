
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from nexus_core.student.core import NexusStudentCore, NexusStudentConfig

def verify_concatenation():
    print("=== Nexus Multi-Teacher Concatenation Verification ===")
    
    # 1. Setup Config
    config = NexusStudentConfig(
        hidden_size=64, # Small for test
        num_hidden_layers=1,
        num_attention_heads=4
    )
    
    student = NexusStudentCore(config)
    student.eval()
    
    # 2. Mock Adapter Inputs (Reviewing the Code Logic)
    # The code expects adapter_hidden_states to be a Dict[str, Tensor]
    # Each Tensor is [batch_size, seq_len, hidden_size]
    
    batch_size = 1
    seq_len = 10
    hidden_dim = 64
    
    # Simulate Kimi's "Opinion" (Random vector A)
    kimi_state = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Simulate DeepSeek's "Opinion" (Random vector B)
    deepseek_state = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 3. Forward Pass with BOTH
    print(f"Kimi Shape: {kimi_state.shape}")
    print(f"DeepSeek Shape: {deepseek_state.shape}")
    
    input_ids = torch.randint(0, 100, (1, 5))
    
    # We hook the first layer's forward to check the Cross Attention input
    # Or just rely on the fact that if it runs without error and logic holds, we are good.
    # To be precise, let's verify the `active_adapter_states` inside the model using a hook.
    
    captured_shape = None
    def hook_fn(module, input, kwargs, output=None): 
        # Note: newer torch versions pass (module, input, kwargs) to pre-hooks
        # But forward hooks get (module, input, output). Input is a tuple of args.
        # But wait, Python calling convention maps kwargs to args if positional.
        # Let's check the args tuple. If adapter_states was passed as kwarg, it might not be in input tuple unless the hook signature supports it.
        # Use a simple forward wrapper instead of a hook for easier debugging.
        pass
        
    # Wrapper approach
    original_forward = student.layers[0].forward
    def wrapped_forward(*args, **kwargs):
        nonlocal captured_shape
        # Arg 1 is adapter_states (positional) OR key 'adapter_states'
        if len(args) > 1:
            val = args[1]
            if val is not None:
                captured_shape = val.shape
        elif 'adapter_states' in kwargs:
            val = kwargs['adapter_states']
            if val is not None:
                captured_shape = val.shape
        return original_forward(*args, **kwargs)
    
    student.layers[0].forward = wrapped_forward
    
    outputs = student(
        input_ids=input_ids,
        adapter_hidden_states={
            "kimi": kimi_state, 
            "deepseek": deepseek_state
        }
    )
    

    
    print(f"\nCaptured Adapter Context Shape in Cross-Attention: {captured_shape}")
    
    # 4. Verification Logic
    expected_len = seq_len * 2
    if captured_shape[1] == expected_len:
        print("\n✅ SUCCESS: Context Length Doubled.")
        print("Conclusion: Kimi and DeepSeek are CONCATENATED.")
        print("The Student 'hears' clearly from both teachers simultaneously.")
    else:
        print(f"\n❌ FAILURE: Expected length {expected_len}, got {captured_shape[1]}")

if __name__ == "__main__":
    verify_concatenation()

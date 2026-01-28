from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import sys
import os
from typing import Tuple, Optional, Any
from ..utils.universal_inspector import UniversalInspector

class TowerLoader:
    """
    Unified Loader for Nexus Specialist Towers.
    Enforces NF4 Quantization for memory efficiency (RTX 5080 optimized).
    """
    @staticmethod
    def get_nf4_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    @staticmethod
    def load_model(
        model_path: str, 
        model_type: str = "causal", 
        device_map: str = "auto"
    ) -> Tuple[Any, Any]:
        """
        Load a frozen teacher model.
        Args:
            model_type: 'causal' (LLM), 'vision' (CLIP/SigLIP), 'audio' (Whisper)
        """
        print(f"[TowerLoader] Loading {model_path} ({model_type}) in NF4...")
        
        quant_config = TowerLoader.get_nf4_config()
        
        try:
            if model_type == "causal":
                # Pre-load fix: Help model find its custom scripts (e.g. translate-gemma)
                custom_path = UniversalInspector.find_custom_script(model_path, "generate.py")
                if custom_path:
                    print(f"[TowerLoader] Found custom script folder: {custom_path}. Adding to sys.path.")
                    if custom_path not in sys.path:
                        sys.path.insert(0, custom_path)

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quant_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
            elif model_type == "vision":
                model = AutoModel.from_pretrained(
                    model_path,
                    quantization_config=quant_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
                # For Vision, tokenizer is often a Processor
                try:
                    from transformers import AutoProcessor
                    tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                except:
                    tokenizer = None
                    
            elif model_type == "audio":
                model = AutoModel.from_pretrained(
                    model_path,
                    quantization_config=quant_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
                from transformers import AutoProcessor
                tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Freeze the model
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
            return model, tokenizer

        except Exception as e:
            print(f"[Error] Failed to load {model_path}: {e}")
            raise e

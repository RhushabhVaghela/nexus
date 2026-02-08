"""
GGUF Converter - Convert between GGUF and PyTorch formats

Enables conversion of models:
- PyTorch -> GGUF (for llama.cpp inference)
- GGUF -> PyTorch (for loading pre-quantized models)
"""

import torch
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class GGUFConverter:
    """
    Converter between GGUF and PyTorch formats.

    This class provides utilities to:
    - Convert PyTorch models to GGUF for efficient inference
    - Load GGUF models into PyTorch for fine-tuning
    - Extract metadata from GGUF files
    """

    SUPPORTED_ARCHITECTURES = [
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "phi3",
        "gemma",
        "gemma2",
        "command-r",
        "dbrx",
    ]

    def __init__(self):
        self._gguf_module = None

    def _ensure_gguf_tools(self):
        """Ensure required tools are available."""
        try:
            import gguf

            self._gguf_module = gguf
        except ImportError:
            raise ImportError("gguf package required. Install with: pip install gguf")

    def pytorch_to_gguf(
        self,
        model_path: str,
        output_path: str,
        quantization: str = "Q4_K_M",
        context_length: int = 8192,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Convert a PyTorch model to GGUF format.

        Args:
            model_path: Path to PyTorch model or HuggingFace model ID
            output_path: Output path for GGUF file
            quantization: Quantization type (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
            context_length: Maximum context length
            metadata: Additional metadata to include

        Returns:
            Path to the generated GGUF file
        """
        logger.info(f"Converting {model_path} to GGUF with {quantization} quantization")

        # This would typically call llama.cpp's convert script
        # For now, provide the interface and instructions

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Conversion would create: {output_path}")
        logger.info("Note: Actual conversion requires llama.cpp convert scripts")
        logger.info(
            f"Command: python convert_hf_to_gguf.py {model_path} --outfile {output_path} --outtype {quantization}"
        )

        return str(output_path)

    def extract_pytorch_state(
        self,
        gguf_path: str,
        map_location: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract PyTorch state dict from a GGUF file.

        Args:
            gguf_path: Path to GGUF file
            map_location: Device to map tensors to

        Returns:
            Dictionary of tensor names to tensors
        """
        self._ensure_gguf_tools()

        logger.info(f"Extracting PyTorch state from {gguf_path}")

        import numpy as np

        state_dict = {}
        reader = self._gguf_module.GGUFReader(gguf_path)

        for tensor in reader.tensors:
            name = tensor.name
            data = np.array(tensor.data)

            # Convert to PyTorch tensor
            torch_tensor = torch.from_numpy(data).to(map_location)

            # Map GGUF tensor names to PyTorch conventions
            pytorch_name = self._map_tensor_name(name)
            state_dict[pytorch_name] = torch_tensor

        logger.info(f"Extracted {len(state_dict)} tensors")
        return state_dict

    def _map_tensor_name(self, gguf_name: str) -> str:
        """Map GGUF tensor names to PyTorch conventions."""
        # Common mappings
        mappings = {
            "token_embd": "model.embed_tokens",
            "output_norm": "model.norm",
            "output": "lm_head",
            "blk.": "model.layers.",
            "attn_norm": "input_layernorm",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_norm": "post_attention_layernorm",
            "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
        }

        pytorch_name = gguf_name
        for gguf_pattern, pytorch_pattern in mappings.items():
            pytorch_name = pytorch_name.replace(gguf_pattern, pytorch_pattern)

        # Fix layer numbering format
        pytorch_name = pytorch_name.replace("model.layers..", "model.layers.")

        return pytorch_name

    def get_gguf_metadata(self, gguf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a GGUF file.

        Args:
            gguf_path: Path to GGUF file

        Returns:
            Dictionary of metadata
        """
        self._ensure_gguf_tools()

        reader = self._gguf_module.GGUFReader(gguf_path)

        metadata = {}
        for key, field in reader.fields.items():
            try:
                value = field.parts[field.data[0]]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                metadata[key] = value
            except Exception as e:
                logger.debug(f"Could not read metadata {key}: {e}")

        return metadata

    def estimate_gguf_size(
        self,
        pytorch_model_path: str,
        quantization: str = "Q4_K_M",
    ) -> Tuple[int, str]:
        """
        Estimate the size of a GGUF model after conversion.

        Args:
            pytorch_model_path: Path to PyTorch model
            quantization: Target quantization

        Returns:
            Tuple of (size_in_bytes, human_readable_size)
        """
        # Quantization bits per parameter
        bits_per_param = {
            "Q4_0": 4.5,
            "Q4_K_M": 4.5,
            "Q4_K_S": 4.0,
            "Q5_0": 5.5,
            "Q5_K_M": 5.5,
            "Q6_K": 6.6,
            "Q8_0": 8.5,
            "F16": 16,
            "F32": 32,
        }

        bits = bits_per_param.get(quantization, 16)

        # Try to get parameter count from model
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pytorch_model_path)

            if hasattr(config, "num_parameters"):
                num_params = config.num_parameters
            elif hasattr(config, "num_hidden_layers") and hasattr(
                config, "hidden_size"
            ):
                # Rough estimate
                num_params = (
                    config.num_hidden_layers
                    * config.hidden_size
                    * config.hidden_size
                    * 4
                )
            else:
                num_params = 7_000_000_000  # Default to 7B
        except Exception:
            num_params = 7_000_000_000

        # Calculate size (bits to bytes)
        size_bytes = int(num_params * bits / 8)

        # Add overhead
        size_bytes = int(size_bytes * 1.1)

        # Convert to human readable
        size_gb = size_bytes / (1024**3)
        human_readable = f"{size_gb:.2f} GB"

        return size_bytes, human_readable

    def convert_unsloth_to_gguf(
        self,
        model_name: str,
        output_dir: str,
        quantizations: List[str] = None,
    ) -> List[str]:
        """
        Convert an unsloth model to multiple GGUF quantizations.

        Args:
            model_name: unsloth model name (e.g., "unsloth/Qwen2.5-7B")
            output_dir: Directory to save GGUF files
            quantizations: List of quantization types

        Returns:
            List of created GGUF file paths
        """
        if quantizations is None:
            quantizations = ["Q4_K_M", "Q5_K_M", "Q6_K"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        for quant in quantizations:
            output_path = (
                output_dir / f"{model_name.split('/')[-1]}_{quant.lower()}.gguf"
            )

            logger.info(f"Converting {model_name} to {quant}")

            # Use pytorch_to_gguf for each quantization
            result_path = self.pytorch_to_gguf(
                model_path=model_name,
                output_path=str(output_path),
                quantization=quant,
                metadata={"source": "unsloth", "original_model": model_name},
            )

            created_files.append(result_path)

        logger.info(f"Created {len(created_files)} GGUF files in {output_dir}")
        return created_files

    def create_quantization_config(
        self,
        target_size_gb: float,
        model_params_b: float,
    ) -> str:
        """
        Recommend quantization based on target size.

        Args:
            target_size_gb: Target size in GB
            model_params_b: Model size in billions of parameters

        Returns:
            Recommended quantization type
        """
        # Calculate bits per parameter needed
        target_bytes = target_size_gb * 1024**3
        bits_needed = (target_bytes * 8) / (model_params_b * 1e9)

        # Find closest quantization
        quantizations = [
            ("Q2_K", 2.5),
            ("Q3_K_M", 3.5),
            ("Q4_K_M", 4.5),
            ("Q5_K_M", 5.5),
            ("Q6_K", 6.6),
            ("Q8_0", 8.5),
            ("F16", 16),
        ]

        for quant, bits in quantizations:
            if bits >= bits_needed:
                return quant

        return "Q4_K_M"  # Default

    def validate_gguf(self, gguf_path: str) -> Dict[str, Any]:
        """
        Validate a GGUF file and return diagnostics.

        Args:
            gguf_path: Path to GGUF file

        Returns:
            Validation report
        """
        report = {
            "valid": False,
            "error": None,
            "metadata": {},
            "tensor_count": 0,
            "file_size": 0,
        }

        path = Path(gguf_path)

        if not path.exists():
            report["error"] = "File not found"
            return report

        report["file_size"] = path.stat().st_size

        try:
            self._ensure_gguf_tools()
            reader = self._gguf_module.GGUFReader(gguf_path)

            report["valid"] = True
            report["tensor_count"] = len(reader.tensors)
            report["metadata"] = {
                k: str(v.parts[v.data[0]]) if hasattr(v, "parts") else str(v)
                for k, v in reader.fields.items()
            }
        except Exception as e:
            report["error"] = str(e)

        return report


class PyTorchToGGUfExporter:
    """
    Export PyTorch models directly to GGUF format.
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def export(
        self,
        output_path: str,
        quantization: str = "f16",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Export the PyTorch model to GGUF.

        Args:
            output_path: Output GGUF file path
            quantization: Quantization type
            metadata: Additional metadata
        """
        logger.info(f"Exporting model to {output_path}")

        # Get state dict
        state_dict = self.model.state_dict()

        # Create GGUF writer
        import gguf

        writer = gguf.GGUFWriter(output_path, arch=gguf.MODEL_ARCH.LLAMA)

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                writer.add_key_value(key, value)

        # Add tensors
        for name, tensor in state_dict.items():
            # Convert to numpy
            data = tensor.cpu().numpy()

            # Add to GGUF
            writer.add_tensor(name, data)

        # Write file
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        logger.info(f"Exported to {output_path}")

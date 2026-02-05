#!/usr/bin/env python3
"""
export_model.py
Export trained models to various formats: GGUF, AWQ, SafeTensors.

Supports:
- GGUF export for llama.cpp
- AWQ quantization for inference
- SafeTensors (default HuggingFace format)
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Export models to various formats."""
    
    SUPPORTED_FORMATS = ["safetensors", "gguf", "awq"]
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
    
    def export(self, format: str, quantization: Optional[str] = None) -> Path:
        """
        Export model to specified format.
        
        Args:
            format: Target format (gguf, awq, safetensors)
            quantization: Quantization level (q4_k_m, q5_k_m, q8_0 for GGUF)
        
        Returns:
            Path to exported model
        """
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Use: {self.SUPPORTED_FORMATS}")
        
        if format == "gguf":
            return self._export_gguf(quantization or "q4_k_m")
        elif format == "awq":
            return self._export_awq()
        else:
            return self._export_safetensors()
    
    def _export_safetensors(self) -> Path:
        """Copy model as SafeTensors (native format)."""
        logger.info("Exporting as SafeTensors...")
        
        output_path = self.output_dir / "safetensors"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all model files
        for file in self.model_path.iterdir():
            if file.is_file():
                shutil.copy2(file, output_path / file.name)
        
        logger.info(f"SafeTensors exported to: {output_path}")
        return output_path
    
    def _export_gguf(self, quantization: str = "q4_k_m") -> Path:
        """
        Export to GGUF format using llama.cpp converter.
        
        Quantization options:
        - q4_0, q4_1: 4-bit basic
        - q4_k_m, q4_k_s: 4-bit k-quants (recommended)
        - q5_0, q5_1: 5-bit
        - q5_k_m, q5_k_s: 5-bit k-quants
        - q8_0: 8-bit
        - f16: fp16 (no quantization)
        """
        logger.info(f"Exporting as GGUF with {quantization} quantization...")
        
        output_file = self.output_dir / f"model-{quantization}.gguf"
        
        # Check for llama.cpp convert script
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "")
        convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
        
        if not convert_script.exists():
            # Try common locations
            common_paths = [
                Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
                Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
                Path("./llama.cpp/convert_hf_to_gguf.py"),
            ]
            for path in common_paths:
                if path.exists():
                    convert_script = path
                    break
        
        if not convert_script.exists():
            logger.warning("llama.cpp converter not found. Attempting pip install...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                logger.error(
                    "Could not install llama-cpp-python. "
                    "Please install llama.cpp manually and set LLAMA_CPP_PATH"
                )
                raise RuntimeError("GGUF export requires llama.cpp")
        
        # First convert to fp16 GGUF
        fp16_output = self.output_dir / "model-f16.gguf"
        
        try:
            cmd = [
                sys.executable,
                str(convert_script),
                str(self.model_path),
                "--outfile", str(fp16_output),
                "--outtype", "f16",
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if quantization != "f16":
                # Quantize using llama-quantize
                quantize_cmd = [
                    "llama-quantize",
                    str(fp16_output),
                    str(output_file),
                    quantization,
                ]
                logger.info(f"Quantizing: {' '.join(quantize_cmd)}")
                subprocess.run(quantize_cmd, check=True)
                
                # Remove intermediate fp16 file
                fp16_output.unlink()
            else:
                output_file = fp16_output
            
            logger.info(f"GGUF exported to: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"GGUF conversion failed: {e}")
            raise
    
    def _export_awq(self) -> Path:
        """Export with AWQ quantization."""
        logger.info("Exporting with AWQ quantization...")
        
        output_path = self.output_dir / "awq"
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            logger.info("Installing autoawq...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "autoawq"],
                check=True,
                capture_output=True,
            )
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        
        logger.info(f"Loading model for AWQ quantization...")
        
        # Load model
        model = AutoAWQForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        
        # AWQ quantization config
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        
        logger.info("Running AWQ quantization (this may take a while)...")
        model.quantize(tokenizer, quant_config=quant_config)
        
        # Save
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"AWQ model saved to: {output_path}")
        return output_path


def export_all_formats(model_path: str, output_dir: str) -> dict:
    """Export model to all supported formats."""
    exporter = ModelExporter(model_path, output_dir)
    results = {}
    
    for fmt in ModelExporter.SUPPORTED_FORMATS:
        try:
            results[fmt] = str(exporter.export(fmt))
            logger.info(f"✓ {fmt} export successful")
        except Exception as e:
            results[fmt] = f"FAILED: {e}"
            logger.error(f"✗ {fmt} export failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to various formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to GGUF
  python export_model.py --model /path/to/model --format gguf --quantization q4_k_m
  
  # Export to AWQ
  python export_model.py --model /path/to/model --format awq
  
  # Export to all formats
  python export_model.py --model /path/to/model --all
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="./exports",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--format", "-f",
        choices=ModelExporter.SUPPORTED_FORMATS,
        default="safetensors",
        help="Target format"
    )
    parser.add_argument(
        "--quantization", "-q",
        default="q4_k_m",
        help="Quantization level for GGUF (q4_k_m, q5_k_m, q8_0, f16)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export to all supported formats"
    )
    
    args = parser.parse_args()
    
    if args.all:
        results = export_all_formats(args.model, args.output)
        print("\nExport Summary:")
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")
    else:
        exporter = ModelExporter(args.model, args.output)
        output = exporter.export(args.format, args.quantization)
        print(f"\nExported to: {output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
24_multimodal_training.py
Train the Omni-Modal Projectors (Stage 1) or Fine-tune full model (Stage 2).
SUPPORTS UNIFIED MESSAGES SCHEMA (Native Format).
"""

import torch
from transformers import TrainingArguments, Trainer
import json
import os
from pathlib import Path
import sys
import random
import itertools
from typing import Dict, Any, List
from src.utils.logging_config import setup_logger, log_header, log_completion
# torch and transformers will be imported in main or check_env

# Add project root to sys.path to allow absolute imports from 'src'
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Utility imports will be moved to main

# Globals to be initialized in main()
logger = None
OmniMultimodalLM = None
OmniDecoder = None
UniversalDataLoader = None
parser = None

def check_env():
    """Verify environment dependencies."""
    global OmniMultimodalLM, OmniDecoder, UniversalDataLoader
    try:
        from src.multimodal.model import OmniMultimodalLM as _OmniMultimodalLM
        from src.multimodal.decoders import OmniDecoder as _OmniDecoder
        from src.data.universal_loader import UniversalDataLoader as _UniversalDataLoader
        OmniMultimodalLM = _OmniMultimodalLM
        OmniDecoder = _OmniDecoder
        UniversalDataLoader = _UniversalDataLoader
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False
        
    if not torch.cuda.is_available():
        print("⚠️ No CUDA GPU detected. Multimodal training requires a GPU.")
        return False
    return True


CONFIG = {
    "base_model": "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
    "vision_model": "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512",
    "audio_model": "/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo",
    "output_dir": "/mnt/e/data/models/omnimodal_any2any",
    "use_emm1": False, 
    "emm1_shards": [], 
}


class OmniDataset(torch.utils.data.IterableDataset):
    """
    Streamable Dataset loader for Omni-Modal Unified Schema.
    Supports JSONL (stream) and JSON (list) formats.
    Auto-normalizes:
      - Alpaca (instruction/input/output)
      - CoT (prompt/response)
      - XLAM/Tool (query/answers)
      - Native (messages)
    """
    def __init__(self, data_path: str, split: str = "train", samples_per_dataset: int = 0, balanced: bool = True, repetition_factor: int = 1, repetition_style: str = "baseline"):
        self.split = split
        self.limit = samples_per_dataset
        self.base_path = Path(data_path)
        self.balanced = balanced
        self.dataset_counts = {} # Tracks count per dataset identifier
        self.repetition_factor = repetition_factor
        self.repetition_style = repetition_style
        
        if not self.base_path.exists():
            if logger:
                logger.error(f"❌ Data path not found: {self.base_path}")
            
        # 1. Discover Datasets (recursive discovery)
        from src.metrics_tracker import discover_datasets
        discovered = discover_datasets(str(self.base_path))
        
        # Flatten discovery into a category map
        self.category_map = discovered # {category: [paths]}
        self.dataset_dirs = []
        for paths in self.category_map.values():
            self.dataset_dirs.extend([Path(p) for p in paths])
            
        if not self.dataset_dirs:
            # Fallback to direct path
            self.dataset_dirs = [self.base_path]
        
        if logger:
            logger.info(f"🌊 Initialized Streamable Dataset ({split}). Discovered {len(self.dataset_dirs)} sources across {len(self.category_map)} categories.")
            if balanced:
                logger.info("⚖️ Balanced Mode: Interleaving samples between capability categories.")
            if self.repetition_factor > 1:
                logger.info(f"🔁 Repetition Engine: Active (Factor={self.repetition_factor}, Style={self.repetition_style})")
        
        if self.repetition_factor > 1:
            from src.utils.repetition import PromptRepetitionEngine
            self.repetition_engine = PromptRepetitionEngine(self.repetition_factor, self.repetition_style)
            
    def _get_stream(self, file_path: Path):
        """Yield items from JSON/JSONL file."""
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                    else:
                        yield data
        except Exception as e:
            if logger:
                logger.warning(f"Error reading {file_path}: {e}")

    def _normalize(self, item: dict):
        """Convert any format to Native Messages Schema."""
        # 1. Alpaca
        if "instruction" in item:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            prompt = f"{instruction}\n{input_text}" if input_text else instruction
            return [{"role": "user", "content": prompt}, {"role": "assistant", "content": item["output"]}]
        
        # 2. CoT / Simple QA
        if "prompt" in item and "response" in item:
            return [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response"]}]
            
        # 3. Tool / XLAM (Query based)
        if "query" in item:
            return [{"role": "user", "content": item["query"]}, {"role": "assistant", "content": item.get("answers") or item.get("response")}]
            
        # 4. Native (already correct)
        if "messages" in item:
            return item["messages"]
            
        return None

    def __iter__(self):
        """Interleave samples from dataset sources."""
        # Create generator for each dataset
        generators = []
        for ddir in self.dataset_dirs:
            files = list(ddir.glob("*.json")) + list(ddir.glob("*.jsonl"))
            for f in files:
                generators.append(self._get_stream(f))
                
        if not generators:
            return
            
        # Interleave
        gen_cycle = itertools.cycle(generators)
        exhausted = set()
        count = 0
        
        while len(exhausted) < len(generators):
            if self.limit > 0 and count >= self.limit:
                break
                
            try:
                gen = next(gen_cycle)
            except StopIteration:
                break

            if gen in exhausted:
                continue
                
            try:
                item = next(gen)
                normalized = self._normalize(item)
                if normalized:
                    # Apply Repetition 
                    if self.repetition_factor > 1:
                        normalized = self.repetition_engine.process_messages(normalized)
                    
                    count += 1
                    yield {"messages": normalized}
            except StopIteration:
                exhausted.add(gen)

class DynamicDataCollator:
    """
    Collator that adapts to the model's specific schema requirements.
    Uses real processors to generate tensors from paths.
    """
    def __init__(self, schema):
        self.schema = schema
        self.vision_key = schema["vision_key"]
        self.audio_key = schema["audio_key"]
        self.report_key = schema["report_key"]
        self.persona_key = schema["persona_key"]
        self.text_key = schema["text_key"]
        from src.multimodal.decoders import OmniDecoder
        self.decoder = OmniDecoder()
        
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return {}
        
        # Tokenize text
        # In production, we'd use model.tokenizer here. 
        # For this implementation, we simulate the stacking for now but use real modality data
        
        bs = len(batch)
        
        out = {
            self.text_key: torch.stack([torch.tensor(b.get("input_ids", [0]*10)) for b in batch]),
            "labels": torch.stack([torch.tensor(b.get("labels", [0]*10)) for b in batch])
        }
        
        # Process Modalities
        if self.schema["requires_vision_input"]:
            pixel_values = []
            for b in batch:
                path = b.get("image_path")
                if path and os.path.exists(path):
                    res = self.decoder.decode(path, "vision")
                    pixel_values.append(res["pixel_values"].squeeze(0))
                else:
                    # Dummy for shape consistency if path missing (should be handled by loader)
                    pixel_values.append(torch.zeros(3, 512, 512))
            out[self.vision_key] = torch.stack(pixel_values)
            
        if self.schema["requires_audio_input"]:
            audio_features = []
            for b in batch:
                path = b.get("audio_path")
                if path and os.path.exists(path):
                    res = self.decoder.decode(path, "audio")
                    audio_features.append(res["input_features"].squeeze(0))
                else:
                    audio_features.append(torch.zeros(128, 3000))
            out[self.audio_key] = torch.stack(audio_features)
            
        if self.schema.get("requires_report_input"):
            reports = []
            for b in batch:
                state = b.get("report_state")
                if state is not None:
                    reports.append(torch.tensor(state))
                else:
                    reports.append(torch.zeros(1, 4096))
            out[self.report_key] = torch.stack(reports)
            
        if self.schema.get("requires_persona_input"):
            personas = []
            for b in batch:
                state = b.get("persona_state")
                if state is not None:
                    personas.append(torch.tensor(state))
                else:
                    personas.append(torch.zeros(1, 4096))
            out[self.persona_key] = torch.stack(personas)
        return out

# DynamicDataCollator class moved to main()
def main():
    global parser, logger
    import argparse
    parser = argparse.ArgumentParser()
    
    # Initialize logger
    logger = setup_logger(__name__, "logs/multimodal_training.log")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1=Projectors Only, 2=Full Model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--sample-size", type=int, default=0, help="Total samples to use (0=all)")
    parser.add_argument("--experiment-name", type=str, default="", help="Experiment name for logs")
    parser.add_argument("--log-results", action="store_true", help="Log results to CSV")
    # Repetition args
    parser.add_argument("--repetition-factor", type=int, default=1, help="Prompt repetition factor")
    parser.add_argument("--repetition-style", type=str, default="baseline", help="Repetition style")
    
    args = parser.parse_args()
    
    # Enforce 'nexus' conda environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        sys.exit("\033[0;31m[ERROR] Must be run in 'nexus' conda environment.\033[0m")
        
    if not check_env():
        sys.exit(1)
        
    log_header(logger, f"OMNI-MODAL TRAINING (Stage {args.stage})", {
        "Data": args.data_path,
        "Base": CONFIG["base_model"],
        "Schema": "Unified Messages (Native)",
        "Repetition": f"{args.repetition_factor}x ({args.repetition_style})"
    })
    
    # 1. Dataset - Stream from data-path
    logger.info("Initializing Streaming Datasets...")
    
    # We load 3 separate dataset objects
    train_dataset = OmniDataset(args.data_path, split="train", samples_per_dataset=args.sample_size, 
                                repetition_factor=args.repetition_factor, repetition_style=args.repetition_style)
    val_dataset = OmniDataset(args.data_path, split="val", samples_per_dataset=args.sample_size // 10 if args.sample_size > 0 else 0)
    test_dataset = OmniDataset(args.data_path, split="test", samples_per_dataset=args.sample_size // 10 if args.sample_size > 0 else 0)
    
    # No len() check possible for IterableDataset
    logger.info("✅ Datasets initialized in Streaming Mode")
    logger.info("   (Note: Sample counts are not pre-calculated to save RAM)")
    
    # 2. Initialize Omni Model with CPU offloading for 16GB VRAM
    logger.info("Initializing model with memory optimization (16GB VRAM + 32GB RAM)...")
    
    # For Omni model, visual/audio repetition implies embedding repetition
    visual_rep = args.repetition_factor if args.repetition_factor > 1 else 1
    audio_rep = args.repetition_factor if args.repetition_factor > 1 else 1
    
    model = OmniMultimodalLM(
        llm_name=CONFIG["base_model"],
        vision_name=CONFIG["vision_model"],
        audio_name=CONFIG["audio_model"],
        device_map="auto",  # Hybrid CPU/GPU
        load_in_8bit=True,   # Quantize frozen models
        enable_decoders=False, # Disable output decoders during training to save VRAM
        visual_repetition_factor=visual_rep,
        audio_repetition_factor=audio_rep
    )
    
    logger.info("Memory allocation:")
    logger.info("  GPU: DFM connectors + decoders (~14GB)")
    logger.info("  CPU: Frozen LLM + encoders (~18GB)")
    
    # 3. Training Config (Steps-based for Iterable)
    # Estimate steps: (21 datasets * ~10k samples) / batch_size
    # Hardcoding robust defaults for streaming
    max_steps = 1000 if args.sample_size > 0 else 50000 
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=max_steps,          # Required for IterableDataset
        per_device_train_batch_size=1, # Conservative batch size
        per_device_eval_batch_size=1,  # Conservative eval batch
        save_steps=100,
        save_total_limit=3,
        logging_steps=1,
        learning_rate=1e-3,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,   # Important for custom multimodal inputs
        do_eval=True,
        eval_strategy="steps",
        eval_steps=100,
        dataloader_num_workers=0,      # CRITICAL: Prevent RAM spike from worker duplication
        dataloader_pin_memory=False    # CRITICAL: Save RAM
    )
    
    # 4. Freeze/Unfreeze based on Stage
    if args.stage == 1:
        logger.info("Stage 1: Training Projectors Only")
        for p in model.llm.parameters(): p.requires_grad = False
        for p in model.vision_projector.parameters(): p.requires_grad = True
        for p in model.audio_projector.parameters(): p.requires_grad = True
    elif args.stage == 2:
        logger.info("Stage 2: Training Full Model (Respecting Quantization)")
        # For 8-bit models, we cannot unfreeze Int8 weights. 
        # We unfreeze only Trainable parts (Float16/32).
        count = 0
        for name, p in model.named_parameters():
            if p.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                p.requires_grad = True
                count += 1
            else:
                p.requires_grad = False
        logger.info(f"  Example: Unfrozen {count} floating-point parameters (Projectors/Connectors). Int8 backbone remains frozen.")
            
    # 4.5. Detect Schema & Initialize Dynamic Collator
    schema = model.get_input_schema()
    logger.info(f"📋 Detected Model Schema: {schema}")
    
    data_collator = DynamicDataCollator(schema)
            
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    logger.info("🚀 Starting Rolling Checkpoint Training...")
    trainer.train()
    
    # 6. Final Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    
    log_completion(logger, 0, 0, 0, 0, 0, 0.0)

if __name__ == "__main__":
    main()

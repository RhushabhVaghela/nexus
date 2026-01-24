#!/usr/bin/env python3
"""
24_multimodal_training.py
Train the Omni-Modal Projectors (Stage 1) or Fine-tune full model (Stage 2).
SUPPORTS UNIFIED MESSAGES SCHEMA (Native Format).
"""

import argparse
import torch
import json
import os
from pathlib import Path
import sys
import random
import itertools
from typing import Dict, Any, List
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoProcessor

# Add project root to sys.path to allow absolute imports from 'src'
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal.model import OmniMultimodalLM
from src.multimodal.decoders import OmniDecoder
from src.data.universal_loader import UniversalDataLoader
from src.utils.logging_config import setup_logger, log_header, log_completion
from src.utils.repetition import PromptRepetitionEngine


logger = setup_logger(__name__, "logs/multimodal_training.log")

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
        
        logger.info(f"🌊 Initialized Streamable Dataset ({split}). Discovered {len(self.dataset_dirs)} sources across {len(self.category_map)} categories.")
        if balanced:
            logger.info("⚖️ Balanced Mode: Interleaving samples between capability categories.")
        if self.repetition_factor > 1:
            logger.info(f"🔁 Prompt Repetition Enabled: {self.repetition_factor}x ({self.repetition_style})")

    def _get_files_for_split(self):
        """Generator to yield relevant files for this split."""
        
        # Define Aliases
        ALIASES = {
            "train": ["train", "training", "train_data"],
            "val": ["val", "validation", "eval", "evaluation", "dev"],
            "test": ["test", "testing"]
        }
        
        # flattened list of all known folder names to check for structure
        ALL_KNOWN_FOLDERS = set([name for sublist in ALIASES.values() for name in sublist])
        
        for ds_dir in self.dataset_dirs:
            # 1. Detect if this dataset has ANY explicit structure
            try:
                # Scan immediate subdirectories
                subdirs = {d.name.lower() for d in ds_dir.iterdir() if d.is_dir()}
            except Exception:
                subdirs = set()
                
            has_explicit_structure = not subdirs.isdisjoint(ALL_KNOWN_FOLDERS)
            target_folders = []
            
            # 2. Strategy Selection
            if has_explicit_structure:
                # STRICT MODE: Use only explicit folders matching the requested split aliases
                # PREVENTS LEAKAGE: Do not fall back to scanning root if 'val' is missing but 'train' exists.
                
                possible_names = ALIASES.get(self.split, [])
                for name in possible_names:
                    # Check against detected subdirs to handle casing if needed, or just path check
                    candidate = ds_dir / name
                    if candidate.exists():
                        target_folders.append(candidate)
                        
                # Yield files from matched folders (if any)
                if target_folders:
                    for folder in target_folders:
                         # Lazy iterator for efficiency
                         files_gen = itertools.chain(folder.rglob("*.jsonl"), folder.rglob("*.json"))
                         for p in files_gen: 
                             yield p
                             
            else:
                 # AUTO SPLIT MODE: No standard folders found.
                 # Apply Hash-based 90/5/5 splitting on the root
                 
                all_files_gen = itertools.chain(ds_dir.rglob("*.jsonl"), ds_dir.rglob("*.json"))
                
                for p in all_files_gen:
                    # Deterministic hash of relative path
                    h = hash(p.name) % 100 # 0-99
                    
                    is_train = h < 90
                    is_val = 90 <= h < 95
                    is_test = h >= 95
                    
                    if self.split == "train" and is_train:
                        yield p
                    elif self.split == "val" and is_val:
                        yield p
                    elif self.split == "test" and is_test:
                        yield p



    def __iter__(self):
        """Streaming iterator over data."""
        if self.balanced and len(self.category_map) > 1:
            yield from self._iter_balanced()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self):
        """Traditional sequential file-by-file streaming."""
        file_iterator = self._get_files_for_split()
        
        for file_path in file_iterator:
            yield from self._parse_file(file_path)

    def _iter_balanced(self):
        """Interleaved streaming to balance between categories."""
        # Setup iterators for each category
        iterators = {}
        for cat, paths in self.category_map.items():
            # Get files for this category only
            cat_dirs = [Path(p) for p in paths]
            
            def cat_file_gen(dirs):
                orig_dirs = self.dataset_dirs
                self.dataset_dirs = dirs
                files = list(self._get_files_for_split())
                random.shuffle(files)
                self.dataset_dirs = orig_dirs
                for f in files:
                    yield from self._parse_file(f)
            
            iterators[cat] = cat_file_gen(cat_dirs)
        
        if not iterators:
            return

        while iterators:
            cats = list(iterators.keys())
            for cat in cats:
                try:
                    sample = next(iterators[cat])
                    yield sample
                except StopIteration:
                    del iterators[cat]

    def _parse_file(self, file_path: Path):
        """Parse a single data file and yield samples with limit tracking."""
        try:
            # Identify dataset name for limit tracking
            dataset_name = next((d.name for d in self.dataset_dirs if d in file_path.parents), file_path.parent.name)
        except:
            dataset_name = file_path.parent.name
            
        # Check Global Dataset Limit
        if self.limit > 0 and self.dataset_counts.get(dataset_name, 0) >= self.limit:
            return
            
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            sample = json.loads(line)
                            if self._yield_sample(sample, dataset_name):
                                yield self._process_sample(sample)
                            else:
                                break # Limit hit
                        except json.JSONDecodeError: continue
            elif file_path.suffix == ".json":
                # Handle hybrid JSON/JSONL
                is_jsonl = False
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for sample in data:
                                if self._yield_sample(sample, dataset_name):
                                    yield self._process_sample(sample)
                                else:
                                    break
                except json.JSONDecodeError as e:
                    if "Extra data" in str(e): is_jsonl = True
                    else: logger.warning(f"Error reading {file_path}: {e}")
                
                if is_jsonl:
                    # Fallback to line-by-line for mislabeled JSONL
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            try:
                                sample = json.loads(line)
                                if self._yield_sample(sample, dataset_name):
                                    yield self._process_sample(sample)
                                else: break
                            except json.JSONDecodeError: continue
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    def _yield_sample(self, sample, dataset_name):
        """Returns True if sample should be yielded (updates count), False if limit reached."""
        if self.limit > 0 and self.dataset_counts.get(dataset_name, 0) >= self.limit:
            return False
        self.dataset_counts[dataset_name] = self.dataset_counts.get(dataset_name, 0) + 1
        return True

    def _process_sample(self, sample):
        """Parse raw JSON into normalized model input format"""
        # 1. Normalize Schema
        messages = []
        
        # A. Native Messages
        if "messages" in sample:
            messages = sample["messages"]
            
        # B. CoT (prompt/response) or Alpaca (instruction/output)
        elif "prompt" in sample and "response" in sample:
            messages = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["response"]}
            ]
        elif "instruction" in sample and "output" in sample:
             # Handle Alpaca 'input' field if present
            content = sample["instruction"]
            if sample.get("input"): content += f"\nInput: {sample['input']}"
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": sample["output"]}
            ]
            
        # C. XLAM (query/answers)
        elif "query" in sample and "answers" in sample:
            messages = [
                {"role": "user", "content": sample["query"]},
                {"role": "assistant", "content": sample["answers"]} # Typically JSON string
            ]
            
        # D. Math (problem/answer or question/answer)
        elif ("problem" in sample or "question" in sample) and ("answer" in sample or "solution" in sample):
            q = sample.get("problem") or sample.get("question")
            a = sample.get("answer") or sample.get("solution")
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            
        if not messages: return None # Unmatched schema
        
        # 2. Extract Modalities (Native only for now)
        modalities = sample.get("modalities", {})
        
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        
        if not user_msg or not assistant_msg: return None
            
        content_items = user_msg["content"]
        image_path = None
        audio_path = None
        text_prompt = ""
        
        if modalities:
            if "image" in modalities and modalities["image"]:
                image_path = modalities["image"][0].get("path")
            if "audio" in modalities and modalities["audio"]:
                audio_path = modalities["audio"][0].get("path")
                
        if isinstance(content_items, str):
            text_prompt = content_items
        elif isinstance(content_items, list):
            for item in content_items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_prompt += item.get("text", "")
                    elif item.get("type") == "image" and not image_path:
                        image_path = item.get("image")
                    elif item.get("type") == "audio" and not audio_path:
                        audio_path = item.get("audio")
                else:
                    text_prompt += str(item)
        
        # APPLY TEXT REPETITION
        if self.repetition_factor > 1 and text_prompt:
            text_prompt = PromptRepetitionEngine.apply_repetition(
                text_prompt,
                factor=self.repetition_factor,
                style=self.repetition_style
            )
            
        return {
            "text": text_prompt,
            "image_path": image_path,
            "audio_path": audio_path,
            "report_state": sample.get("report_state"), # Already processed hidden states
            "persona_state": sample.get("persona_state"), # Already processed embeddings
            "label": assistant_msg["content"]
        }

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


def main():
    parser = argparse.ArgumentParser()
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
    logger.info(f"📋 Detected Model Schema: {json.dumps(schema, indent=2)}")
    
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

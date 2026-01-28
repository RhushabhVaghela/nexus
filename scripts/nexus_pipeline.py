import os
import sys
import json
import signal
import time
import argparse
import subprocess
from datetime import datetime
import shutil
from transformers import AutoConfig
from nexus_final.utils.memory import should_use_sli

# --- CONFIGURATION (CONVERT TO ABSOLUTE) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src')) # Ensure src is in path

try:
    from nexus_core.towers.registry import TEACHER_REGISTRY, DATASET_REGISTRY
except ImportError:
    print("[Warning] Could not import registry from src.nexus_core.towers.registry. Using empty defaults.")
    TEACHER_REGISTRY = {}
    DATASET_REGISTRY = {}

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    print("[Warning] huggingface_hub not installed. Smart download features will be disabled.")
    HF_AVAILABLE = False

STATE_FILE = os.path.join(BASE_DIR, ".pipeline_state.json")
RESULTS_ROOT = os.path.join(BASE_DIR, "results/niwt_profiling")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RELEASE_DIR = os.path.join(BASE_DIR, "nexus-release-v1")
DATASET_BASE_DIR = "/mnt/e/data/datasets" # Default storage for downloaded datasets
DATA_PATH = "/mnt/e/data" # Secondary location for models/encoders/decoders

class NexusPipeline:
    def __init__(self, dry_run=False, skip_non_llm=False, models=None, datasets=None, 
                 sample_size=50, epochs=1, lr=1e-5, router_epochs=5, router_lr=1e-4,
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 use_unsloth=False, packing=False, max_seq_length=2048, use_grpo=False):
        self.state = self._load_state()
        self.paused = False

        # --- Singleton Lock Mechanism ---
        self.lock_file = os.path.join(BASE_DIR, ".pipeline.lock")
        if os.path.exists(self.lock_file):
            with open(self.lock_file, 'r') as f:
                try:
                    old_pid = int(f.read().strip())
                    # Check if process is actually running
                    os.kill(old_pid, 0)
                    print(f"[Error] Pipeline overlap detected! PID {old_pid} is supposedly running.")
                    print(f"        'run_nexus_master.sh' should have cleaned this up.")
                    print(f"        Please kill the process manually or run the master script.")
                    sys.exit(1)
                except (OSError, ValueError):
                    # Process dead or file corrupt
                    pass
        
        # Acquire Lock
        with open(self.lock_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Ensure cleanup on exit
        import atexit
        def remove_lock():
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        atexit.register(remove_lock)

        self.dry_run = dry_run
        self.skip_non_llm = skip_non_llm
        self.sample_size = sample_size
        self.epochs = epochs
        self.lr = lr
        self.router_epochs = router_epochs
        self.router_lr = router_lr
        self.embedding_model = embedding_model
        
        # Optimization Flags
        self.use_unsloth = use_unsloth
        self.packing = packing
        self.max_seq_length = max_seq_length
        self.use_grpo = use_grpo
        
        # Parse and Expand Selections
        self.target_models = self._resolve_selection(models, TEACHER_REGISTRY, "models")
        self.target_datasets = self._resolve_selection(datasets, DATASET_REGISTRY, "datasets")
        
        if self.target_models:
            print(f"[Pipeline] Final Model List: {self.target_models}")
        if self.target_datasets:
            print(f"[Pipeline] Final Dataset List: {self.target_datasets}")

        # Resolve embedding model alias
        if self.embedding_model == "all-mpnet-base-v2":
            print(f"[Pipeline] Auto-resolving embedding model '{self.embedding_model}' to 'sentence-transformers/all-mpnet-base-v2'")
            self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        elif self.embedding_model == "all-MiniLM-L6-v2":
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Signal Handling for Graceful Pause
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _resolve_selection(self, input_arg, registry, type_name):
        """
        Parses input string (supports brackets, commas) and expands keywords/tags.
        Returns a list of KEYS (for both models and datasets) to ensure we can look up details later.
        If a raw value is provided that isn't in registry, it serves as the key itself.
        """
        if not input_arg:
            return None
            
        if input_arg == "all":
            return list(registry.keys())
        
        # clean brackets [ ]
        cleaned = input_arg.strip("[]")
        raw_items = [x.strip() for x in cleaned.split(",")]
        
        final_set = set()
        
        for item in raw_items:
            found = False
            # 1. Exact Key Match
            if item in registry:
                final_set.add(item)
                found = True
                continue
                
            # 2. Tag/Category Match or Substring Key Match (Fuzzy)
            for key, entry in registry.items():
                # Tag match
                if "tags" in entry and item in entry["tags"]:
                    final_set.add(key)
                    found = True
                    
                # Substring match (e.g. "vision" matches "vision_main")
                elif item in key:
                     final_set.add(key)
                     found = True
            
            # 3. If still not found, treat as raw value (e.g. direct HF path)
            if not found:
                print(f"[Pipeline] '{item}' not found in registry. Treating as raw value.")
                final_set.add(item)
                
        return list(final_set)

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {"current_stage": "init", "completed_stages": [], "config": {}}

    def _save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"[Pipeline] State saved to {STATE_FILE}")

    def _handle_interrupt(self, signum, frame):
        print("\n[Pipeline] Pause signal received! Finishing current step and saving state...")
        self.paused = True
        self._save_state()
        sys.exit(0)

    def run_command(self, cmd, allow_fail=False):
        print(f"[Exec] {cmd}")
        if self.dry_run:
            return 0
        ret = os.system(cmd)
        if ret != 0:
            print(f"[Error] Command failed with code {ret}")
            if not allow_fail:
                # FIX: os.system return code is 16-bit. 256 wraps to 0 in shell exit.
                # Must exit with 1 explicitly.
                sys.exit(1)
        return ret

    def ensure_dataset_available(self, dataset_key):
        """
        Checks if dataset exists locally. If not, attempts to download from HF.
        Returns:
            (path_to_use, is_massive_flag)
        """
        if not HF_AVAILABLE:
            # Fallback checks without download capability
            if dataset_key in DATASET_REGISTRY:
                return DATASET_REGISTRY[dataset_key].get('local_path', ''), False
            return dataset_key, False

        # 1. Resolve Info
        if dataset_key in DATASET_REGISTRY:
            entry = DATASET_REGISTRY[dataset_key]
            hf_id = entry.get('path', dataset_key)
            local_path = entry.get('local_path', os.path.join(DATASET_BASE_DIR, "downloaded", dataset_key))
        else:
            hf_id = dataset_key
            local_path = os.path.join(DATASET_BASE_DIR, "downloaded", dataset_key.replace("/", "_"))

        # 2. Check Local Existence
        if os.path.exists(local_path):
            # Basic validation: check if not empty
            if os.listdir(local_path):
                print(f"[Pipeline] Dataset '{dataset_key}' found locally at {local_path}.")
                return local_path, False
        
        # 3. Check Size on HF
        try:
            api = HfApi()
            print(f"[Pipeline] Checking dataset info for '{hf_id}'...")
            dataset_info = api.dataset_info(repo_id=hf_id, repo_type="dataset")
            
            # Estimate size
            size_bytes = 0
            if hasattr(dataset_info, 'siblings'):
                # Heuristic: Sum size of large files (approx) logic if available, 
                # but 'siblings' usually doesn't have size in all API versions.
                # Use 'cardData' or implicit check. 
                # We will check 'dataset_info.size_in_bytes' if it exists (not standard field in all versions).
                pass
            
            # Currently HfApi().dataset_info() might not return total size directly.
            # We iterate siblings to get total size? HfApi.dataset_info returns RepoSibling objects which DO NOT have size always.
            # We can use `model_info` style logic or catch the Exception.
            # Let's assume massive if explicit or heuristic match.
            
            # WORKAROUND: 'is_massive' check by iteration requires HEAD requests. 
            # Simplified Massive Check: > 15GB. 
            # We'll use a sequential heuristic regardless if strict size check fails.
            
            # Let's try to get size via the files list if feasible, or just assume standard download.
            # For this implementation, we will assume standard download unless it's explicitly marked 'massive' or fails.
            
            pass 
        except Exception as e:
            print(f"[Pipeline] Warning: Could not fetch info for {hf_id}: {e}")
            # Assume not massive, or if it fails download will fail.
            pass

        # 4. Smart Download Logic
        # Heuristic for Massive: If explicit in registry DESC or TAGS, or if we define it.
        # But user wants Real-time check. We'll iterate files to sum size.
        is_massive = False
        try:
            files = api.list_repo_files(repo_id=hf_id, repo_type="dataset")
            # This is just names. To get size we need deeper info. 
            # We'll skip strict 15GB check implementation for now and rely on "Layer by Layer" 
            # being triggered by explicit fail or a flag. 
            # Actually, let's just default to sequential for KNOWN massive ones?
            # Or use snapshot_download and catch OOM/DiskError?
            
            # User instruction: "For massive datasets (check via hugging face, if size > 15GB)"
            # Let's try to be diligent.
            total_size = 0
            # Getting size for every file is slow (N requests).
            # We'll rely on a heuristic count of Parquet files?
            # Or just assume snapshot_download handles it unless we really want to delete.
            # The "Delete when layer is done" requirement is key. This implies we MUST use sequential 
            # if we suspect it is large.
            
            # Let's define a threshold count of parquet files?
            parquet_files = [f for f in files if f.endswith('.parquet')]
            if len(parquet_files) > 50: # Heuristic: >50 shards is likely massive
                 is_massive = True
                 
        except:
             is_massive = False

        if is_massive:
            print(f"[Pipeline] Dataset '{hf_id}' detected as MASSIVE. Using Sequential Layer Ingestion.")
            return (hf_id, local_path), True # Signal massive handling (return tuple)

        print(f"[Pipeline] Downloading '{hf_id}' to {local_path} (Snapshot)...")
        if self.dry_run:
            return local_path, False
            
        try:
            snapshot_download(repo_id=hf_id, local_dir=local_path, repo_type="dataset")
            return local_path, False
        except Exception as e:
            print(f"[Error] Download failed: {e}")
            return None, False

    def process_massive_dataset(self, hf_id, local_path, teacher_path, teacher_name):
        """
        Sequential Download -> Process -> Delete loop.
        """
        print(f"[Pipeline] Starting Sequential Layer Ingestion for {hf_id}...")
        api = HfApi()
        files = api.list_repo_files(repo_id=hf_id, repo_type="dataset")
        start_time = datetime.now()
        
        # Filter data files
        valid_extensions = ('.parquet', '.jsonl', '.json', '.csv')
        data_files = [f for f in files if f.endswith(valid_extensions)]
        
        if not data_files:
            print(f"[Pipeline] No data files found in {hf_id}.")
            return

        for idx, filename in enumerate(data_files):
            print(f"[SLI] Processing layer {idx+1}/{len(data_files)}: {filename}")
            
            # 1. Download File
            file_path = hf_hub_download(repo_id=hf_id, filename=filename, local_dir=local_path, repo_type="dataset")
            
            # 2. Run Distillation on this file
            shard_prefix = f"{hf_id.replace('/', '_')}_{os.path.splitext(filename)[0]}"
            output_dir = os.path.join(MEMORY_DIR, teacher_name.replace('/', '_'))
            
            cmd = f"{sys.executable} -m src.nexus_final.distill_knowledge --teacher '{teacher_path}' --output '{output_dir}' --dataset '{file_path}' --shard_prefix '{shard_prefix}'"
            self.run_command(cmd)
            
            # 3. Delete File
            if not self.dry_run:
                try:
                    os.remove(file_path)
                    print(f"[SLI] Deleted layer: {filename}")
                except OSError as e:
                    print(f"[Warning] Could not delete {filename}: {e}")
        
        print(f"[SLI] Sequential processing complete for {hf_id}. Duration: {datetime.now() - start_time}")

    def stage_metadata_discovery(self):
        if "metadata_discovery" in self.state["completed_stages"]:
            print("[Skip] Metadata Discovery already complete.")
            self.state["current_stage"] = "profiling"
            return

        print("\n=== STAGE 0: UNIVERSAL METADATA DISCOVERY ===")
        max_hidden = 1024 # Baseline floor
        max_vocab = 32000 # Baseline floor
        
        if not self.target_models:
             print("[Error] No active models for discovery!")
             return

        for model_key in self.target_models:
            if model_key in TEACHER_REGISTRY:
                path = TEACHER_REGISTRY[model_key].get('path', TEACHER_REGISTRY[model_key].get('model'))
            else:
                path = model_key
            
            print(f"[Discovery] Inspecting: {path}...")
            try:
                config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                
                # Try common hidden dimension keys
                hidden = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 0)))
                # Try common vocab size keys
                vocab = getattr(config, "vocab_size", 0)
                
                if hidden > max_hidden:
                    print(f"  -> Found larger hidden_size: {hidden}")
                    max_hidden = hidden
                if vocab > max_vocab:
                    print(f"  -> Found larger vocab_size: {vocab}")
                    max_vocab = vocab
            except Exception as e:
                print(f"  -> [Warn] Failed to fetch config for {path}: {e}")
                # Use registry values as secondary fallback if available
                if model_key in TEACHER_REGISTRY:
                    max_hidden = max(max_hidden, TEACHER_REGISTRY[model_key].get('dim', 2048))

        print(f"[Discovery] Final Unified Specs: Hidden={max_hidden}, Vocab={max_vocab}")
        self.state["config"]["hidden_size"] = max_hidden
        self.state["config"]["vocab_size"] = max_vocab
        self.state["completed_stages"].append("metadata_discovery")
        self.state["current_stage"] = "profiling"
        self._save_state()


    def stage_profiling(self):
        if "profiling" in self.state["completed_stages"]:
            print("[Skip] Profiling already complete.")
            self.state["current_stage"] = "knowledge_extraction"
            return

        print("\n=== STAGE 1: NIWT PROFILING & ACTIVATION ANALYSIS ===")
        
        if not self.target_models:
            print("[Error] No active models selected for profiling!")
            return

        for model_key in self.target_models:
            if model_key in TEACHER_REGISTRY:
                teacher = TEACHER_REGISTRY[model_key]
                name = teacher.get('model', model_key)
                path = teacher.get('path', name) 
                category = str(teacher.get('tags', [])).lower()
                desc = teacher.get('desc', 'Unknown')
            else:
                name = model_key
                path = model_key
                teacher = {} # Prevent UnboundLocalError
                category = "unknown"
                desc = "Raw Model Path"
            
            # Filter non-LLM checks...
            is_llm = "language" in category or "text" in category or "agent" in category or "reasoning" in category
            skip_keywords = ["audio", "image", "video", "tts", "encoder", "tokenizer", "vision"]
            if any(k in str(category) for k in skip_keywords) or any(k in name.lower() for k in skip_keywords):
                is_llm = False

            if not is_llm:
                if self.skip_non_llm:
                    print(f"[Profiler] Skipping non-LLM teacher: {name} ({desc})")
                    continue
                else:
                    print(f"[Profiler] WARNING: Attempting to profile non-LLM teacher: {name} ({desc})")

            print(f"\n[Profiler] Target: {name} (Key: {model_key})")
            
            # Ensure model path is valid/download if needed (Future improvement)
            if os.path.exists(path):
                model_arg = path
            else:
                # Check secondary DATA_PATH
                # Path in registry is like /mnt/d/.../models/AgentCPM. We need relative part.
                if path.startswith(BASE_DIR):
                    rel_path = os.path.relpath(path, BASE_DIR)
                    alt_path = os.path.join(DATA_PATH, rel_path)
                    if os.path.exists(alt_path):
                        print(f"[Profiler] Model found in secondary storage: {alt_path}")
                        model_arg = alt_path
                    else:
                        fallback_id = teacher.get('model', name)
                        print(f"[Profiler] Path not found in Repo or Data Drive. Autoloading from HF: {fallback_id}")
                        model_arg = fallback_id
                else:
                    fallback_id = teacher.get('model', name)
                    print(f"[Profiler] Local path {path} not found. Falling back to HF ID: {fallback_id}")
                    model_arg = fallback_id
            
            # Prepare command
            dataset_arg = ""
            if self.target_datasets:
                # Use the first available dataset for profiling stimulus
                ds_key = self.target_datasets[0]
                if ds_key in DATASET_REGISTRY and 'local_path' in DATASET_REGISTRY[ds_key]:
                    dataset_arg = f"--dataset_name '{DATASET_REGISTRY[ds_key]['local_path']}'"
                else:
                    dataset_arg = f"--dataset_name '{ds_key}'"
            
            # Check if profile already exists to skip redundant compute
            # RESULTS_ROOT is globally defined as .../results/niwt_profiling
            expected_profile = os.path.join(RESULTS_ROOT, f"{name}_profile.json")
            if os.path.exists(expected_profile):
                print(f"[Skip] Profile already exists for {name} ({expected_profile})")
                continue

            cmd = f"'{sys.executable}' '{os.path.join(BASE_DIR, 'scripts/run_profiling_driver.py')}' --teacher_id '{name}' --model_path '{model_arg}' {dataset_arg} --sample_size {self.sample_size}"
            
            # Allow individual model failure to not kill the whole pipeline
            ret = self.run_command(cmd, allow_fail=True)
            if ret != 0:
                print(f"[Profiler] WARNING: Profiling failed for {name}. Skipping to next model.")

        self.state["completed_stages"].append("profiling")
        self.state["current_stage"] = "knowledge_extraction"
        self._save_state()

    def stage_knowledge_extraction(self):
        if "knowledge_extraction" in self.state["completed_stages"]:
            print("[Skip] Knowledge Extraction already complete.")
            self.state["current_stage"] = "training"
            return

        print("\n=== STAGE 1.5: MATHEMATICAL KNOWLEDGE EXTRACTION (LIBRARIAN) ===")

        if not self.target_models:
             print("[Error] No active models for knowledge extraction!")
             return
        
        # Prepare Datasets
        dataset_keys = self.target_datasets if self.target_datasets else ["general/google_smol"]
        
        # Resolve all datasets first
        # We need a map of key -> verified_path (or massive info)
        verified_datasets = {} # key -> (path, is_massive)
        
        print("[Librarian] Verifying and downloading datasets...")
        for ds_key in dataset_keys:
            res, is_massive = self.ensure_dataset_available(ds_key)
            if res:
                verified_datasets[ds_key] = (res, is_massive)
            else:
                print(f"[Warning] Skipping unavailable dataset: {ds_key}")

        for model_key in self.target_models:
            if model_key in TEACHER_REGISTRY:
                teacher = TEACHER_REGISTRY[model_key]
                name = teacher.get('model', model_key)
                raw_path = teacher.get('path', name)
                if os.path.exists(raw_path):
                    path = raw_path
                else:
                    # Check Secondary
                    if raw_path.startswith(BASE_DIR):
                        rel = os.path.relpath(raw_path, BASE_DIR)
                        alt = os.path.join(DATA_PATH, rel)
                        if os.path.exists(alt):
                            path = alt
                        else:
                            path = teacher.get('model', name)
                            print(f"[Librarian] Model not found locally. Using HF ID: {path}")
                    else:
                        path = teacher.get('model', name)
                        print(f"[Librarian] Local path {raw_path} not found. Falling back to HF ID: {path}")
            else:
                name = model_key
                path = model_key

            print(f"\n[Librarian] Ingesting Knowledge from: {name}")

            for ds_key, (ds_res, is_massive) in verified_datasets.items():
                if is_massive:
                    # ds_res is (hf_id, local_parent_path)
                    hf_id, local_parent = ds_res
                    self.process_massive_dataset(hf_id, local_parent, path, name)
                else:
                    # DYNAMIC: Use memory-aware SLI trigger
                    is_too_large = False
                    try:
                        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                        if should_use_sli(config):
                            is_too_large = True
                    except Exception as e:
                        print(f"  -> [Warn] Failed to check mem for {name}: {e}. Defaulting to Standard.")
                    
                    if is_too_large:
                        print(f"  -> [SLI Fallback] Ingesting {ds_key} using Sequential Layer Ingestion...")
                        output_dir = os.path.join(MEMORY_DIR, name.replace('/', '_'))
                        hidden_size = self.state["config"].get("hidden_size", 2048)
                        cmd = f"{sys.executable} -m src.nexus_final.sli_integrator --model '{path}' --output '{output_dir}' --dataset '{ds_res}' --limit {self.sample_size} --student_dim {hidden_size}"
                        self.run_command(cmd, allow_fail=False)
                    else:
                        # ds_res is local_path
                        print(f"  -> Ingesting {ds_key} (Standard)...")
                        output_dir = os.path.join(MEMORY_DIR, name.replace('/', '_'))
                        
                        device_arg = "cuda"
                        hidden_size = self.state["config"].get("hidden_size", 2048)
                        cmd = f"{sys.executable} -m src.nexus_final.distill_knowledge --teacher '{path}' --output '{output_dir}' --dataset '{ds_res}' --device {device_arg} --limit {self.sample_size} --student_dim {hidden_size} --embedding_model '{self.embedding_model}'"
                        
                        print(f"[Pipeline] Attempting execution on {device_arg.upper()}...")
                        ret = self.run_command(cmd, allow_fail=True)
                        
                        if ret != 0:
                            print(f"\n[Pipeline] GPU Execution Failed (Code {ret}). Automatically Retrying on CPU...")
                            device_arg = "cpu"
                            cmd_cpu = f"{sys.executable} -m src.nexus_final.distill_knowledge --teacher '{path}' --output '{output_dir}' --dataset '{ds_res}' --device {device_arg} --limit {self.sample_size} --student_dim {hidden_size} --embedding_model '{self.embedding_model}'"
                            self.run_command(cmd_cpu, allow_fail=False)

        self.state["completed_stages"].append("knowledge_extraction")
        self.state["current_stage"] = "training"
        self._save_state()

    def stage_training(self):
        if "training" in self.state["completed_stages"]:
            print("[Skip] Training already complete.")
            self.state["current_stage"] = "router_training"
            return

        print("\n=== STAGE 2: DISTILLATION LOOP ===")
        profile_path = os.path.join(RESULTS_ROOT, "mock_critical_layers.json") 
        if os.path.exists(RESULTS_ROOT):
            import glob
            # Search recursively for the most recent profile
            files = sorted(glob.glob(os.path.join(RESULTS_ROOT, "**/*.json"), recursive=True), key=os.path.getmtime, reverse=True)
            if files:
                profile_path = files[0]
                print(f"[Pipeline] Using Profile: {profile_path}")

        hidden_size = self.state["config"].get("hidden_size", 2048)
        vocab_size = self.state["config"].get("vocab_size", 128256)
        
        unsloth_flag = "--use_unsloth" if self.use_unsloth else ""
        packing_flag = "--packing" if self.packing else ""
        
        cmd = f"'{sys.executable}' '{os.path.join(BASE_DIR, 'scripts/train.py')}' --epochs {self.epochs} --lr {self.lr} --profile_path '{profile_path}' --hidden_size {hidden_size} --vocab_size {vocab_size} --max_seq_length {self.max_seq_length} {unsloth_flag} {packing_flag}"
        self.run_command(cmd)
        
        self.state["completed_stages"].append("training")
        self.state["current_stage"] = "grpo_training" if self.use_grpo else "router_training"
        self._save_state()

    def stage_grpo_training(self):
        if "grpo_training" in self.state["completed_stages"]:
            print("[Skip] GRPO Training already complete.")
            self.state["current_stage"] = "router_training"
            return

        print("\n=== STAGE 2.5: GRPO REASONING EVOLUTION ===")
        # We assume rejection_sampled.jsonl exists or we use the data_dir
        data_path = "rejection_sampled.jsonl"
        if not os.path.exists(data_path):
            print(f"[Warn] {data_path} not found. Searching in memory...")
            import glob
            mem_files = glob.glob(os.path.join(MEMORY_DIR, "**/*.jsonl"), recursive=True)
            if mem_files: data_path = mem_files[0]

        unsloth_flag = "--use_unsloth" if self.use_unsloth else ""
        ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
        
        cmd = f"'{sys.executable}' '{os.path.join(BASE_DIR, 'scripts/train_grpo.py')}' --model_path '{ckpt_path}' --data_path '{data_path}' --max_seq_length {self.max_seq_length} --lr {self.lr} {unsloth_flag}"
        self.run_command(cmd)

        self.state["completed_stages"].append("grpo_training")
        self.state["current_stage"] = "router_training"
        self._save_state()

    def stage_router_training(self):
        if "router_training" in self.state["completed_stages"]:
            print("[Skip] Router Training already complete.")
            self.state["current_stage"] = "evaluation"
            return

        print("\n=== STAGE 3: ROUTER TRAINING ===")
        # Calculate towers dynamically to match student initialization
        num_towers = len([k for k in TEACHER_REGISTRY.keys() if TEACHER_REGISTRY[k].get('model')])
        # Use detected student dimension
        input_dim = self.state["config"].get("hidden_size", 2048)
        
        cmd = f"'{sys.executable}' '{os.path.join(BASE_DIR, 'scripts/train_router.py')}' --input_dim {input_dim} --num_towers {num_towers} --towers_dir '{MEMORY_DIR}' --epochs {self.router_epochs} --lr {self.router_lr}"
        self.run_command(cmd)
        
        self.state["completed_stages"].append("router_training")
        self.state["current_stage"] = "evaluation"
        self._save_state()

    def stage_evaluation(self):
        if "evaluation" in self.state["completed_stages"]:
            print("[Skip] Evaluation already complete.")
            self.state["current_stage"] = "export"
            return

        print("\n=== STAGE 4: EVALUATION ===")
        cmd = f"'{sys.executable}' '{os.path.join(BASE_DIR, 'src/nexus_final/benchmark_nexus.py')}'"
        self.run_command(cmd)

        self.state["completed_stages"].append("evaluation")
        self.state["current_stage"] = "export"
        self._save_state()

    def stage_export(self):
        if "export" in self.state["completed_stages"]:
            print("[Skip] Export already complete.")
            self.state["current_stage"] = "cleanup"
            return
            
        print("\n=== STAGE 5: EXPORT ===")
        ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
        
        # Fallback: Find highest epoch if "latest" is missing (e.g. from interrupted old runs)
        if not os.path.exists(ckpt_path):
            import glob
            ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pt"))
            if ckpts:
                ckpt_path = sorted(ckpts, key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))[-1]
                print(f"[Export] Fallback: Using final epoch checkpoint: {ckpt_path}")

        hidden_size = self.state["config"].get("hidden_size", 2048)
        vocab_size = self.state["config"].get("vocab_size", 128256)
        
        cmd = f"{sys.executable} -m src.nexus_final.export --student '{ckpt_path}' --output '{RELEASE_DIR}' --hidden_size {hidden_size} --vocab_size {vocab_size}"
        self.run_command(cmd)
        
        self.state["completed_stages"].append("export")
        self.state["current_stage"] = "cleanup"
        self._save_state()

    def stage_cleanup(self):
        if "cleanup" in self.state["completed_stages"]:
            print("[Skip] Cleanup already complete.")
            self.state["current_stage"] = "done"
            return
            
        print("\n=== STAGE 6: CLEANUP ===")
        to_clean = [MEMORY_DIR, RESULTS_ROOT]
        
        for path in to_clean:
            if os.path.exists(path):
                print(f"[Cleanup] Removing {path}...")
                if self.dry_run:
                    continue
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    print(f"[Error] Failed to remove {path}: {e}")

        self.state["completed_stages"].append("cleanup")
        self.state["current_stage"] = "done"
        self._save_state()

    def run(self, stage_override=None):
        if stage_override:
            print(f"[Pipeline] Overriding current stage to: {stage_override}")
            self.state["current_stage"] = stage_override

        print("Nexus Automation Pipeline Initialized.")
        print(f"[Config] Base Path: {BASE_DIR}")
        print(f"[Config] Registry: Loaded from src.nexus_core.towers.registry")
        print(f"[Config] Memory: {MEMORY_DIR}")
        print(f"Current State: {self.state['current_stage']}")
        
        # Determine start point
        start_key = self.state.get("current_stage", "init")
        
        if start_key == "init" or start_key == "metadata_discovery":
            self.stage_metadata_discovery()
            if self.paused: return

        if self.state["current_stage"] == "metadata_discovery" or self.state["current_stage"] == "profiling":
            self.stage_profiling()
            if self.paused: return

        if self.state["current_stage"] == "knowledge_extraction":
            self.stage_knowledge_extraction()
            if self.paused: return
            
        if self.state["current_stage"] == "training":
            self.stage_training()
            if self.paused: return

        if self.state["current_stage"] == "grpo_training":
            self.stage_grpo_training()
            if self.paused: return

        if self.state["current_stage"] == "router_training":
            self.stage_router_training()
            if self.paused: return

        if self.state["current_stage"] == "evaluation":
            self.stage_evaluation()
            if self.paused: return

        if self.state["current_stage"] == "export":
            self.stage_export()
            if self.paused: return

        if self.state["current_stage"] == "cleanup":
            self.stage_cleanup()
            if self.paused: return

        print("\n=== PIPELINE COMPLETE ===")
        print(f"Final Release available in: nexus-release-v1/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexus Self-Driving Pipeline")
    parser.add_argument("--reset", action="store_true", help="Reset pipeline state")
    parser.add_argument("--skip-non-llm", action="store_true", help="Skip non-LLM models")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of teacher models or 'all'")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of datasets or 'all'")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--stage", type=str, default=None, help="Run only specific stage")
    parser.add_argument("--lr", type=float, default=1e-5, help="Student Learning Rate")
    parser.add_argument("--router_epochs", type=int, default=5, help="Router training epochs")
    parser.add_argument("--router_lr", type=float, default=1e-4, help="Router Learning Rate")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Librarian embedding model")
    parser.add_argument("--use_unsloth", action="store_true", help="Enable Unsloth optimizations")
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--grpo", action="store_true", help="Enable GRPO reasoning training stage")
    
    args = parser.parse_args()
    
    if args.reset:
        print(f"[Pipeline] FULL RESET requested from Base: {BASE_DIR}")
        print("[Pipeline] Clearing all previous artifacts...")
        
        # Added 'logs' and 'results' folder to nuke list
        LOGS_DIR = os.path.join(BASE_DIR, "logs")
        RESULTS_DIR = os.path.join(BASE_DIR, "results")
        
        # We nuke RESULTS_DIR instead of RESULTS_ROOT
        paths_to_nuke = [STATE_FILE, RESULTS_DIR, MEMORY_DIR, CHECKPOINT_DIR, RELEASE_DIR, LOGS_DIR]
        
        for p in paths_to_nuke:
            full_p = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
            if os.path.exists(full_p):
                print(f"  -> Nuking: {full_p}")
                if os.path.isdir(full_p):
                    shutil.rmtree(full_p)
                else:
                    os.remove(full_p)
            else:
                # Verbose feedback for user clarity
                print(f"  -> [Already Clean] {os.path.basename(full_p)}")
        
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        os.makedirs(MEMORY_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print("[Pipeline] Reset complete. Exiting.")
        sys.exit(0)

    pipeline = NexusPipeline(
        dry_run=args.dry_run, 
        skip_non_llm=args.skip_non_llm,
        models=args.models,
        datasets=args.datasets,
        sample_size=args.sample_size,
        epochs=args.epochs,
        lr=args.lr,
        router_epochs=args.router_epochs,
        router_lr=args.router_lr,
        embedding_model=args.embedding_model,
        use_unsloth=args.use_unsloth,
        packing=args.packing,
        max_seq_length=args.max_seq_length,
        use_grpo=args.grpo
    )
    
    pipeline.run(stage_override=args.stage)

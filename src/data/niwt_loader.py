import logging
from typing import Iterator, List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from src.data.universal_loader import UniversalDataLoader
from src.utils.schema_normalizer import SchemaNormalizer

logger = logging.getLogger(__name__)

class NIWTDataLoader:
    """
    High-performance, switchable data loader for the NIWT pipeline.
    Supports batched streaming and capability-based loading.
    """
    
    CAPABILITY_MAP = {
        "reasoning": ["openai_gsm8k", "cais_mmlu", "math"],
        "agentic": ["GAIA", "tools", "gorilla-llm_Berkeley-Function-Calling-Leaderboard"],
        "vision": ["visual_genome", "coco", "LucasFang_JourneyDB-GoT"],
        "coding": ["MiniMaxAI_OctoCodingBench", "princeton-nlp_SWE-bench"]
    }

    def __init__(self, base_path: str, batch_size: int = 32):
        self.base_path = Path(base_path)
        self.batch_size = batch_size
        self.current_loader: Optional[UniversalDataLoader] = None
        
    def _find_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """
        Locates the dataset directory or file within the base path.
        """
        # 1. Direct check
        direct_path = self.base_path / dataset_name
        if direct_path.exists():
            return direct_path
            
        # 2. Recursive search
        matches = list(self.base_path.rglob(dataset_name))
        if matches:
            return matches[0]
            
        return None

    def load_capability(self, capability: str, split: str = "test") -> Iterator[Tuple[List[Dict], List[Dict]]]:
        """
        Loads datasets associated with a capability.
        Yields batches of (normalized_samples, metadata).
        """
        target_datasets = self.CAPABILITY_MAP.get(capability, [])
        if not target_datasets:
            logger.warning(f"No datasets defined for capability: {capability}")
            return

        for dataset_name in target_datasets:
            path = self._find_dataset_path(dataset_name)
            if not path:
                logger.warning(f"Dataset {dataset_name} not found in {self.base_path}")
                continue
            
            logger.info(f"Loading dataset: {dataset_name} from {path}")
            yield from self.load_dataset(str(path), dataset_name, split)

    def load_dataset(self, path: str, dataset_name: str, split: str = "test") -> Iterator[Tuple[List[Dict], List[Dict]]]:
        """
        Streams batches from a specific dataset.
        """
        loader = UniversalDataLoader(path, dataset_name=dataset_name, split=split, fast_mode=True)
        
        # We need to implement a streaming iterator in UniversalDataLoader or handle it here.
        # UniversalDataLoader.load() loads everything or a sample. 
        # Ideally we'd modify UniversalDataLoader to support streaming, but for now 
        # we can use its index map logic if it's sharded, or load chunks.
        
        # Using a simpler approach: Load in chunks if supported, or load all and batch (if small).
        # For huge datasets (Parquet/JSONL), UniversalDataLoader has logic for specific indices.
        
        # Let's verify size first
        fmt = loader.detect_format()
        total_samples = 1000 # Default estimate
        
        if fmt in ["parquet_sharded", "jsonl_sharded"]:
            # Iterate through indices
            current_batch = []
            
            # This is a bit inefficient without a true iterator in UniversalDataLoader
            # We'll assume a "reasonable" max or implement a proper generator there.
            # For now, let's implement a batch fetcher here using load_sample.
            
            # Initialize index map
            if not loader.index_map:
                try:
                    loader.load_sample(0) # Trigger map build
                except: pass
            
            if loader.index_map:
                total_samples = loader.index_map.total_count
                
            for i in range(total_samples):
                sample = loader.load_sample(i)
                if not sample: break
                
                # Normalize
                normalized = SchemaNormalizer.normalize(sample, dataset_name)
                current_batch.append(normalized)
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch, [{"source": dataset_name, "index": i-j} for j in range(len(current_batch))]
                    current_batch = []
                    
            if current_batch:
                yield current_batch, [{"source": dataset_name}]
                
        else:
            # Load all (assuming it fits in memory for non-sharded)
            result = loader.load()
            if not result.dataset:
                return

            dataset = result.dataset
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i : i + self.batch_size]
                normalized_batch = [SchemaNormalizer.normalize(s, dataset_name) for s in batch]
                yield normalized_batch, [{"source": dataset_name} for _ in batch]

    def get_prompt_target_batch(self, batch: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Helper to extract prompts and targets from normalized batch.
        """
        prompts = []
        targets = []
        for item in batch:
            if not item.get("messages"):
                prompts.append("")
                targets.append("")
                continue
                
            # Nexus Schema: user -> assistant
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
            
            prompts.append(user_content if isinstance(user_content := user_msg, str) else str(user_content))
            targets.append(assistant_content if isinstance(assistant_content := asst_msg, str) else str(assistant_content))
            
        return prompts, targets

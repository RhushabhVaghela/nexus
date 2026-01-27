import csv
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelInfo:
    name: str
    parameters: str
    category: str
    best_feature: str
    path: str  # Inferred local path

class NexusConfig:
    def __init__(self, csv_path: str, models_root: str, benchmarks_root: str):
        self.models_root = models_root
        self.benchmarks_root = benchmarks_root
        self.models: Dict[str, ModelInfo] = self._load_model_registry(csv_path)
        
        # Mapping categories to benchmark paths
        self.benchmark_map = {
            "Agent (LLM-based)": os.path.join(benchmarks_root, "math/openai_gsm8k/main/test-00000-of-00001.parquet"), # Placeholder, ideal is GAIA
            "Language model (likely scope/PT variant)": os.path.join(benchmarks_root, "general/cais_mmlu/main/test-00000-of-00001.parquet"),
            "Language model (MoE)": os.path.join(benchmarks_root, "code/human_eval/main/test-00000-of-00001.parquet"),
             # Add others as needed, falling back to general if unknown
        }

    def _load_model_registry(self, csv_path: str) -> Dict[str, ModelInfo]:
        registry = {}
        if not os.path.exists(csv_path):
            print(f"Warning: CSV path {csv_path} does not exist.")
            return registry

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Model Name", "").strip()
                if not name:
                    continue
                
                # Infer local path: /mnt/e/data/models/<Name>
                # Handle slashes in names like "zai-org/GLM-4.7-Flash" -> "GLM-4.7-Flash" or keep full structure?
                # Usually local mirror might just be the name or the last part.
                # For now, assuming direct mapping or user manual alignment.
                # Let's assume the folder name matches the Model Name exactly.
                local_path = os.path.join(self.models_root, name.replace("/", "_")) 
                
                registry[name] = ModelInfo(
                    name=name,
                    parameters=row.get("Parameters", ""),
                    category=row.get("Category", ""),
                    best_feature=row.get("Best Feature", ""),
                    path=local_path
                )
        return registry

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        return self.models.get(model_name)

    def get_benchmark_path(self, model_name: str) -> str:
        info = self.models.get(model_name)
        if not info:
            return ""
        
        # Default fallback
        default_bench = os.path.join(self.benchmarks_root, "math/openai_gsm8k/main/test-00000-of-00001.parquet")
        
        # Try to find specific match
        for key, path in self.benchmark_map.items():
            if key in info.category:
                return path
        
        return default_bench

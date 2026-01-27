import json
import csv
import os
import argparse
from typing import Dict, Any, List, Optional

class TeacherRegistry:
    def __init__(self, csv_path: str, structure_files: List[str], mapping_override: Optional[str] = None):
        self.csv_path = csv_path
        self.structure_files = structure_files
        self.mapping_override = mapping_override
        self.registry: List[Dict[str, Any]] = []
        
    def _parse_csv(self) -> List[Dict[str, str]]:
        teachers = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                teachers.append({
                    "name": row['Model Name'].strip(),
                    "params": row['Parameters'].strip(),
                    "category": row['Category'].strip(),
                    "feature": row['Best Feature'].strip()
                })
        return teachers

    def _find_model_path(self, model_name: str) -> Optional[str]:
        # Heuristic 1: Check resource_mapping.json first if available
        if self.mapping_override and os.path.exists(self.mapping_override):
            with open(self.mapping_override, 'r') as f:
                mapping = json.load(f)
                if model_name in mapping.get("models", {}):
                    return mapping["models"][model_name]
        
        # Heuristic 2: Scan structure files
        # Normalize name for search (e.g., "AgentCPM-Explore" -> "AgentCPM-Explore")
        search_term = model_name.replace("/", "_")
        
        for s_file in self.structure_files:
            if not os.path.exists(s_file):
                continue
            with open(s_file, 'r') as f:
                for line in f:
                    path = line.strip()
                    if search_term in path and (path.endswith("config.json") or path.endswith(".safetensors")):
                        # Return the directory containing this file
                        return os.path.dirname(path)
        return None

    def _determine_capabilities(self, category: str) -> Dict[str, Any]:
        """
        Maps CSV category to capabilities contract based on retention_contracts.md
        """
        cat_lower = category.lower()
        capabilities = {}

        # Text & Reasoning
        if "language model" in cat_lower or "agent" in cat_lower:
            capabilities["reasoning"] = {
                "benchmark": "gsm8k",
                "metric": "accuracy",
                "retain_fraction": 0.97
            }
            capabilities["knowledge"] = {
                "benchmark": "mmlu",
                "metric": "accuracy",
                "retain_fraction": 0.97
            }
            if "agent" in cat_lower:
                capabilities["agent"] = {
                    "benchmark": "gaia",
                    "metric": "success_rate",
                    "retain_fraction": 0.97
                }
            if "coding" in cat_lower or "code" in cat_lower: # Infer from category string if explicit
                 capabilities["code"] = {
                    "benchmark": "humaneval",
                    "metric": "pass@1",
                    "retain_fraction": 0.95
                }

        # Vision
        elif "vision" in cat_lower or "image" in cat_lower:
            if "generation" in cat_lower:
                 capabilities["image_generation"] = {
                    "benchmark": "fid",
                    "metric": "score",
                    "retain_fraction": 0.95
                }
            elif "encoder" in cat_lower:
                 capabilities["vision_encoder"] = {
                    "benchmark": "imagenet_linear",
                    "metric": "accuracy",
                    "retain_fraction": 0.97
                }
            else: # Standard VQA/VL
                capabilities["vision_qa"] = {
                    "benchmark": "vqav2",
                    "metric": "accuracy",
                    "retain_fraction": 0.97
                }

        # Audio
        elif "audio" in cat_lower or "tts" in cat_lower or "asr" in cat_lower:
            if "asr" in cat_lower:
                capabilities["asr"] = {
                    "benchmark": "librispeech",
                    "metric": "wer",
                    "retain_fraction": 1.03 # Allow 3% degradation
                }
            elif "tts" in cat_lower:
                capabilities["tts"] = {
                    "benchmark": "cosine_sim",
                    "metric": "similarity",
                    "retain_fraction": 0.85
                }
            elif "tokenizer" in cat_lower:
                 capabilities["audio_tokenizer"] = {
                    "benchmark": "reconstruction",
                    "metric": "mse",
                    "retain_fraction": 0.99
                }
            else:
                 capabilities["voice_chat"] = {
                    "benchmark": "moshi_eval",
                    "metric": "coherence",
                    "retain_fraction": 0.95
                }
                
        # Video
        elif "video" in cat_lower:
             capabilities["video"] = {
                "benchmark": "ucf101",
                "metric": "fvd",
                "retain_fraction": 0.95
            }
            
        return capabilities

    def build_registry(self) -> List[Dict[str, Any]]:
        raw_teachers = self._parse_csv()
        final_registry = []
        
        for t in raw_teachers:
            path = self._find_model_path(t['name'])
            
            capabilities = self._determine_capabilities(t['category'])
            
            entry = {
                "teacher_id": t['name'],
                "category": t['category'],
                "best_feature": t['feature'],
                "path": path if path else "MISSING",
                "status": "ready" if path else "missing",
                "capabilities": capabilities
            }
            final_registry.append(entry)
            
        self.registry = final_registry
        return final_registry

    def save(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        print(f"Registry saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--structure-files", nargs="+", required=True)
    parser.add_argument("--mapping", help="Path to resource_mapping.json")
    parser.add_argument("--out", required=True)
    
    args = parser.parse_args()
    
    registry = TeacherRegistry(args.csv, args.structure_files, args.mapping)
    registry.build_registry()
    registry.save(args.out)

import json
import os
from typing import List, Tuple

class NexusDataLoader:
    """
    Handles loading of local benchmarks (GSM8K, GAIA) for NIWT profiling.
    Implements the 'Switchable' logic defined in 'chat-11'.
    """
    def __init__(self, base_path: str = "/mnt/d/Research Experiments/nexus/data"):
        self.base_path = base_path
        
    def load_gsm8k(self, num_samples: int = 50) -> List[Tuple[str, str]]:
        """
        Load Reasoning Benchmark (GSM8K).
        Returns: [(Question, Answer)]
        """
        # Adjust path to match user's likely structure or the one in chat-10
        # The chat mentioned /mnt/e, but user is on /mnt/d. We use relative or config.
        # Fallback to a clear relative path or absolute if known.
        path = os.path.join(self.base_path, "benchmarks/math/openai_gsm8k/main/test.jsonl")
        
        if not os.path.exists(path):
            print(f"[Warning] GSM8K not found at {path}. Returning Mock Data.")
            return [("Mock Question", "Mock Answer") for _ in range(num_samples)]

        samples = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                q = data['question']
                # Extract numeric answer after ####
                a = data['answer'].split('####')[-1].strip()
                samples.append((q, a))
                if len(samples) >= num_samples:
                    break
        return samples

    def load_gaia(self, num_samples: int = 50) -> List[Tuple[str, str]]:
        """
        Load Agentic Benchmark (GAIA).
        """
        path = os.path.join(self.base_path, "benchmarks/general/gaia/validation.jsonl")
        if not os.path.exists(path):
            print(f"[Warning] GAIA not found at {path}. Returning Mock Data.")
            return [("Mock Agent Task", "Mock Result") for _ in range(num_samples)]
            
        samples = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                q = data['task']
                a = data.get('final_answer', '')
                samples.append((q, a))
                if len(samples) >= num_samples:
                    break
        return samples

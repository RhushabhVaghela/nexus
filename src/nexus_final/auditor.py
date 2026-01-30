import torch
import zlib
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizer

class MemorizationAuditor:
    """
    Implements memorization metrics inspired by arXiv:2601.15394.
    
    1. Zlib Entropy: Predicts memorizability based on compressibility.
    2. Discoverable Memorization: Exact match check for greedy generation.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, prefix_len: int = 50, suffix_len: int = 50):
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

    @staticmethod
    def calculate_zlib_entropy(text: str) -> float:
        """Calculates the zlib entropy of a text string."""
        if not text:
            return 0.0
        compressed = zlib.compress(text.encode('utf-8'))
        # Normalized entropy: compressed size / original size
        return len(compressed) / len(text)

    def audit_sample(self, model: torch.nn.Module, text: str, device: str = "cuda") -> Dict[str, Any]:
        """
        Performs a 'Discoverable Memorization' audit on a single text sample.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) < self.prefix_len + self.suffix_len:
            return {"status": "skipped", "reason": "text too short"}

        prefix_tokens = tokens[:self.prefix_len]
        ground_truth_suffix = tokens[self.prefix_len : self.prefix_len + self.suffix_len]
        
        input_ids = torch.tensor([prefix_tokens]).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=self.suffix_len,
                do_sample=False, # Forced greedy as per paper
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated parts
        generated_suffix = generated_ids[0, self.prefix_len:].tolist()
        
        # Check for exact match
        memorized = False
        if len(generated_suffix) >= self.suffix_len:
            memorized = (generated_suffix[:self.suffix_len] == ground_truth_suffix)
            
        return {
            "status": "success",
            "entropy": self.calculate_zlib_entropy(text),
            "memorized": memorized,
            "match_ratio": self._calculate_match_ratio(generated_suffix, ground_truth_suffix)
        }

    def _calculate_match_ratio(self, generated: List[int], target: List[int]) -> float:
        if not target:
            return 0.0
        matches = sum(1 for g, t in zip(generated, target) if g == t)
        return matches / len(target)

    def batch_audit(self, model: torch.nn.Module, texts: List[str], device: str = "cuda") -> Dict[str, Any]:
        results = []
        for text in texts:
            results.append(self.audit_sample(model, text, device))
        
        valid_results = [r for r in results if r["status"] == "success"]
        if not valid_results:
            return {"avg_memorization_rate": 0.0, "count": 0}
            
        mem_rate = sum(1 for r in valid_results if r["memorized"]) / len(valid_results)
        avg_entropy = sum(r["entropy"] for r in valid_results) / len(valid_results)
        
        return {
            "avg_memorization_rate": mem_rate,
            "avg_entropy": avg_entropy,
            "sample_count": len(valid_results)
        }

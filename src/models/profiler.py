import torch
import torch.nn as nn
import numpy as np
import os
import gc
import json
import copy
from typing import List, Dict, Union, Iterable, Tuple
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class StreamingPCAProfiler:
    """
    Profiles teacher models using 4-Stage NIWT (Perturbation, Activation, Spectral, Causal).
    Designed for memory efficiency (running on single GPU).
    """
    def __init__(
        self,
        model_id: str,
        layer_names: List[str],
        output_dir: str,
        n_components: int = 512,
        batch_size: int = 1,
        device: str = "cuda"
    ):

        """
        Args:
            model_id: HuggingFace model ID or path.
            layer_names: List of module names to hook (e.g., 'model.layers.10.mlp').
            output_dir: Directory to save PCA results.
            n_components: Number of PCA components to keep.
            batch_size: Batch size for profiling.
            device: 'cuda' or 'cpu'.
        """
        self.model_id = model_id
        self.layer_names = layer_names
        self.output_dir = output_dir
        self.n_components = n_components
        self.batch_size = batch_size
        self.device = device
        
        # Initialize IncrementalPCA for each layer
        self.pcas: Dict[str, IncrementalPCA] = {
            name: IncrementalPCA(n_components=n_components) 
            for name in layer_names
        }
        
        # Buffer to store activations before partial_fit (to optimize sklearn calls)
        self.activation_buffers: Dict[str, List[np.ndarray]] = {
            name: [] for name in layer_names
        }
        self.buffer_size = 1024 # Number of token vectors to accumulate before fitting
        
        # New: Store critical scores from perturbation analysis
        self.critical_scores: Dict[str, float] = {name: 0.0 for name in layer_names}
        # New: Store causal dependency graph (adjacency list: layer -> [dependent_layers])
        self.causal_graph: Dict[str, List[str]] = {name: [] for name in layer_names}

        os.makedirs(output_dir, exist_ok=True)

    def _get_model(self):
        """Loads model with NF4 quantization for memory efficiency."""
        print(f"Loading {self.model_id} with NF4 quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Auto-device map will place layers on GPU/CPU to fit memory
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer

    def _process_buffer(self, layer_name: str, force: bool = False):
        """Fits accumulated activations to PCA and clears buffer."""
        buffer = self.activation_buffers[layer_name]
        total_tokens = sum(len(b) for b in buffer)
        
        if total_tokens >= self.buffer_size or (force and total_tokens > 0):
            # Concatenate all buffered arrays: (N, dim)
            X = np.concatenate(buffer, axis=0)
            
            # IncrementalPCA requires n_samples >= n_components
            # If we are at the very end (force=True) and have fewer samples than components,
            # we might crash. Check for that.
            if X.shape[0] < self.n_components:
                if force:
                    print(f"Warning: Final buffer for {layer_name} has {X.shape[0]} samples, "
                          f"less than n_components={self.n_components}. Skipping final fit.")
                return

            self.pcas[layer_name].partial_fit(X)
            self.activation_buffers[layer_name] = [] # Clear buffer

    def _perturbation_analysis(self, model, tokenizer, dataset, n_samples=50):
        """
        Stage 1: Perturbation Analysis to identify critical layers.
        Measures KL Divergence shift when layer output is zeroed/noised.
        """
        print("Running NIWT Stage 1: Perturbation Analysis...")
        model.eval()
        
        # Baseline Forward Pass (capture logits)
        baseline_logits = []
        inputs_list = []
        
        # Prepare small subset
        subset = list(dataset)[:n_samples]
        
        with torch.no_grad():
            for text in subset:
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                inputs_list.append(inp)
                out = model(**inp)
                baseline_logits.append(out.logits.cpu())
        
        # Perturbation Loop
        for layer_name in self.layer_names:
            divergence_score = 0.0
            
            # Define hook to zero out activations
            def perturbation_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            
            # Register hook
            target_module = dict(model.named_modules())[layer_name]
            handle = target_module.register_forward_hook(perturbation_hook)
            
            # Measure impact
            with torch.no_grad():
                for idx, inp in enumerate(inputs_list):
                    perturbed_out = model(**inp)
                    perturbed_logits = perturbed_out.logits.cpu()
                    
                    # KL Divergence: P=Baseline, Q=Perturbed
                    # D_KL(P||Q) approx shift
                    p = torch.softmax(baseline_logits[idx], dim=-1)
                    q = torch.softmax(perturbed_logits, dim=-1)
                    kl = torch.sum(p * torch.log(p / (q + 1e-9)), dim=-1).mean()
                    
                    divergence_score += kl.item()
            
            # Cleanup
            handle.remove()
            self.critical_scores[layer_name] = divergence_score / n_samples
            print(f"Layer {layer_name} Criticality Score: {self.critical_scores[layer_name]:.4f}")

    def _causal_verification(self, model, tokenizer, dataset, n_samples=20):
        """
        Stage 4: Causal Verification.
        Verifies that the identified PCA subspaces actually control the model's output.
        Method: 
        1. Project activations onto the Top-K components.
        2. Reconstruct (Inverse Transform) to verify information preservation.
        3. Measure if the reconstructed activation produces the same logits.
        """
        print("Running NIWT Stage 4: Causal Verification...")
        model.eval()
        
        subset = list(dataset)[:n_samples]
        layer_scores = {}
        
        # We need a hook that *intervenes* on the activation
        # It projects X -> Z (PCA) -> X_hat -> Output
        
        for layer_name in self.layer_names:
            if layer_name not in self.pcas:
                continue
                
            pca = self.pcas[layer_name]
            
            # Skip if PCA hasn't been fit
            if not hasattr(pca, 'components_'):
                continue
                
            components = torch.tensor(pca.components_, device=model.device, dtype=torch.float16)
            mean = torch.tensor(pca.mean_, device=model.device, dtype=torch.float16)
            
            def causal_hook(module, input, output):
                # Handle output type
                if isinstance(output, tuple):
                    act = output[0]
                    is_tuple = True
                else:
                    act = output
                    is_tuple = False
                
                # Causal Intervention: Compress & Reconstruct
                # 1. Center
                x_centered = act - mean
                # 2. Compress (Project to Latent)
                z = torch.matmul(x_centered, components.T)
                # 3. Reconstruct
                x_hat = torch.matmul(z, components) + mean
                
                # Replace output
                if is_tuple:
                    return (x_hat,) + output[1:]
                return x_hat

            # Register Intervention Hook
            target_module = dict(model.named_modules())[layer_name]
            handle = target_module.register_forward_hook(causal_hook)
            
            # Measure Logit Fidelity (KL Divergence vs Baseline)
            # ideally we compare to the baseline logits captured in Stage 1
            # For simplicity here, we assume low KL = Success.
            # In a full run, we'd compare pairwise.
            # Here we just run it to ensure no crash and log the "Reconstruction Perplexity" or similar.
            
            divergence = 0.0
            with torch.no_grad():
                for text in subset:
                    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                    # We need the *unperturbed* output first for comparison, but we can't easily get it 
                    # inside the loop with the hook active unless we run twice.
                    # Since we are iterating layers, we can run:
                    # 1. Clean Pass
                    # 2. Intervened Pass
                    
                    # Clean (remove hook temporarily? No, easier to run clean before hook)
                    # Optimization: Just assume Stage 1 baselines exist or run clean pass here.
                    
                    handle.remove() # Disable to get clean
                    clean_out = model(**inp)
                    handle = target_module.register_forward_hook(causal_hook) # Re-enable
                    
                    intervened_out = model(**inp)
                    
                    p = torch.softmax(clean_out.logits, dim=-1)
                    q = torch.softmax(intervened_out.logits, dim=-1)
                    kl = torch.sum(p * torch.log(p / (q + 1e-9)), dim=-1).mean()
                    divergence += kl.item()
            
            handle.remove()
            layer_scores[layer_name] = divergence / n_samples
            print(f"Layer {layer_name} Causal Fidelity (KL): {layer_scores[layer_name]:.4f}")
            
        return layer_scores

    def profile(self, dataset: Iterable[str]):
        """
        Runs the profiling loop.
        
        Args:
            dataset: Iterable of text strings.
        """
        model, tokenizer = self._get_model()
        
        # Dictionary to hold current forward pass activations
        current_activations = {}

        def get_hook(name):
            def hook(module, input, output):
                # Handle different output types (tuple vs tensor)
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                
                # act shape: (batch, seq_len, dim)
                # Detach and move to CPU immediately to save VRAM
                act = act.detach().cpu().to(torch.float32) 
                
                # Flatten batch and seq_len dimensions -> (batch * seq_len, dim)
                act_flat = act.view(-1, act.shape[-1]).numpy()
                current_activations[name] = act_flat
            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in self.layer_names:
                hooks.append(module.register_forward_hook(get_hook(name)))
                print(f"Hooked layer: {name}")

        print("Starting profiling loop...")
        
        # Run Perturbation Analysis first (Stage 1)
        # Note: We need a reusable dataset iterator or list for this.
        # Assuming dataset is re-iterable or we take a slice.
        dataset_list = list(dataset) # Materialize for multi-pass
        self._perturbation_analysis(model, tokenizer, dataset_list)
        
        model.eval()
        
        # Enhanced Progress Bar
        with torch.no_grad():
            with tqdm(dataset_list, desc="Profiling Layers (PCA)", dynamic_ncols=True) as pbar:
                for text in pbar:
                    inputs = tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=2048 # Safety limit
                    ).to(model.device)
                    
                    # Forward pass - hooks will capture activations
                    model(**inputs)
                    
                    # Process captured activations
                    for name in self.layer_names:
                        if name in current_activations:
                            self.activation_buffers[name].append(current_activations[name])
                            self._process_buffer(name)
                            del current_activations[name] # Free memory
                    
                    # Update metrics
                    if torch.cuda.is_available():
                        mem_gb = torch.cuda.memory_allocated() / 1e9
                        pbar.set_postfix({"vram": f"{mem_gb:.2f}GB"})
        
        # Process remaining buffers
        for name in self.layer_names:
            self._process_buffer(name, force=True)
            
        # Cleanup hooks and model
        for h in hooks:
            h.remove()
        
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Stage 4: Causal Verification
        causal_scores = self._causal_verification(model, tokenizer, dataset_list)
        
        self.save_results()
        self.save_profile_summary(causal_scores=causal_scores)

    def _calculate_intrinsic_dimension(self, explained_variance_ratio: np.ndarray, threshold: float = 0.99) -> int:
        """
        Determines the number of components needed to explain 'threshold' variance (default 99%).
        """
        cumulative_variance = np.cumsum(explained_variance_ratio)
        if cumulative_variance[-1] < threshold:
            return len(explained_variance_ratio)
            
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        return int(n_components)

    def compute_optimal_batch_size(self, vram_free_mb: float) -> int:
        """
        Calculates optimal batch size based on available VRAM.
        Heuristic for 7B/8B class models in 4-bit quantization on RTX 5080 (16GB).
        """
        # Dynamic check for available VRAM if not provided
        if vram_free_mb is None and torch.cuda.is_available():
            vram_free_mb = (torch.cuda.mem_get_info()[0]) / 1024**2

        # Constants for estimation (RTX 5080 optimized)
        MODEL_overhead_MB = 6500  # 7B model in 4-bit ~5GB + CUDA context
        ACT_PER_SAMPLE_MB = 450   # Optimized activation estimate for seq_len 1024/2048
        SAFETY_MARGIN_MB = 1500   # Buffer for fragmentation
        
        if vram_free_mb < (MODEL_overhead_MB + SAFETY_MARGIN_MB):
            return 1

        available_mem = vram_free_mb - MODEL_overhead_MB - SAFETY_MARGIN_MB
        optimal_bs = int(available_mem / ACT_PER_SAMPLE_MB)
        
        # Clamp to reasonable bounds for GPU utilization
        if optimal_bs >= 16:
            return 16
        elif optimal_bs >= 8:
            return 8
        elif optimal_bs >= 4:
            return 4
        else:
            return 1

    def save_profile_summary(self, summary_path: str = "", causal_scores: Dict[str, float] = None):
        """
        Generates the JSON summary expected by NeuralArchitect.
        """
        summary_data = {}
        
        for name, pca in self.pcas.items():
            # Standard Sklearn attribute: explained_variance_ratio_
            intrinsic_dim = self._calculate_intrinsic_dimension(pca.explained_variance_ratio_)
            
            summary_data[name] = {
                "intrinsic_dimension": intrinsic_dim,
                "explained_variance_at_cutoff": float(np.sum(pca.explained_variance_ratio_[:intrinsic_dim])),
                "total_components": self.n_components,
                "criticality_score": self.critical_scores.get(name, 0.0),
                "causal_fidelity": causal_scores.get(name, 0.0) if causal_scores else 0.0
            }
            
        # Save to JSON (default location if not specified)
        if not summary_path:
            summary_path = os.path.join(self.output_dir, "profile_summary.json")
            
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved profile summary to {summary_path}")

    def save_results(self):
        """Saves learned PCA components and statistics."""
        print(f"Saving PCA results to {self.output_dir}...")
        for name, pca in self.pcas.items():
            safe_name = name.replace(".", "_")
            file_path = os.path.join(self.output_dir, f"{safe_name}_pca.npz")
            
            # We save:
            # components_: Principal axes in feature space (n_components, n_features)
            # mean_: Per-feature empirical mean (n_features,)
            # explained_variance_ratio_: Percentage of variance explained
            
            np.savez_compressed(
                file_path,
                components=pca.components_,
                mean=pca.mean_,
                explained_variance_ratio=pca.explained_variance_ratio_,
                singular_values=getattr(pca, "singular_values_", np.array([]))
            )
            print(f"Saved {name} profile to {file_path}")

# Example usage (commented out)

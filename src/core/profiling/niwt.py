import torch
import torch.nn as nn
import json
import os
import time
import gc
from typing import List, Dict, Tuple, Optional, Any
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter([])
from src.core.utils.universal_inspector import UniversalInspector


class ThermalProtection:
    """
    Hardware Safety Monitor.
    Pauses execution if GPU temperature exceeds critical threshold (83C).
    """

    def __init__(self, threshold=83.0, cooldown_sec=30):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec

    def check(self):
        import subprocess
        import shutil

        if shutil.which("nvidia-smi"):
            try:
                # Query GPU 0 Temperature
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    encoding="utf-8",
                )
                temp = int(result.strip())
                if temp >= self.threshold:
                    print(
                        f"\n[CRITICAL] GPU Temp {temp}C exceeds limit {self.threshold}C! Cooling down for {self.cooldown_sec}s..."
                    )
                    time.sleep(self.cooldown_sec)
            except Exception as e:
                # Silent fail on environments where nvidia-smi might be weird
                pass


class EvaluationDataset(Dataset):
    """Simple Dataset wrapper for batched evaluation."""

    def __init__(self, samples: List[Tuple[str, str]]):
        self.samples = samples  # List of (prompt, target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NIWTCore:
    """
    The Neural Information-Weighted Tower (NIWT) Engine.
    Implements the 4-Stage extraction pipeline with Optimized Batch Processing.
    """

    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.critical_layers = []
        self.neuron_mask = {}
        self.thermal = ThermalProtection()

        # Optimization Config
        self.batch_size = config.get("batch_size", 4)

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # =========================================================
    # STAGE 1: PERTURBATION (Layer-Level)
    # =========================================================
    def run_stage_1_perturbation(self, test_cases: List[Tuple[str, str]]) -> List[Dict]:
        """
        Surgically disable layers to see which tasks break.
        Returns a list of Critical Layers.
        """
        print(
            f"\n[NIWT Stage 1] Starting Layer Perturbation Analysis (BS={self.batch_size})..."
        )

        # 1. Establish Baseline
        print("Calculating Optimized Baseline...")
        baseline_score = self._evaluate_capability_batched(test_cases)
        print(f"[Baseline] Score: {baseline_score:.2%}")

        results = []

        # 2. Iterate Layers
        layers = self._get_model_layers()

        # Create a progress bar
        pbar = tqdm(range(len(layers)), desc="Layer Profiling")

        for i in pbar:
            # Thermal check every few layers
            if i % 5 == 0:
                self.thermal.check()

            # Hook mechanism to bypass layer
            # We use a simple identity bypass for the "Perturbation"
            original_forward = layers[i].forward
            layers[i].forward = lambda *args, **kwargs: args[
                0
            ]  # Identity: Pass hidden states through

            # Evaluate
            score = self._evaluate_capability_batched(test_cases)
            drop = (baseline_score - score) / (baseline_score + 1e-9)

            # Restore
            layers[i].forward = original_forward

            # Classify
            is_critical = drop > 0.15  # 15% threshold from docs

            status = "CRITICAL" if is_critical else "OK"
            pbar.set_postfix({"Layer": i, "Drop": f"{drop:+.2%}", "Status": status})

            if is_critical:
                self.critical_layers.append({"layer": i, "drop": drop, "score": score})

            results.append(
                {"layer": i, "drop": drop, "critical": is_critical, "score": score}
            )

            # Cleanup VRAM
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # 3. Fallback: If 0 critical layers found, take the top 3 most impactful anyway
        if not self.critical_layers and results:
            print(
                "[Stage 1 Fallback] No layers crossed threshold. Selecting top 3 most impactful layers..."
            )
            # Sort by drop descending
            sorted_results = sorted(results, key=lambda x: x["drop"], reverse=True)
            for res in sorted_results[:3]:
                self.critical_layers.append(
                    {"layer": res["layer"], "drop": res["drop"], "score": res["score"]}
                )
                print(
                    f"  [Fallback] Selected Layer {res['layer']:02d} | Drop: {res['drop']:.2%}"
                )

        print(f"[Stage 1] Complete. Found {len(self.critical_layers)} critical layers.")
        return self.critical_layers

    # =========================================================
    # STAGE 2: ACTIVATION ANALYSIS (Neuron-Level)
    # =========================================================
    def run_stage_2_activation_analysis(self, calibration_data: List[str]):
        """
        For Critical Layers, find specifically which neurons fire.
        Generates the 'Feature Bitmask'.
        """
        print("\n[NIWT Stage 2] Starting Activation Analysis...")
        if not self.critical_layers:
            print("[Skip] No critical layers found in Stage 1.")
            return

        # Hooks to capture activations
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                # Simple Max-over-time pooling for firing detection
                # output shape: (batch, seq, hidden)
                # We want to see which hidden dim is active average over batch/seq
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output.detach().abs().mean(dim=(0, 1))

            return hook

        # Register hooks on Critical Layers
        handles = []
        layers = self._get_model_layers()
        for cl in self.critical_layers:
            idx = cl["layer"]
            handle = layers[idx].register_forward_hook(get_activation(f"layer_{idx}"))
            handles.append(handle)

        # Pass Data (Batched)
        print(f"Analyzing activations with {len(calibration_data)} samples...")

        # Simple batching for calibration
        loader = DataLoader(calibration_data, batch_size=self.batch_size, shuffle=False)

        for batch_prompts in tqdm(loader, desc="Activation Scan"):
            inputs = self.tokenizer(
                batch_prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                self.model(**inputs)

            # Clear VRAM after batch
            del inputs
            torch.cuda.empty_cache()

        # Analyze & Threshold
        for name, act_tensor in activations.items():
            # Threshold: Top 30% active neurons
            threshold = torch.quantile(act_tensor, 0.70)
            mask = act_tensor > threshold
            layer_idx = int(name.split("_")[1])
            self.neuron_mask[layer_idx] = mask.nonzero().squeeze().tolist()
            print(
                f"  [Layer {layer_idx}] Masked {len(self.neuron_mask[layer_idx])} / {len(act_tensor)} neurons"
            )

        # Cleanup
        for h in handles:
            h.remove()

    # =========================================================
    # STAGE 3: KNOWLEDGE DISTILLATION (Guided Transfer)
    # =========================================================
    def run_stage_3_distillation(
        self,
        student_model: torch.nn.Module,
        calibration_data: List[str],
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        temperature: float = 1.5,
        alpha: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Use the neuron_mask from Stage 2 to guide knowledge distillation
        from the teacher (self.model) to a student model.

        Only critical layers (from Stage 1) with their active neurons
        (from Stage 2) are used to compute the distillation loss, focusing
        the transfer on the most important knowledge.

        Args:
            student_model: The smaller/quantized student model to train
            calibration_data: List of text prompts for distillation
            num_epochs: Number of distillation epochs
            learning_rate: Optimizer learning rate
            temperature: Temperature for KL divergence softening
            alpha: Weight for distillation vs hard target loss (0.0-1.0)

        Returns:
            Dictionary with distillation metrics (loss history, final loss, etc.)
        """
        print(f"\n[NIWT Stage 3] Starting Guided Knowledge Distillation...")

        if not self.critical_layers:
            print("[Skip] No critical layers from Stage 1. Run Stage 1 first.")
            return {"status": "skipped", "reason": "no_critical_layers"}

        if not self.neuron_mask:
            print(
                "[Warning] No neuron mask from Stage 2. Distilling without neuron guidance."
            )

        # Import QAD loss for distillation
        from src.models.sli.qad_loss import QADDistillationLoss, QADLossConfig

        # Setup loss function
        qad_config = QADLossConfig(
            temperature=temperature,
            alpha=alpha,
            use_hidden_matching=True,
            use_attention_matching=False,  # Attention matching optional
        )
        distill_loss_fn = QADDistillationLoss(qad_config)

        # Setup optimizer for student
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

        # Ensure models in correct mode
        self.model.eval()
        student_model.train()

        # Prepare data loader
        loader = DataLoader(calibration_data, batch_size=self.batch_size, shuffle=True)

        loss_history = []
        critical_layer_indices = {cl["layer"] for cl in self.critical_layers}

        # Hook storage for capturing hidden states at critical layers
        teacher_hiddens = {}
        student_hiddens = {}

        def make_teacher_hook(layer_idx):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                # Apply neuron mask if available
                if layer_idx in self.neuron_mask:
                    mask_indices = self.neuron_mask[layer_idx]
                    if isinstance(mask_indices, list) and len(mask_indices) > 0:
                        # Store only the masked neuron activations
                        teacher_hiddens[layer_idx] = out[:, :, mask_indices].detach()
                        return
                teacher_hiddens[layer_idx] = out.detach()

            return hook

        def make_student_hook(layer_idx):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                if layer_idx in self.neuron_mask:
                    mask_indices = self.neuron_mask[layer_idx]
                    if isinstance(mask_indices, list) and len(mask_indices) > 0:
                        student_hiddens[layer_idx] = out[:, :, mask_indices]
                        return
                student_hiddens[layer_idx] = out

            return hook

        # Register hooks on critical layers
        teacher_layers = self._get_model_layers()
        student_layers = UniversalInspector.find_backbone_layers(student_model)

        teacher_handles = []
        student_handles = []
        for cl in self.critical_layers:
            idx = cl["layer"]
            if idx < len(teacher_layers):
                teacher_handles.append(
                    teacher_layers[idx].register_forward_hook(make_teacher_hook(idx))
                )
            if idx < len(student_layers):
                student_handles.append(
                    student_layers[idx].register_forward_hook(make_student_hook(idx))
                )

        # Distillation loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_prompts in tqdm(
                loader, desc=f"Distill Epoch {epoch + 1}/{num_epochs}"
            ):
                # Thermal check
                if num_batches % 10 == 0:
                    self.thermal.check()

                inputs = self.tokenizer(
                    batch_prompts, padding=True, truncation=True, return_tensors="pt"
                ).to(self.model.device)

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = self.model(**inputs)
                    teacher_logits = (
                        teacher_outputs.logits
                        if hasattr(teacher_outputs, "logits")
                        else teacher_outputs[0]
                    )

                # Student forward
                student_outputs = student_model(**inputs)
                student_logits = (
                    student_outputs.logits
                    if hasattr(student_outputs, "logits")
                    else student_outputs[0]
                )

                # Compute distillation loss on logits
                loss = distill_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                )

                # Add hidden state matching loss for critical layers
                for idx in critical_layer_indices:
                    if idx in teacher_hiddens and idx in student_hiddens:
                        t_h = teacher_hiddens[idx]
                        s_h = student_hiddens[idx]
                        # Ensure same shape before MSE
                        if t_h.shape == s_h.shape:
                            hidden_loss = torch.nn.functional.mse_loss(s_h, t_h)
                            loss = loss + (qad_config.beta * hidden_loss)

                # Backward + optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(), qad_config.gradient_clip
                )
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Clear captured hiddens
                teacher_hiddens.clear()
                student_hiddens.clear()

                del inputs, teacher_outputs, student_outputs
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            print(f"  [Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")

        # Cleanup hooks
        for h in teacher_handles:
            h.remove()
        for h in student_handles:
            h.remove()

        student_model.eval()

        metrics = {
            "status": "completed",
            "num_epochs": num_epochs,
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
            "critical_layers_used": len(self.critical_layers),
            "neurons_masked": {
                k: len(v) if isinstance(v, list) else 0
                for k, v in self.neuron_mask.items()
            },
        }

        print(f"[Stage 3] Complete. Final loss: {metrics['final_loss']:.4f}")
        return metrics

    # =========================================================
    # STAGE 4: VALIDATION & EXPORT
    # =========================================================
    def run_stage_4_validation_export(
        self,
        test_cases: List[Tuple[str, str]],
        output_path: str = "profiles/niwt_profile.json",
        student_model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Validate extraction quality and export the complete NIWT profile.

        Validates by re-running the perturbation test (Stage 1 evaluation)
        and comparing baseline vs post-distillation performance. Exports
        the full profile (critical_layers, neuron_mask, config, metrics).

        Args:
            test_cases: List of (prompt, expected_target) tuples for validation
            output_path: Path to save the NIWT profile JSON
            student_model: Optional student model to validate (if Stage 3 was run)

        Returns:
            Dictionary with validation results and export path
        """
        print(f"\n[NIWT Stage 4] Starting Validation & Export...")

        results = {
            "validation": {},
            "profile_path": None,
        }

        # 4a. Validate teacher model baseline
        print("[Stage 4] Validating teacher baseline...")
        teacher_score = self._evaluate_capability_batched(test_cases)
        results["validation"]["teacher_score"] = teacher_score
        print(f"  Teacher score: {teacher_score:.2%}")

        # 4b. Validate student model if provided
        if student_model is not None:
            print("[Stage 4] Validating student model...")
            # Temporarily swap models for evaluation
            original_model = self.model
            self.model = student_model
            student_model.eval()

            student_score = self._evaluate_capability_batched(test_cases)
            results["validation"]["student_score"] = student_score
            results["validation"]["retention_rate"] = student_score / (
                teacher_score + 1e-9
            )
            print(f"  Student score: {student_score:.2%}")
            print(
                f"  Knowledge retention: {results['validation']['retention_rate']:.2%}"
            )

            # Restore teacher
            self.model = original_model

        # 4c. Build and export profile
        print("[Stage 4] Exporting NIWT profile...")

        # Serialize neuron mask (convert tensor indices to plain lists)
        serializable_mask = {}
        for layer_idx, mask_data in self.neuron_mask.items():
            if isinstance(mask_data, torch.Tensor):
                serializable_mask[str(layer_idx)] = mask_data.tolist()
            elif isinstance(mask_data, list):
                serializable_mask[str(layer_idx)] = mask_data
            else:
                serializable_mask[str(layer_idx)] = list(mask_data)

        profile = {
            "version": "1.0",
            "config": {
                k: v
                for k, v in self.config.items()
                if isinstance(v, (str, int, float, bool, list))
            },
            "critical_layers": self.critical_layers,
            "neuron_mask": serializable_mask,
            "validation": results["validation"],
            "num_critical_layers": len(self.critical_layers),
            "total_masked_neurons": sum(
                len(v) if isinstance(v, list) else 0 for v in self.neuron_mask.values()
            ),
        }

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)

        results["profile_path"] = output_path
        print(f"  Profile saved to: {output_path}")
        print(f"  Critical layers: {len(self.critical_layers)}")
        print(f"  Masked neurons: {profile['total_masked_neurons']}")

        print(f"[Stage 4] Complete.")
        return results

    # =========================================================
    # HELPERS
    # =========================================================
    def _get_model_layers(self):
        # Universal Inspector handles architecture differences
        return UniversalInspector.find_backbone_layers(self.model)

    def _evaluate_capability_batched(self, test_cases: List[Tuple[str, str]]) -> float:
        """
        Evaluate average success on test cases using Batch Processing.
        """
        dataset = EvaluationDataset(test_cases)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_correct = 0
        total_samples = 0

        for prompts, targets in loader:
            inputs = self.tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                # Generate small sample - Force Greedy to avoid Multinomial Crash on NaNs
                gen_kwargs = {
                    "max_new_tokens": 20,
                    "min_new_tokens": 5,
                    "repetition_penalty": 1.2,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,
                    "do_sample": False,
                }

                # Clean up sampling params if do_sample is False to avoid warnings
                if not gen_kwargs.get("do_sample", False):
                    gen_kwargs["top_p"] = None
                    gen_kwargs["top_k"] = None
                    gen_kwargs["temperature"] = None

                outputs = self.model.generate(**inputs, **gen_kwargs)

            decoded_batch = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Check correctness
            for i, response in enumerate(decoded_batch):
                # Simple containment check
                if targets[i].lower() in response.lower():
                    total_correct += 1

            total_samples += len(prompts)

            del inputs, outputs

        return total_correct / (total_samples + 1e-9)

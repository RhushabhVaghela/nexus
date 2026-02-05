import torch
import zlib
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from transformers import PreTrainedTokenizer, PreTrainedModel
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DistillationReport:
    """Report comparing hard vs soft distillation."""
    hard_distillation_rate: float
    soft_distillation_rate: float
    inherited_from_teacher_rate: float
    privacy_recommendation: str
    detailed_metrics: Dict[str, Any]


class MemorizationClassifier:
    """
    Pre-distillation memorization classifier using logistic regression.
    Based on paper 2601.15394 features: zlib entropy, teacher perplexity,
    baseline perplexity, teacher-baseline KLD.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_path = model_path
        self.target_auc_roc = 0.9997
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def extract_features(self, text: str, teacher_model: PreTrainedModel, 
                        baseline_model: Optional[PreTrainedModel] = None,
                        tokenizer: PreTrainedTokenizer = None,
                        device: str = "cuda") -> np.ndarray:
        """
        Extract features for memorization prediction.
        
        Features:
        1. Zlib entropy (normalized)
        2. Teacher perplexity
        3. Baseline perplexity (if available, else 0)
        4. Teacher-baseline KL divergence
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for feature extraction")
            
        # 1. Zlib entropy
        if not text:
            zlib_entropy = 0.0
        else:
            compressed = zlib.compress(text.encode('utf-8'))
            zlib_entropy = len(compressed) / len(text.encode('utf-8'))
        
        # Tokenize for model-based features
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 2. Teacher perplexity
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, labels=inputs["input_ids"])
            teacher_loss = teacher_outputs.loss.item()
            teacher_perplexity = np.exp(teacher_loss)
        
        # 3. Baseline perplexity (if available)
        baseline_perplexity = 0.0
        kld = 0.0
        
        if baseline_model is not None:
            with torch.no_grad():
                baseline_outputs = baseline_model(**inputs, labels=inputs["input_ids"])
                baseline_loss = baseline_outputs.loss.item()
                baseline_perplexity = np.exp(baseline_loss)
                
                # 4. KL Divergence between teacher and baseline logits
                teacher_logits = teacher_outputs.logits
                baseline_logits = baseline_outputs.logits
                
                # Compute KL divergence
                teacher_probs = torch.softmax(teacher_logits, dim=-1)
                baseline_log_probs = torch.log_softmax(baseline_logits, dim=-1)
                kld = torch.sum(teacher_probs * (torch.log(teacher_probs) - baseline_log_probs))
                kld = kld.item()
        
        features = np.array([
            zlib_entropy,
            teacher_perplexity,
            baseline_perplexity,
            kld
        ])
        
        return features
    
    def train(self, texts: List[str], labels: List[int], 
              teacher_model: PreTrainedModel,
              baseline_model: Optional[PreTrainedModel] = None,
              tokenizer: PreTrainedTokenizer = None,
              device: str = "cuda") -> Dict[str, float]:
        """
        Train the memorization classifier.
        
        Args:
            texts: List of text samples
            labels: Binary labels (1 = memorized, 0 = not memorized)
            teacher_model: Teacher model for feature extraction
            baseline_model: Smaller baseline model for comparison
            tokenizer: Tokenizer for models
            device: Device for computation
            
        Returns:
            Training metrics including AUC-ROC
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
        
        logger.info(f"Extracting features for {len(texts)} samples...")
        
        # Extract features for all samples
        X = []
        for text in texts:
            features = self.extract_features(
                text, teacher_model, baseline_model, tokenizer, device
            )
            X.append(features)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression
        logger.info("Training logistic regression classifier...")
        self.model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        auc_roc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        self.is_trained = True
        
        metrics = {
            "auc_roc": auc_roc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "target_achieved": auc_roc >= self.target_auc_roc
        }
        
        logger.info(f"Training complete. AUC-ROC: {auc_roc:.4f} (target: {self.target_auc_roc:.4f})")
        
        return metrics
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict memorization probability."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained or loaded before prediction")
        
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        
        return self.model.predict_proba(features)[0]
    
    def predict(self, features: np.ndarray) -> int:
        """Predict memorization class (0 or 1)."""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained or loaded before prediction")
        
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        
        return self.model.predict(features.reshape(1, -1))[0]
    
    def save(self, path: str):
        """Save the trained classifier."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'target_auc_roc': self.target_auc_roc
            }, f)
        logger.info(f"Classifier saved to {path}")
    
    def load(self, path: str):
        """Load a trained classifier."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.target_auc_roc = data.get('target_auc_roc', 0.9997)
        logger.info(f"Classifier loaded from {path}")


class MemorizationAuditor:
    """
    Implements memorization metrics inspired by arXiv:2601.15394.
    
    1. Zlib Entropy: Predicts memorizability based on compressibility.
    2. Discoverable Memorization: Exact match check for greedy generation.
    3. Pre-distillation Memorization Classifier.
    4. Hard vs Soft Distillation Analysis.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, prefix_len: int = 50, suffix_len: int = 50):
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.classifier = MemorizationClassifier()

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
    
    def predict_memorization_risk(self, text: str, teacher_model: PreTrainedModel,
                                  baseline_model: Optional[PreTrainedModel] = None,
                                  device: str = "cuda") -> Dict[str, Any]:
        """
        Predict memorization risk for a text sample using the classifier.
        
        Args:
            text: Text sample to evaluate
            teacher_model: Teacher model for feature extraction
            baseline_model: Optional smaller baseline model
            device: Device for computation
            
        Returns:
            Dictionary with risk score, prediction, and features
        """
        if not self.classifier.is_trained:
            raise RuntimeError(
                "Memorization classifier must be trained before prediction. "
                "Call train_memorization_classifier() first or load a pre-trained model."
            )
        
        features = self.classifier.extract_features(
            text, teacher_model, baseline_model, self.tokenizer, device
        )
        
        risk_prob = self.classifier.predict_proba(features)[1]  # Probability of class 1 (memorized)
        prediction = self.classifier.predict(features)
        
        return {
            "risk_score": float(risk_prob),
            "is_high_risk": bool(prediction == 1),
            "features": {
                "zlib_entropy": float(features[0]),
                "teacher_perplexity": float(features[1]),
                "baseline_perplexity": float(features[2]),
                "teacher_baseline_kld": float(features[3])
            }
        }
    
    def train_memorization_classifier(self, texts: List[str], labels: List[int],
                                      teacher_model: PreTrainedModel,
                                      baseline_model: Optional[PreTrainedModel] = None,
                                      device: str = "cuda") -> Dict[str, float]:
        """
        Train the memorization classifier.
        
        Args:
            texts: List of text samples
            labels: Binary labels (1 = memorized, 0 = not memorized)
            teacher_model: Teacher model
            baseline_model: Optional baseline model
            device: Device for computation
            
        Returns:
            Training metrics
        """
        return self.classifier.train(
            texts, labels, teacher_model, baseline_model, self.tokenizer, device
        )
    
    def save_classifier(self, path: str):
        """Save the trained memorization classifier."""
        self.classifier.save(path)
    
    def load_classifier(self, path: str):
        """Load a pre-trained memorization classifier."""
        self.classifier = MemorizationClassifier(path)
    
    def analyze_hard_vs_soft_distillation(self, 
                                          student_model: PreTrainedModel,
                                          teacher_model: PreTrainedModel,
                                          texts: List[str],
                                          device: str = "cuda") -> DistillationReport:
        """
        Analyze and compare hard vs soft distillation in terms of memorization.
        
        Hard distillation: Student learns from teacher's argmax predictions
        Soft distillation: Student learns from teacher's full probability distribution
        
        Args:
            student_model: The student model
            teacher_model: The teacher model
            texts: List of text samples to analyze
            device: Device for computation
            
        Returns:
            DistillationReport with comparison metrics
        """
        hard_memorization_count = 0
        soft_memorization_count = 0
        inherited_count = 0
        
        detailed_metrics = {
            "samples_analyzed": len(texts),
            "hard_matches": [],
            "soft_matches": [],
            "teacher_student_agreement": []
        }
        
        for text in texts:
            tokens = self.tokenizer.encode(text, return_tensors="pt").to(device)
            
            if tokens.shape[1] < self.prefix_len + self.suffix_len:
                continue
            
            prefix = tokens[:, :self.prefix_len]
            ground_truth = tokens[:, self.prefix_len:self.prefix_len + self.suffix_len]
            
            # Hard distillation analysis: greedy generation from student
            with torch.no_grad():
                student_hard = student_model.generate(
                    prefix,
                    max_new_tokens=self.suffix_len,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Soft distillation analysis: sample from distribution
                student_soft = student_model.generate(
                    prefix,
                    max_new_tokens=self.suffix_len,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Teacher generation
                teacher_output = teacher_model.generate(
                    prefix,
                    max_new_tokens=self.suffix_len,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Check matches
            student_hard_suffix = student_hard[0, self.prefix_len:].tolist()
            student_soft_suffix = student_soft[0, self.prefix_len:].tolist()
            teacher_suffix = teacher_output[0, self.prefix_len:].tolist()
            ground_truth_suffix = ground_truth[0].tolist()
            
            hard_match = student_hard_suffix == ground_truth_suffix
            soft_match = student_soft_suffix == ground_truth_suffix
            teacher_match = teacher_suffix == ground_truth_suffix
            
            if hard_match:
                hard_memorization_count += 1
            if soft_match:
                soft_memorization_count += 1
            if student_soft_suffix == teacher_suffix:
                inherited_count += 1
            
            detailed_metrics["hard_matches"].append(hard_match)
            detailed_metrics["soft_matches"].append(soft_match)
            detailed_metrics["teacher_student_agreement"].append(student_soft_suffix == teacher_suffix)
        
        total = len(texts)
        hard_rate = hard_memorization_count / total if total > 0 else 0.0
        soft_rate = soft_memorization_count / total if total > 0 else 0.0
        inherited_rate = inherited_count / total if total > 0 else 0.0
        
        # Recommendation based on analysis
        if hard_rate > soft_rate * 1.5:
            recommendation = "SOFT_DISTILLATION_RECOMMENDED"
        elif soft_rate > 0.3:
            recommendation = "ADDITIONAL_PRIVACY_SAFEGUARDS_NEEDED"
        else:
            recommendation = "BOTH_METHODS_ACCEPTABLE"
        
        return DistillationReport(
            hard_distillation_rate=hard_rate,
            soft_distillation_rate=soft_rate,
            inherited_from_teacher_rate=inherited_rate,
            privacy_recommendation=recommendation,
            detailed_metrics=detailed_metrics
        )
    
    def generate_distillation_report(self, report: DistillationReport, output_path: str):
        """Generate a detailed comparison report."""
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HARD VS SOFT DISTILLATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Samples Analyzed: {report.detailed_metrics['samples_analyzed']}\n")
            f.write(f"\nMemorization Rates:\n")
            f.write(f"  Hard Distillation:  {report.hard_distillation_rate:.4f}\n")
            f.write(f"  Soft Distillation:  {report.soft_distillation_rate:.4f}\n")
            f.write(f"  Teacher Inheritance: {report.inherited_from_teacher_rate:.4f}\n")
            
            f.write(f"\nPrivacy Recommendation: {report.privacy_recommendation}\n")
            
            if report.privacy_recommendation == "SOFT_DISTILLATION_RECOMMENDED":
                f.write("\nNote: Hard distillation shows significantly higher memorization.\n")
                f.write("Consider using soft distillation with temperature scheduling\n")
                f.write("to reduce verbatim copying from the teacher model.\n")
            elif report.privacy_recommendation == "ADDITIONAL_PRIVACY_SAFEGUARDS_NEEDED":
                f.write("\nWarning: Both methods show high memorization rates.\n")
                f.write("Consider implementing:\n")
                f.write("  - Data filtering with memorization classifier\n")
                f.write("  - Differential privacy during training\n")
                f.write("  - Stricter temperature schedules\n")
        
        logger.info(f"Distillation report saved to {output_path}")


# Convenience function for CLI usage
def create_auditor(tokenizer_path: str, **kwargs) -> MemorizationAuditor:
    """Factory function to create a MemorizationAuditor from a tokenizer path."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return MemorizationAuditor(tokenizer, **kwargs)

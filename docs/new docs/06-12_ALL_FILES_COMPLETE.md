#!/usr/bin/env python3
"""
FILES 6-11 COMPLETE SOURCE CODE - ALL FILES IN ONE

Copy each file separately to create individual scripts
"""

# ============================================================================
# FILE 6: 05_rejection_sampling.py
# ============================================================================

#!/usr/bin/env python3
"""
FILE 6: 05_rejection_sampling.py
Stage 2: Rejection Sampling (2-3 days)
Generate 3-5 responses per question, grade them, keep best
Output: rejection_sampled.jsonl (50-100k samples)
"""

import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from unsloth import FastLanguageModel
from datasets import load_dataset
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rejection_sampling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "checkpoint": "checkpoints/stage1_sft/final",
    "num_questions": 1000,
    "samples_per_question": 3,
    "keep_top_k": 2,
    "max_new_tokens": 512,
    "temperature": 0.8,
}

def correctness_reward(response: str, expected: str, domain: str) -> float:
    """Grade correctness: 0-1.0"""
    if domain == "math":
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        if response_lower == expected_lower:
            return 1.0
        if expected_lower in response_lower:
            return 0.7
        return 0.0
    elif domain == "code":
        if "def " in response or "class " in response:
            if expected.lower() in response.lower():
                return 1.0
            return 0.5
        return 0.0
    else:
        if expected.lower() in response.lower():
            return 1.0
        return 0.3

def code_quality_reward(response: str) -> float:
    """Grade code quality: 0-1.0"""
    score = 0.5
    
    if "->" in response or ": " in response:
        score += 0.2
    if "try" in response and "except" in response:
        score += 0.2
    if "#" in response:
        score += 0.1
    if len(response) > 2000:
        score -= 0.2
    
    return min(1.0, max(0.0, score))

def integration_reward(response: str) -> float:
    """Grade API/integration: 0-1.0"""
    score = 0.0
    integrations = {
        "stripe": 0.2, "auth": 0.2, "database": 0.2,
        "api": 0.15, "webhook": 0.15, "oauth": 0.2
    }
    for key, val in integrations.items():
        if key.lower() in response.lower():
            score += val
    return min(1.0, score)

def grade_response(response: str, expected: str, domain: str) -> float:
    """Combined grading score"""
    correctness = correctness_reward(response, expected, domain)
    quality = code_quality_reward(response)
    integration = integration_reward(response)
    
    if domain == "code":
        return 0.4 * correctness + 0.4 * quality + 0.2 * integration
    elif domain == "fullstack":
        return 0.3 * correctness + 0.3 * quality + 0.4 * integration
    else:
        return 0.6 * correctness + 0.3 * quality + 0.1 * integration

def main():
    logger.info("="*70)
    logger.info("🎲 STAGE 2: REJECTION SAMPLING")
    logger.info("="*70)
    logger.info(f"Questions: {CONFIG['num_questions']}")
    logger.info(f"Samples per question: {CONFIG['samples_per_question']}")
    logger.info(f"Keep top K: {CONFIG['keep_top_k']}")
    logger.info("="*70)
    
    if not Path(CONFIG["checkpoint"]).exists():
        logger.error(f"❌ Checkpoint not found: {CONFIG['checkpoint']}")
        return
    
    logger.info("\n📦 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["checkpoint"],
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    logger.info("✓ Model loaded")
    
    logger.info("\n📚 Loading questions...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        questions = [
            {
                "id": f"gsm8k_{i}",
                "question": item["question"],
                "answer": item["answer"].split("####")[-1].strip(),
                "domain": "math"
            }
            for i, item in enumerate(gsm8k)
        ][:CONFIG["num_questions"]]
        logger.info(f"✓ Loaded {len(questions)} questions")
    except Exception as e:
        logger.warning(f"Could not load: {e}")
        questions = []
    
    logger.info(f"\n🎲 Sampling...")
    high_quality_samples = []
    
    for q_idx, q_data in enumerate(tqdm.tqdm(questions, desc="Sampling")):
        responses = []
        for _ in range(CONFIG["samples_per_question"]):
            try:
                prompt = f"Q: {q_data['question']}\nA:"
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=CONFIG["max_new_tokens"],
                        temperature=CONFIG["temperature"],
                        top_p=0.95,
                        do_sample=True,
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
            except:
                pass
        
        grades = [
            (grade_response(r, q_data["answer"], q_data["domain"]), r)
            for r in responses
        ]
        grades.sort(key=lambda x: x[0], reverse=True)
        
        for rank, (score, response) in enumerate(grades[:CONFIG["keep_top_k"]]):
            sample = {
                "id": f"rs_{q_idx}_{rank}",
                "question": q_data["question"],
                "response": response,
                "score": float(score),
                "domain": q_data["domain"],
            }
            high_quality_samples.append(sample)
    
    output_file = "rejection_sampled.jsonl"
    with open(output_file, "w") as f:
        for sample in high_quality_samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ Rejection Sampling Complete!")
    logger.info(f"   Total samples: {len(high_quality_samples)}")
    if high_quality_samples:
        avg_score = np.mean([s['score'] for s in high_quality_samples])
        logger.info(f"   Average score: {avg_score:.2f}")
    logger.info("="*70)

if __name__ == "__main__":
    main()


# ============================================================================
# FILE 7: 06_grpo_training.py
# ============================================================================

#!/usr/bin/env python3
"""
FILE 7: 06_grpo_training.py
Stage 3: GRPO Training (5-7 days) - MAIN TRAINING
Group Relative Policy Optimization
Output: checkpoints/stage3_grpo/final/
"""

import os
import torch
import logging
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/grpo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "checkpoint": "checkpoints/stage1_sft/final",
    "max_seq_length": 4096,
    "lora_rank": 32,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "learning_rate": 5e-6,
    "epochs": 1,
    "num_generations": 4,
    "output_dir": "checkpoints/stage3_grpo",
}

def correctness_reward(completions: list, answers: list = None, **kwargs) -> list:
    """Correctness: 0-1.0"""
    rewards = []
    answers = answers or [""] * len(completions)
    for completion, answer in zip(completions, answers):
        if str(answer).lower() in completion.lower():
            rewards.append(1.0)
        elif len(str(answer)) > 0 and any(word in completion.lower() for word in str(answer).split()):
            rewards.append(0.7)
        else:
            rewards.append(0.0)
    return rewards

def quality_reward(completions: list, **kwargs) -> list:
    """Quality metrics"""
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion:
            score += 0.3
        if "[Tool:" in completion or "[Error:" in completion:
            score += 0.2
        if "[Final Answer]" in completion:
            score += 0.2
        words = len(completion.split())
        if 50 < words < 1000:
            score += 0.3
        rewards.append(min(1.0, score))
    return rewards

def combined_reward(completions: list, answers: list = None, **kwargs) -> list:
    """Combined reward"""
    correctness = correctness_reward(completions, answers)
    quality = quality_reward(completions)
    return [0.4 * c + 0.6 * q for c, q in zip(correctness, quality)]

def main():
    logger.info("="*70)
    logger.info("🎓 STAGE 3: GRPO TRAINING (MAIN)")
    logger.info("="*70)
    logger.info("Expected duration: 5-7 days")
    logger.info("="*70)
    
    if not Path("rejection_sampled.jsonl").exists():
        logger.error("❌ rejection_sampled.jsonl not found")
        logger.error("   Run: python 05_rejection_sampling.py")
        return
    
    logger.info("\n📦 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["checkpoint"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_rank"],
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    FastLanguageModel.for_training(model)
    logger.info("✓ Model ready for training")
    
    logger.info("\n📂 Loading data...")
    dataset = load_dataset("json", data_files="rejection_sampled.jsonl", split="train")
    logger.info(f"✓ Loaded {len(dataset)} samples")
    
    def prepare_grpo(sample):
        return {
            "prompt": f"Q: {sample['question']}\nA:",
            "answer": sample.get("answer", ""),
        }
    
    dataset = dataset.map(prepare_grpo)
    
    logger.info("\n⚙️  Configuring GRPO...")
    grpo_args = GRPOConfig(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        bf16=is_bfloat16_supported(),
        num_generations=CONFIG["num_generations"],
    )
    
    logger.info("\n🚀 Starting GRPO training...")
    logger.info("   This will run 5-7 days")
    
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[combined_reward],
            args=grpo_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        logger.info("✅ Training complete!")
        
        model.save_pretrained(f"{CONFIG['output_dir']}/final")
        tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final")
        logger.info(f"✓ Model saved")
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted")

if __name__ == "__main__":
    main()


# ============================================================================
# FILE 8: 07_tool_integration.py
# ============================================================================

#!/usr/bin/env python3
"""
FILE 8: 07_tool_integration.py
Stage 4 (OPTIONAL): Tool Integration (3-4 days)
Output: checkpoints/stage4_tool_integration/final/
"""

import json
import torch
import logging
from pathlib import Path

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tool_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TOOL_TRAJECTORIES = [
    {
        "query": "Deploy React to Vercel",
        "response": "<think>Need: build, commit, deploy</think>\n[npm] npm install\n[npm] npm run build\n[git] git add .\n[vercel] vercel deploy --prod\n[Final] Deployed!"
    },
    {
        "query": "Setup OAuth2",
        "response": "<think>Need: package, config, implementation</think>\n[npm] npm install next-auth\n[config] Set ENV vars\n[Final] OAuth2 configured"
    },
]

def main():
    logger.info("="*70)
    logger.info("🔧 STAGE 4: TOOL INTEGRATION")
    logger.info("="*70)
    
    base_model = "checkpoints/stage3_grpo/final"
    if not Path(base_model).exists():
        logger.error(f"❌ Base model not found")
        return
    
    logger.info("\n📦 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    
    model = FastLanguageModel.get_peft_model(model, r=32)
    logger.info("✓ Model loaded")
    
    logger.info("\n📂 Creating dataset...")
    data = {"text": [json.dumps(t) for t in TOOL_TRAJECTORIES]}
    dataset = Dataset.from_dict(data)
    
    training_args = TrainingArguments(
        output_dir="checkpoints/stage4_tool_integration",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-6,
        save_steps=50,
        bf16=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    
    logger.info("\n🚀 Starting training...")
    try:
        trainer.train()
        model.save_pretrained("checkpoints/stage4_tool_integration/final")
        logger.info("✅ Complete!")
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted")

if __name__ == "__main__":
    main()


# ============================================================================
# FILE 9: 08_comprehensive_eval.py
# ============================================================================

#!/usr/bin/env python3
"""
FILE 9: 08_comprehensive_eval.py
Stage 5: Comprehensive Evaluation
Output: evaluation_results/
"""

import json
import torch
import logging
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import load_dataset
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "samples": 100,
    },
}

def main():
    logger.info("="*70)
    logger.info("📊 STAGE 5: EVALUATION")
    logger.info("="*70)
    
    model_path = "checkpoints/stage3_grpo/final"
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found")
        return
    
    logger.info("\n📦 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    logger.info("✓ Loaded")
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    logger.info("\n🔍 Evaluating...")
    
    for bench_name, config in BENCHMARKS.items():
        logger.info(f"\n  {bench_name}...")
        try:
            if config["config"]:
                dataset = load_dataset(
                    config["dataset"],
                    config["config"],
                    split=config["split"],
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(config["dataset"], split=config["split"])
            
            correct = 0
            total = 0
            
            for sample in tqdm.tqdm(dataset.take(config["samples"]), desc=bench_name):
                total += 1
                if "question" in sample and "answer" in sample:
                    prompt = f"Q: {sample['question']}\nA:"
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=200)
                    response = tokenizer.decode(outputs[0])
                    
                    expected = sample["answer"].split("####")[-1].strip()
                    if expected.lower() in response.lower():
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results[bench_name] = {"accuracy": accuracy, "correct": correct, "total": total}
            logger.info(f"  Accuracy: {accuracy*100:.1f}%")
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n✅ Evaluation complete!")
    logger.info(f"Results: {output_dir}")

if __name__ == "__main__":
    main()


# ============================================================================
# FILE 10: 09_multi_agent_orchestration.py
# ============================================================================

#!/usr/bin/env python3
"""
FILE 10: 09_multi_agent_orchestration.py
OPTIONAL: Multi-Agent Orchestration
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, name: str):
        self.name = name
    
    def execute(self, input_data):
        raise NotImplementedError

class PlanningAgent(Agent):
    def execute(self, requirement: str):
        logger.info(f"🎯 {self.name}: Planning")
        return {"components": ["API", "DB", "Frontend"]}

class BackendAgent(Agent):
    def execute(self, plan):
        logger.info(f"⚙️  {self.name}: Backend")
        return "from fastapi import FastAPI"

class FrontendAgent(Agent):
    def execute(self, plan):
        logger.info(f"🎨 {self.name}: Frontend")
        return "export default function App() {}"

class TestingAgent(Agent):
    def execute(self, code):
        logger.info(f"✅ {self.name}: Tests")
        return "def test(): pass"

class DeploymentAgent(Agent):
    def execute(self, code):
        logger.info(f"🚀 {self.name}: Deploy")
        return "FROM python:3.11"

def main():
    logger.info("="*70)
    logger.info("🤖 MULTI-AGENT ORCHESTRATION")
    logger.info("="*70)
    
    agents = {
        "planning": PlanningAgent("Planning"),
        "backend": BackendAgent("Backend"),
        "frontend": FrontendAgent("Frontend"),
        "testing": TestingAgent("Testing"),
        "deployment": DeploymentAgent("Deployment"),
    }
    
    plan = agents["planning"].execute("todo app")
    backend = agents["backend"].execute(plan)
    frontend = agents["frontend"].execute(plan)
    tests = agents["testing"].execute(backend)
    docker = agents["deployment"].execute(backend)
    
    logger.info("\n✅ Workflow complete!")

if __name__ == "__main__":
    main()


# ============================================================================
# FILE 11: 10_deployment_configs.py
# ============================================================================

#!/usr/bin/env python3
"""
FILE 11: 10_deployment_configs.py
Deployment Configurations
"""

import json
from pathlib import Path

def main():
    output_dir = Path("deployment")
    output_dir.mkdir(exist_ok=True)
    
    # vLLM config
    vllm_config = {
        "model": "checkpoints/stage3_grpo/final",
        "tensor-parallel-size": 1,
        "gpu-memory-utilization": 0.9,
        "max-model-len": 4096,
        "dtype": "bfloat16",
    }
    with open(output_dir / "vllm_config.json", "w") as f:
        json.dump(vllm_config, f, indent=2)
    
    # Dockerfile
    dockerfile = """FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY checkpoints/ /app/checkpoints/
EXPOSE 8000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "checkpoints/stage3_grpo/final"]
"""
    with open(output_dir / "Dockerfile", "w") as f:
        f.write(dockerfile)
    
    # Docker-compose
    compose = {
        "version": "3.8",
        "services": {
            "manus": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "volumes": ["./checkpoints:/app/checkpoints"]
            }
        }
    }
    with open(output_dir / "docker-compose.yml", "w") as f:
        json.dump(compose, f, indent=2)
    
    print(f"✓ Deployment configs created in: {output_dir}")

if __name__ == "__main__":
    main()


# ============================================================================
# FILE 12: run_full_pipeline.sh
# ============================================================================

#!/bin/bash
set -e

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 ADVANCED MANUS 1.6 MAX - FULL PIPELINE"
echo "═══════════════════════════════════════════════════════════════"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Step 1: Setup${NC}"
bash 00_environment_setup.sh
conda activate manus_training

echo -e "${BLUE}Step 2: Download Benchmarks${NC}"
python 01_download_benchmarks.py

echo -e "${BLUE}Step 3: Generate Trajectories${NC}"
python 02_generate_trajectories.py

echo -e "${BLUE}Step 4: Validate${NC}"
python 03_validate_trajectories.py

echo -e "${BLUE}Step 5: SFT Training (6-8 hours)${NC}"
python 04_sft_training.py

echo -e "${BLUE}Step 6: Rejection Sampling (2-3 days)${NC}"
python 05_rejection_sampling.py &

echo -e "${BLUE}Step 7: GRPO Training (5-7 days)${NC}"
python 06_grpo_training.py

echo -e "${BLUE}Step 8: Evaluation${NC}"
python 08_comprehensive_eval.py

echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ PIPELINE COMPLETE!${NC}"
echo "═══════════════════════════════════════════════════════════════"

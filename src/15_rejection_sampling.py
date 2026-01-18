#!/usr/bin/env python3
"""
Stage 2: Rejection Sampling (2-3 days)
Generate 3-5 responses per question, grade them, keep best
Output: rejection_sampled.jsonl (50-100k samples)
"""

import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
import os

from unsloth import FastLanguageModel
from datasets import load_dataset
import tqdm

# Create logs directory if it doesn't exist
try:
    os.makedirs('logs', exist_ok=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

def parse_json_response(response: str) -> List[Dict[str, Any]]:
    """Attempt to parse model's JSON response"""
    try:
        # cleanup markdown code blocks if model adds them
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except:
        return []

def correctness_reward(response_steps: List[Dict], expected: str, domain: str) -> float:
    """Grade correctness: 0-1.0 based on parsed steps"""
    if not response_steps: return 0.0 # Failed to parse
    
    # Extract final answer or last action info
    final_content = ""
    for step in reversed(response_steps):
        if step.get("type") == "final_answer":
            final_content = step.get("content", "")
            break
        if "action" in step:
            final_content += step.get("action", "") + " "
            
    if not final_content:
        # Fallback: check raw text of last step
        final_content = str(response_steps[-1])

    if domain == "math":
        if expected.lower().strip() in final_content.lower().strip():
            return 1.0
        return 0.0
    elif domain == "code":
        if "def " in final_content or "class " in final_content:
            return 1.0
        return 0.0
    else:  # fullstack
        if len(final_content) > 20: # Heuristic for meaningful output
            return 1.0
        return 0.3

def code_quality_reward(response_steps: List[Dict]) -> float:
    """Grade code quality from actions"""
    score = 0.5
    if not response_steps: return 0.0
    
    code_content = ""
    for step in response_steps:
        if "input" in step: # 'input' often contains code in write_file
            code_content += str(step.get("input", ""))
            
    if "->" in code_content or ": " in code_content: score += 0.2
    if "try" in code_content: score += 0.2
    
    return min(1.0, score)

def integration_reward(response_steps: List[Dict]) -> float:
    """Check for tool usage"""
    score = 0.0
    if not response_steps: return 0.0
    
    tools_used = set()
    for step in response_steps:
        if "action" in step:
            tools_used.add(step.get("action", "").split(":")[0]) # rough tool name
            
    if len(tools_used) > 1: score += 0.5
    if len(tools_used) > 2: score += 0.5
    
    return min(1.0, score)

def grade_response(response: str, expected: str, domain: str) -> float:
    """Combined grading score"""
    steps = parse_json_response(response)
    if not steps: 
        return 0.0 # Immediate fail if not valid JSON
        
    correctness = correctness_reward(steps, expected, domain)
    quality = code_quality_reward(steps)
    integration = integration_reward(steps)
    
    # Domain-specific weights
    if domain == "code":
        return 0.4 * correctness + 0.4 * quality + 0.2 * integration
    elif domain == "fullstack":
        return 0.3 * correctness + 0.3 * quality + 0.4 * integration
    else:  # reasoning
        return 0.6 * correctness + 0.3 * quality + 0.1 * integration

def sample_responses(model, tokenizer, question: str, num_samples: int) -> List[str]:
    """Generate multiple responses for question using Chat Template"""
    responses = []
    
    # System prompt is critical for JSON output
    sys_prompt = "You are an advanced AI. Solve the task step-by-step. Output ONLY a valid JSON list of steps."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    for _ in range(num_samples):
        try:
            inputs = tokenizer(text_input, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    temperature=CONFIG["temperature"],
                    top_p=0.95,
                    do_sample=True,
                )
            # Decode ONLY the new tokens
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            responses.append(response)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            continue
    
    return responses

def main():
    logger.info("="*70)
    logger.info("🎲 STAGE 2: REJECTION SAMPLING")
    logger.info("="*70)
    logger.info(f"Questions to sample: {CONFIG['num_questions']}")
    logger.info(f"Samples per question: {CONFIG['samples_per_question']}")
    logger.info(f"Keep top K: {CONFIG['keep_top_k']}")
    logger.info(f"Expected duration: 2-3 days")
    logger.info("="*70)
    
    # Load model
    logger.info("\n📦 Loading SFT model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["checkpoint"],
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model from {CONFIG['checkpoint']}: {e}")
        return
    
    # Load benchmark questions
    logger.info("\n📚 Loading benchmark questions...")
    try:
        # Load our own filtered cold start data as the source for rejection sampling
        # (Since we want to improve OUR domain performance, not just GSM8K)
        # But for 'main' benchmark compliance we kept GSM8K in original.
        # Here we will switch to loading a mix: GSM8K + subset of our generated data.
        
        # 1. GSM8K (Math)
        questions = []
        try:
            gsm8k = load_dataset("openai/gsm8k", "main", split="train")
            questions.extend([
                {
                    "id": f"gsm8k_{i}",
                    "user_query": item["question"], # Standardize to user_query
                    "answer": item["answer"].split("####")[-1].strip(),
                    "domain": "math"
                }
                for i, item in enumerate(gsm8k)
            ][:500]) # Take 500
        except Exception:
            pass

        # 2. Our Generated Data (Fullstack/Replica)
        # We load a subset of the validated training data
        import glob
        # UPDATE: Support clean datasets/ folder structure
        replica_files = glob.glob("E:/datasets/train/**/*_validated.jsonl", recursive=True)
        if not replica_files:
             replica_files = glob.glob("data_train_*_validated.jsonl") # Fallback
             
        count_loaded = 0
        TARGET_COUNT = 500
        
        for f_path in replica_files:
            if count_loaded >= TARGET_COUNT: break
            
            with open(f_path) as f:
                for line in f:
                    if count_loaded >= TARGET_COUNT: break
                    try:
                        d = json.loads(line)
                        questions.append({
                            "id": f"gen_{d.get('id')}",
                            "user_query": d.get("user_query"),
                            "answer": "", 
                            "domain": d.get("domain")
                        })
                        count_loaded += 1
                    except:
                        pass
        
        logger.info(f"✓ Loaded {len(questions)} questions (Mix of GSM8K + Replica)")
    except Exception as e:
        logger.warning(f"Could not load questions: {e}")
        questions = []
    
    # Rejection sampling with pause support
    logger.info(f"\n🎲 Sampling {len(questions)} questions...")
    logger.info("💡 Tip: Run 'python3 utils/control_training.py --flag-dir flags' to pause")
    
    # Pause/Resume Support
    FLAG_DIR = "flags"
    os.makedirs(FLAG_DIR, exist_ok=True)
    pause_file = os.path.join(FLAG_DIR, "pause.flag")
    progress_file = "rejection_sampling_progress.json"
    
    # Load previous progress if exists
    start_idx = 0
    high_quality_samples = []
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as f:
                progress = json.load(f)
                start_idx = progress.get("last_index", 0)
                high_quality_samples = progress.get("samples", [])
                logger.info(f"🔄 Resuming from question {start_idx}")
        except:
            pass
    
    grade_dist = defaultdict(int)
    
    for q_idx, q_data in enumerate(tqdm.tqdm(questions[start_idx:], desc="Sampling", initial=start_idx)):
        # Check for pause every 10 questions
        if q_idx % 10 == 0 and os.path.exists(pause_file):
            logger.info(f"\n💾 Pause detected at question {start_idx + q_idx}. Saving progress...")
            with open(progress_file, "w") as f:
                json.dump({"last_index": start_idx + q_idx, "samples": high_quality_samples}, f)
            logger.info("✓ Progress saved. Exiting...")
            return
        
        q_idx = start_idx + q_idx  # Adjust index
        question = q_data["user_query"]
        expected = q_data["answer"]
        domain = q_data["domain"]
        
        # Generate responses
        responses = sample_responses(model, tokenizer, question, CONFIG["samples_per_question"])
        
        # Grade each
        grades = [
            (grade_response(r, expected, domain), r)
            for r in responses
        ]
        grades.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top K
        for rank, (score, response) in enumerate(grades[:CONFIG["keep_top_k"]]):
            sample = {
                "id": f"rs_{q_idx}_{rank}",
                "question": question,
                "response": response,
                "score": float(score),
                "domain": domain,
                "rank": rank,
            }
            high_quality_samples.append(sample)
            grade_dist[int(score * 10)] += 1
    
    # Save
    output_file = "rejection_sampled.jsonl"
    with open(output_file, "w") as f:
        for sample in high_quality_samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info("="*70)
    logger.info(f"✅ Rejection Sampling Complete!")
    logger.info(f"   Total samples: {len(high_quality_samples)}")
    if len(high_quality_samples) > 0:
        logger.info(f"   Average score: {np.mean([s['score'] for s in high_quality_samples]):.2f}")
    logger.info("="*70)
    logger.info(f"Next: Run GRPO Training")
    logger.info(f"  python 06_grpo_training.py")

if __name__ == "__main__":
    main()

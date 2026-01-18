#!/usr/bin/env python3
"""
Stage 4 (OPTIONAL): Tool Integration Fine-Tuning (3-4 days)
Learn to use npm, pip, API calls, Docker, deployment
AND Research/Vision Tools.
Output: checkpoints/stage4_tool_integration/final/
"""

import json
import torch
import logging
from pathlib import Path
from typing import List, Dict
import os

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

# Create logs directory if it doesn't exist
try:
    os.makedirs('logs', exist_ok=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tool_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced Tool Trajectories
TOOL_TRAJECTORIES = [
    {
        "query": "Deploy a React app to Vercel",
        "tools": ["npm", "git", "vercel"],
        "trajectory": [
            "<think>Need to build, commit, and deploy</think>",
            "[Tool: npm] npm install",
            "[Result] ✓ Dependencies installed",
            "[Tool: npm] npm run build",
            "[Result] ✓ Built successfully",
            "[Tool: git] git add .",
            "[Tool: git] git commit -m 'Deploy'",
            "[Tool: vercel] vercel deploy --prod",
            "[Result] ✓ Deployed to https://app.vercel.app",
            "[Final Answer] App successfully deployed!"
        ]
    },
    {
        "query": "Check latest LangChain docs for Agents",
        "tools": ["browser"],
        "trajectory": [
            "<think>User asked for latest docs. I should browse the official documentation.</think>",
            "[Tool: browser] search_google 'langchain agents documentation latest'",
            "[Observation] Found https://python.langchain.com/docs/modules/agents/",
            "[Tool: browser] read_url https://python.langchain.com/docs/modules/agents/",
            "[Observation] LangChain Agents use LCEL...",
            "[Final Answer] The latest LangChain Agents documentation indicates usage of LCEL..."
        ]
    },
    {
        "query": "Create a React Counter component",
        "tools": ["create_artifact", "update_artifact"],
        "trajectory": [
            "<think>I need to create a new artifact for the React component.</think>",
            "[Tool: create_artifact] create_artifact(id='counter', type='react', title='Counter Component')",
            "[Result] Artifact 'counter' created",
            "<think>Now I will write the code.</think>",
            "[Tool: update_artifact] update_artifact(id='counter', content='export default function Counter() {...}')",
            "[Result] Artifact updated",
            "[Final Answer] I have created the Counter component in the artifact."
        ]
    }
]

def create_tool_dataset() -> Dataset:
    """Create dataset from tool trajectories"""
    data = []
    for i, traj in enumerate(TOOL_TRAJECTORIES):
        data.append({
            "id": f"tool_{i}",
            "query": traj["query"],
            "response": "\n".join(traj["trajectory"]),
            "tools": ",".join(traj["tools"])
        })
    
    return Dataset.from_dict({
        "text": [json.dumps({"query": d["query"], "response": d["response"]}) for d in data]
    })

def main():
    logger.info("="*70)
    logger.info("🔧 STAGE 4: TOOL INTEGRATION (ENHANCED)")
    logger.info("="*70)
    logger.info("Purpose: Learn tool usage (Terminal, Browser, Vision)")
    logger.info("Duration: 3-4 days")
    logger.info("="*70)
    
    # Check base model
    base_model = "checkpoints/stage3_grpo/final"
    if not Path(base_model).exists():
        logger.error(f"❌ Base model not found: {base_model}")
        logger.error("   Run Stage 3 first: python 06_grpo_training.py")
        return
    
    # Load model
    logger.info("\n📦 Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load base model: {e}")
        return
    
    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    logger.info("✓ Model loaded")
    
    # Create tool dataset
    logger.info("\n📂 Creating tool trajectory dataset...")
    dataset = create_tool_dataset()
    logger.info(f"✓ Created {len(dataset)} tool trajectories")
    
    # Training
    logger.info("\n⚙️  Starting tool integration training...")
    training_args = TrainingArguments(
        output_dir="checkpoints/stage4_tool_integration",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        save_steps=50,
        logging_steps=5,
        bf16=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        packing=False,
    )
    
    try:
        trainer.train()
        logger.info("✅ Tool integration training complete!")
        
        # Save
        model.save_pretrained("checkpoints/stage4_tool_integration/final")
        tokenizer.save_pretrained("checkpoints/stage4_tool_integration/final")
        logger.info("✓ Model saved")
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()

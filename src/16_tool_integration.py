
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
import os
import gc

from transformers import TrainingArguments, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import Dataset

# Import our custom architecture
from multimodal.model import OmniMultimodalLM

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

def create_tool_dataset(tokenizer) -> Dataset:
    """Create dataset from tool trajectories formatted for Qwen2"""
    data = []
    
    # Simple Qwen2/ChatML format
    # user -> tool request
    # model -> thought -> tool call
    
    for i, traj in enumerate(TOOL_TRAJECTORIES):
        # Format as conversation
        messages = [
            {"role": "user", "content": traj["query"]},
            {"role": "assistant", "content": "\n".join(traj["trajectory"])}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        data.append({"text": text})
    
    return Dataset.from_dict({
        "text": [d["text"] for d in data]
    })

def main():
    logger.info("="*70)
    logger.info("🔧 STAGE 5: TOOL INTEGRATION (OMNI ARCHITECTURE)")
    logger.info("="*70)
    logger.info("Purpose: Fine-tune OmniMultimodalLM on tool usage trajectories")
    logger.info("Strategy: Use Safe Loading (FP16 Encoders + Int4 LLM)")
    logger.info("="*70)
    
    # Configuration
    base_model_path = "./checkpoints/manus_fine_tuning" # Stage 2 output
    # If Stage 2 hasn't finished, use the base model path and we'll init fresh
    if not os.path.exists(base_model_path) or len(os.listdir(base_model_path)) == 0:
        logger.warning(f"⚠️  Stage 2 Checkpoint not found at {base_model_path}. Using base Qwen2.5-7B-Instruct.")
        base_model_path = "Qwen/Qwen2.5-7B-Instruct" 

    # Load Tokenizer
    logger.info("\n📦 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model (Using Safe Logic from Stage 4)
    logger.info("\n📦 Loading OmniMultimodalLM...")
    try:
        # 1. Clean Memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # 2. Init Model
        model = OmniMultimodalLM(
            llm_name="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4",
            vision_name="/mnt/e/data/encoders/vision encoders/siglip2-so400m-patch16-512",
            audio_name="/mnt/e/data/encoders/audio encoders/whisper-large-v3-turbo",
            inject_vision=True,
            inject_audio=True
        )
        
        # 3. Enable Gradient Checkpointing (Critical for VRAM)
        model.llm.gradient_checkpointing_enable() 
        model.wrapper.vision_encoder.requires_grad_(False) # Freeze Encoders
        model.wrapper.audio_encoder.requires_grad_(False)
        
        # Unfreeze Projectors? Or LoRA?
        # For this script (demo), we'll rely on LoRA on the LLM if using Unsloth.
        # But we replaced Unsloth.
        # So we should probably target the specific query/key/value projections using PEFT.
        
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Apply LoRA to the LLM part of OmniMultimodalLM
        # Note: OmniMultimodalLM wraps the LLM in self.llm
        # BUT: OmniMultimodalLM.forward delegates. SFTTrainer expects a standard HF model.
        # This is tricky. SFTTrainer expects `model(input_ids, labels) -> loss`.
        # OmniMultimodalLM.forward signature: (input_ids, pixel_values, ...) -> logits/loss.
        # It IS compatible if we verify the signature.
        
        # Apply LoRA to internal LLM
        model.llm = get_peft_model(model.llm, lora_config)
        model.print_trainable_parameters()
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Create tool dataset
    logger.info("\n📂 Creating tool trajectory dataset...")
    dataset = create_tool_dataset(tokenizer)
    logger.info(f"✓ Created {len(dataset)} tool trajectories")
    
    # Training Arguments
    logger.info("\n⚙️  Starting tool integration training...")
    training_args = TrainingArguments(
        output_dir="checkpoints/stage5_tool_integration",
        num_train_epochs=1,
        per_device_train_batch_size=1, # Safe batch size
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        save_steps=50,
        logging_steps=1,
        fp16=True, # Use FP16 matching our Encoders
        remove_unused_columns=False, # Critical for custom models!
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    
    try:
        trainer.train()
        logger.info("✅ Tool integration training complete!")
        
        # Save
        trainer.save_model("checkpoints/stage5_tool_integration/final")
        tokenizer.save_pretrained("checkpoints/stage5_tool_integration/final")
        logger.info("✓ Model saved")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

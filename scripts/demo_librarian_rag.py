#!/usr/bin/env python3
"""
Nexus Librarian Demo: "Super Llama" Mode
----------------------------------------
This script demonstrates how to key "Immediate Utility" from Nexus without training.
It connects a Standard Model (Llama-3, Qwen, etc.) to the Nexus Librarian (Knowledge Base).

Usage:
    python scripts/demo_librarian_rag.py --query "What is the secret?"

Requirements:
    - A populated Knowledge Base (run distill_knowledge.py first)
    - A standard HF model (default: Qwen/Qwen2.5-0.5B-Instruct for speed)
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.nexus_final.knowledge import KnowledgeTower

def main():
    parser = argparse.ArgumentParser(description="Nexus Librarian RAG Demo")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Standard HF Model ID")
    parser.add_argument("--top_k", type=int, default=3, help="Number of context shards to retrieve")
    args = parser.parse_args()

    import time
    
    t0 = time.time()
    print(f"[{time.time()-t0:.2f}s] 🚀 Initializing Standard Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Fix: use dtype instead of torch_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.float16)
    print(f"[{time.time()-t0:.2f}s] -> Model Loaded.")

    print(f"[{time.time()-t0:.2f}s] 📚 Connecting to Nexus Librarian...")
    # Initialize Tower (loads Index from memory/ directory)
    tower = KnowledgeTower(student_dim=128, device="cpu") 
    
    # Check if index is built (mocking for demo if empty)
    if not tower.index or tower.index.ntotal == 0:
        print("⚠️  Warning: Librarian Index is empty! (Run distill_knowledge.py to populate)")
        print("   -> Temporarily injecting dummy knowledge for demo.")
        tower.build_index([
            "Nexus Architecture uses Sequential Layer Ingestion (SLI) for massive models.",
            "The secret code is 77-Alpha-Baker.",
            "Llama-3 can become a Super Llama by accessing external SSD memory."
        ])
    print(f"[{time.time()-t0:.2f}s] -> Librarian Ready.")

    # 1. RETRIEVE (The Magic Step)
    t_search = time.time()
    print(f"\n🔍 Searching 10TB Library for: '{args.query}'")
    context_shards = tower.retrieve_text_context(args.query, top_k=args.top_k)
    print(f"✅ Found {len(context_shards)} relevant shards in {(time.time()-t_search)*1000:.1f}ms.")
    print(f"✅ Found {len(context_shards)} relevant shards.")

    # 2. INJECT
    context_block = "\n".join([f"- {shard}" for shard in context_shards])
    
    system_prompt = f"""You are an AI assistant enhanced with the Nexus Librarian.
Use the following retrieved context to answer the user's question.
If the answer is in the context, cite it.

RETRIEVED KNOWLEDGE:
{context_block}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.query}
    ]
    
    # 3. GENERATE
    print("\n🤖 Generating Answer...")
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "="*40)
    print("FINAL ANSWER:")
    print("="*40)
    print(response)
    print("="*40)

if __name__ == "__main__":
    main()

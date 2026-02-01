#!/usr/bin/env python3
"""
A/B Test Comparison: Standard RAG vs. Nexus Native RAG
------------------------------------------------------
This script demonstrates the two ways to consume knolwedge from the Nexus Librarian.

Scenario:
We have a query "What is the secret of the universe?"
We have a Knowledge Base containing shards about "42" and "Donuts".

Side A: Standard Model (Llama-3/GPT-4)
-> Retrieves Raw Text
-> Injects into Prompt
-> "Read this text and answer..."

Side B: Nexus Student (Native)
-> Retrieves Hidden State Vectors (DeepSeek Thinking)
-> Projects directly into Brain
-> "Dream this thought and answer..."
"""

import sys
import torch
import os
from src.nexus_final.knowledge import KnowledgeTower

def setup_dummy_knowledge_base(tower):
    """Populates the tower with some dummy facts."""
    print("📚 Populating Knowledge Base...")
    facts = [
        "The secret to the universe is 42.",
        "Donuts are the primary fuel source for police officers.",
        "The Nexus architecture uses latent injection for 10x speed.",
        "Python is named after Monty Python, not the snake."
    ]
    tower.build_index(facts)
    print(f"✅ Indexed {len(facts)} documents.")

def run_standard_rag_side_a(tower, query):
    """
    SIDE A: Standard RAG (Text Injection)
    Compatible with: Llama-3, GPT-4, Mistral, etc.
    """
    print("\n" + "="*50)
    print("🔵 SIDE A: Standard Model (Text RAG)")
    print("="*50)
    
    # 1. Retrieve
    print(f"🔍 Querying Librarian: '{query}'")
    context_docs = tower.retrieve_text_context(query, top_k=2)
    
    # 2. Inject
    prompt = f"""
    SYSTEM: You are a helpful assistant. Use the context below to answer.
    
    CONTEXT:
    {context_docs}
    
    USER: {query}
    """
    
    print("\n--- Constructing Prompt ---")
    print(prompt.strip())
    print("---------------------------")
    print("✅ Result: The model reads the text above and generates an answer.")

def run_nexus_side_b(tower, query):
    """
    SIDE B: Nexus Native (Latent Injection)
    Compatible with: Nexus Student only.
    """
    print("\n" + "="*50)
    print("🟣 SIDE B: Nexus Student (Native RAG)")
    print("="*50)
    
    # 1. Retrieve & Project
    print(f"🧠 Querying Librarian (Latent Mode): '{query}'")
    
    # This retrieves TENSORS, not text
    with torch.no_grad():
        latents = tower(query, top_k=2) # Returns [1, 2, student_dim]
        
    print(f"\n--- Constructing Neural Inputs ---")
    print(f"Received Latent Tensor Shape: {latents.shape}")
    print(f"Content: (Abstract Neural Activations - Not Human Readable)")
    print(latents[0, 0, :10].numpy().tolist(), "...")
    print("----------------------------------")
    print("✅ Result: These tensors are injected directly into the Student's attention layers.")
    print("   Benefit: No token limits, faster processing, deeper understanding.")

def main():
    print("🚀 Initializing Nexus Librarian (KnowledgeTower)...")
    # Using small dimension for demo
    tower = KnowledgeTower(student_dim=64, embedding_dim=384, device="cpu")
    
    setup_dummy_knowledge_base(tower)
    
    query = "What is the secret?"
    
    run_standard_rag_side_a(tower, query)
    run_nexus_side_b(tower, query)
    
    print("\n🏁 A/B Test Complete.")

if __name__ == "__main__":
    main()

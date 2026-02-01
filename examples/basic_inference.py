#!/usr/bin/env python3
"""
Basic Inference Example - Nexus

Demonstrates how to perform basic text generation using a Nexus-compatible model.

Usage:
    python examples/basic_inference.py --model "microsoft/Phi-3-mini-4k-instruct"
    python examples/basic_inference.py --model "/path/to/local/model"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_inference(model_name: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7):
    """
    Run basic text generation inference.
    
    Args:
        model_name: HuggingFace model name or local path
        prompt: Input prompt for generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model loaded on: {model.device}")
    print("-" * 50)
    
    # Format prompt with chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = f"User: {prompt}\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(model.device)
    
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated text
    input_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    response = generated_text[input_length:].strip()
    
    print(f"Response: {response}")
    print("-" * 50)
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Basic Inference Example")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Input prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("NEXUS - Basic Inference Example")
    print("=" * 50)
    
    run_inference(
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()

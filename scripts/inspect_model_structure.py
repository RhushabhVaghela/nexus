import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

model_path = "/mnt/e/data/models/google_translategemma-4b-it"

print(f"Inspecting structure of {model_path}...")

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu", # Use CPU just for attribute inspection to save VRAM
    quantization_config=quantization_config,
    trust_remote_code=True
)

print(f"Model Class: {type(model)}")
print(f"Model attributes: {dir(model)}")

if hasattr(model, "model"):
    print(f"model.model Class: {type(model.model)}")
    print(f"model.model attributes: {dir(model.model)}")
    if hasattr(model.model, "layers"):
        print("FOUND model.model.layers")
    elif hasattr(model.model, "decoder"):
        print("FOUND model.model.decoder")
        if hasattr(model.model.decoder, "layers"):
             print("FOUND model.model.decoder.layers")

# Print the attribute tree for recursion
def print_tree(obj, prefix="", depth=0):
    if depth > 2: return
    for attr in dir(obj):
        if attr.startswith("_"): continue
        val = getattr(obj, attr)
        if "ModuleList" in str(type(val)):
            print(f"{prefix}{attr} -> {type(val)} (LENGTH: {len(val)})")
        elif "Module" in str(type(val)):
             print(f"{prefix}{attr} -> {type(val)}")
             # print_tree(val, prefix + "  ", depth + 1)

print("\nModule Tree:")
print_tree(model)
if hasattr(model, "model"):
    print("\nUnder .model:")
    print_tree(model.model)

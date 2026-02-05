
import json
import random
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_FILE = "nexus_agent_mix.jsonl"

def generate_synthetic_podcast(n=500):
    """
    Synthesizes 'Perfect Podcast' scripts to teach the JSON format.
    """
    print(f"🎙️  Synthesizing {n} Podcast examples...")
    data = []
    
    topics = ["AI Agents", "Python Coding", "Space Travel", "Cooking", "History of Rome"]
    
    for _ in range(n):
        topic = random.choice(topics)
        
        # We manually construct the prompt-response pair to force the JSON constraint
        prompt = f"Generate a 2-speaker podcast script regarding {topic}.\n"
        prompt += "Hosts: Host A (Analytical), Host B (Funny).\n"
        prompt += "Output strictly as JSON."
        
        # Perfect JSON response
        script = {
            "turns": [
                {"speaker": "Host A", "text": f"Welcome back. Today we are discussing {topic}."},
                {"speaker": "Host B", "text": "Oh boy, I have so many questions!"},
                {"speaker": "Host A", "text": "Let's dive in. It started with..."},
                {"speaker": "Host B", "text": "Wait, really? That's wild!"}
            ]
        }
        
        data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(script, indent=2)}
            ],
            "source": "synthetic_podcast"
        })
    return data

def generate_streaming_context(n=200):
    """
    Synthesizes Tri-Streaming Context responses.
    """
    print(f"🌊 Synthesizing {n} Streaming examples...")
    data = []
    for _ in range(n):
        prompt = "System: Analyze the current context.\n"
        prompt += "[VISION: User is waving hand]\n"
        prompt += "[AUDIO: 'Hello computer']\n"
        
        response = "The user is waving and greeting me. I should respond."
        
        data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "<think>User waved and spoke.</think> " + response}
            ],
            "source": "synthetic_streaming"
        })
    return data

def prepare_data():
    all_data = []
    
    # 1. OpenO1 (Reasoning) - 2000 samples
    print("🧠 Loading OpenO1-SFT (Subset: 2000)...")
    try:
        ds = load_dataset("O1-OPEN/OpenO1-SFT", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= 2000: break
            # Keep original instruction format
            all_data.append({
                "messages": [
                    {"role": "user", "content": row['instruction']},
                    {"role": "assistant", "content": row['output']} # Contains <think>
                ],
                "source": "openo1"
            })
            count += 1
    except Exception as e:
        print(f"⚠️ Failed to load OpenO1: {e}")

    # 2. XLAM (Tools) - 1000 samples
    print("🔧 Loading XLAM Function Calling (Subset: 1000)...")
    try:
        ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= 1000: break
            all_data.append({
                "messages": [
                    {"role": "user", "content": row['query']},
                    {"role": "assistant", "content": row['answers']} # JSON output
                ],
                "source": "xlam_tools"
            })
            count += 1
    except Exception as e:
        print(f"⚠️ Failed to load XLAM: {e}")

    # 3. Synthetic
    all_data.extend(generate_synthetic_podcast(500))
    all_data.extend(generate_streaming_context(200))
    
    # Shuffle
    random.shuffle(all_data)
    
    # Save
    print(f"💾 Saving {len(all_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
            
    print("✅ Data Preparation Complete.")

if __name__ == "__main__":
    prepare_data()

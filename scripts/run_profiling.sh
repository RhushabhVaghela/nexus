#!/bin/bash
# Nexus Profiling Runner
# Usage: ./run_profiling.sh [BATCH_SIZE] [MODEL_PATH]

BATCH_SIZE=${1:-4}
MODEL_PATH=${2:-"/mnt/e/data/models/AgentCPM-Explore"}

echo "=== Nexus NIWT Profiler ==="
echo "Batch Size: $BATCH_SIZE"
echo "Model: $MODEL_PATH"

# Ensure PYTHONPATH includes src
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the python script wrapper (we need a small wrapper to invoke the class)
python3 -c "
import os
from nexus_core.profiling.niwt import NIWTCore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd

MODEL_PATH = '$MODEL_PATH'
DATA_PATH = '/mnt/e/data/benchmarks/math/openai_gsm8k/main/test-00000-of-00001.parquet'

print(f'Loading model: {MODEL_PATH}')
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
except Exception as e:
    print(f'Failed to load model: {e}')
    exit(1)

# Load Data
print(f'Loading data from {DATA_PATH}')
try:
    df = pd.read_parquet(DATA_PATH)
    test_cases = []
    # Take top 20 for profiling
    for _, row in df.head(20).iterrows():
        q = row['question']
        a = row['answer'].split('####')[-1].strip()
        # Format prompt
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [{'role': 'user', 'content': q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f'Question: {q}\nLet\'s think step by step.\nAnswer:'
        test_cases.append((prompt, a))
except Exception as e:
    print(f'Failed to load data: {e}')
    test_cases = [('What is 2+2?', '4'), ('Who is the president?', 'Biden')] # Fallback

# Run NIWT
config = {'batch_size': $BATCH_SIZE}
niwt = NIWTCore(model, tokenizer, config)
niwt.run_stage_1_perturbation(test_cases)
niwt.run_stage_2_activation_analysis([t[0] for t in test_cases])
"

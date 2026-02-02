#!/usr/bin/env python3
"""
FILE 2: 02_generate_trajectories.py
Generate 5,000 diverse trajectories using GPT-4o
Cost: ~$25-30 | Duration: 2-3 hours
Output: cold_start_trajectories.jsonl
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any
import openai
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_trajectories.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Domain-specific prompts
DOMAIN_PROMPTS = {
    "math": [
        "A store sells apples for $2 each. If you buy 5 apples and pay with a $20 bill, how much change do you get?",
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
        "Solve for x: 2(x + 3) = 14",
    ],
    "code": [
        "Write a function to find the median of an unsorted list",
        "Implement a binary search algorithm",
        "Create a class to represent a linked list",
    ],
    "fullstack": [
        "Design a user authentication system with OAuth2",
        "Build a RESTful API for a todo application",
        "Create a React component for file upload with progress bar",
    ],
    "analysis": [
        "Analyze the trend in this data: [1,2,2,3,3,3,4,4,4,4]",
        "What patterns do you see in this dataset?",
        "Interpret the following statistics...",
    ]
}

TRAJECTORY_TEMPLATE = """
You are generating a detailed reasoning trajectory for an AI model to learn from.
The trajectory should show:
1. Initial thinking about the problem
2. Attempted solution or tool use
3. Possible errors and recovery
4. Final answer with explanation

Format the response as JSON with this structure:
{{
  "user_query": "<user question>",
  "difficulty": "<easy|medium|hard>",
  "domain": "<domain>",
  "trajectory": [
    {{"step": 1, "type": "think", "content": "..."}},
    {{"step": 2, "type": "action", "tool": "python_exec", "input": "...", "description": "..."}},
    {{"step": 3, "type": "observation", "result": "..."}},
    {{"step": 4, "type": "error", "error_type": "ValueError", "error_message": "..."}},
    {{"step": 5, "type": "recovery", "content": "...", "action": "..."}},
    {{"step": 6, "type": "action", "tool": "python_exec", "input": "...", "description": "..."}},
    {{"step": 7, "type": "observation", "result": "..."}},
    {{"step": 8, "type": "final_answer", "content": "..."}}
  ]
}}

Important:
- Include 5-15 steps per trajectory
- Always include at least one error and recovery
- Domain determines trajectory type
- Ensure final_answer is clear and correct
"""

def generate_trajectory(domain: str, question: str, client) -> Dict[str, Any]:
    """Generate single trajectory using GPT-4o"""
    prompt = f"""{TRAJECTORY_TEMPLATE}

Domain: {domain}
User Question: {question}
Difficulty: medium

Generate realistic, detailed trajectory."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        result["domain"] = domain
        return result
    
    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return None

def main():
    logger.info("="*70)
    logger.info("🎲 GENERATING TRAJECTORIES WITH GPT-4o")
    logger.info("="*70)
    logger.info("Samples: 5,000")
    logger.info("Duration: 2-3 hours")
    logger.info("Cost: ~$25-30")
    logger.info("="*70)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not set")
        logger.error("   Run: export OPENAI_API_KEY='sk-...'")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    # Generate trajectories
    trajectories = []
    failed = 0
    
    domains = list(DOMAIN_PROMPTS.keys())
    samples_per_domain = 5000 // len(domains)
    
    logger.info(f"\n🚀 Generating {5000} trajectories...")
    logger.info(f"   Samples per domain: {samples_per_domain}")
    
    pbar = tqdm.tqdm(total=5000, desc="Generating")
    
    for domain in domains:
        for _ in range(samples_per_domain):
            question = random.choice(DOMAIN_PROMPTS[domain])
            trajectory = generate_trajectory(domain, question, client)
            
            if trajectory:
                trajectories.append(trajectory)
            else:
                failed += 1
            
            pbar.update(1)
    
    pbar.close()
    
    # Save trajectories
    output_path = Path("cold_start_trajectories.jsonl")
    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAJECTORY GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Generated: {len(trajectories):,}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {len(trajectories)/(len(trajectories)+failed)*100:.1f}%")
    logger.info(f"Output: {output_path}")
    logger.info("="*70)
    logger.info(f"\nNext: Run Validation")
    logger.info(f"  python 03_validate_trajectories.py")

if __name__ == "__main__":
    import os
    main()

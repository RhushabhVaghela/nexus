import json
import random
import os
import argparse
from pathlib import Path

# Define output directory
OUTPUT_DIR = Path("/mnt/e/data/datasets/remotion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"

# Categories
CATEGORIES = [
    "math", "graph", "flow", "annotator", "3d", "audio", 
    "timeline", "chart", "map", "story", "lifestyle"
]

# Load Knowledge Base
def load_kb():
    kb = {}
    for domain in ["history", "business", "science", "story", "lifestyle"]:
        path = KB_DIR / f"{domain}.json"
        if path.exists():
            with open(path) as f:
                kb[domain] = json.load(f)
    return kb

KB = load_kb()

def generate_sample(id, weights=None):
    if weights:
        cat = random.choices(CATEGORIES, weights=weights)[0]
    else:
        cat = random.choice(CATEGORIES)
    
    instruction = ""
    tsx = ""

    if cat == "timeline" and "history" in KB:
        data = random.choice(KB["history"]["timelines"])
        instruction = f"Create a timeline of {data['topic']}."
        tsx = f"<NexusTimeline events={{{json.dumps(data['events'])}}} />"

    elif cat == "chart" and "business" in KB:
        chart_type = random.choice(["charts", "funnels"])
        if chart_type == "charts":
            data = random.choice(KB["business"]["charts"])
            instruction = f"Visualize {data['topic']} using a {data['type']} chart."
            tsx = f"<NexusChart data={{{json.dumps(data['data'])}}} type='{data['type']}' title='{data['topic']}' />"
        else:
            data = random.choice(KB["business"]["funnels"])
            instruction = f"Show the stages of {data['topic']}."
            nodes = [{"id": f"n{i}", "label": s, "x": 300 + i*300, "y": 500} for i, s in enumerate(data["steps"])]
            edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(len(nodes)-1)]
            tsx = f"<NexusFlow nodes={{{json.dumps(nodes)}}} edges={{{json.dumps(edges)}}} color='#f39c12' />"

    elif cat == "map":
        cities = [
            {"lat": 40.7128, "lng": -74.0060, "label": "New York"},
            {"lat": 51.5074, "lng": -0.1278, "label": "London"},
            {"lat": 35.6762, "lng": 139.6503, "label": "Tokyo"}
        ]
        subset = random.sample(cities, k=random.randint(2, 3))
        instruction = f"Show the global locations of {', '.join([c['label'] for c in subset])}."
        tsx = f"<NexusMap markers={{{json.dumps(subset)}}} />"

    elif cat == "story" and "story" in KB:
        scenario = random.choice(KB["story"]["scenarios"])
        instruction = f"Generate a {scenario['genre']} scene in a {scenario['setting']}."
        bg = "space.jpg" if "Mars" in scenario['setting'] else "city.jpg"
        char = scenario['characters'][0]
        line = scenario['dialogue'][0]
        tsx = f"<NexusScene background='{bg}' filter='contrast'>\\n  <NexusCharacter src='{char}.png' name='{char}' />\\n  <NexusDialogue text='{line}' speaker='{char}' />\\n</NexusScene>"

    elif cat == "lifestyle" and "lifestyle" in KB:
        topic_type = random.choice(["recipes", "routines"])
        data = random.choice(KB["lifestyle"][topic_type])
        instruction = f"Show the steps for {data['topic']}."
        # Convert simplified steps to ListItem format
        items = [{"title": s["title"], "detail": s.get("detail", "")} for s in data["steps"]]
        tsx = f"<NexusList items={{{json.dumps(items)}}} type='number' />"

    # Fallback to existing logic
    else:
        if cat == "math":
            instruction = "Explain Euler's Identity."
            tsx = "<NexusMath latex='e^{i\\\\pi} + 1 = 0' />"
        elif cat == "graph":
            instruction = "Plot a sine wave."
            tsx = "<NexusGraph fn={(x) => Math.sin(x)} />"
        elif cat == "3d":
            instruction = "Visualize a vector field."
            tsx = "<Nexus3D vectors={[{'x':1, 'y':1, 'z':1}]} />"
        else:
            instruction = "Create a generic flow."
            tsx = "<NexusFlow nodes={[]} edges={[]} />"

    return {
        "id": f"remotion_{id}",
        "instruction": instruction,
        "input": "",
        "output": f"import React from 'react';\nimport {{ NexusMath, NexusGraph, NexusFlow, NexusAnnotator, NexusAudio, Nexus3D, NexusTimeline, NexusChart, NexusMap, NexusScene, NexusCharacter, NexusDialogue, NexusList }} from './NexusLib';\n\nexport const Scene = () => (\\n  <div style={{{{ flex: 1, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}}}>\\n    {tsx}\\n  </div>\\n);",
        "category": "remotion-explainer"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--category-weights", nargs="+", help="Weights for specific categories (e.g. math=50 story=20). Unassigned categories split the remainder.")
    args = parser.parse_args()
    
    total_samples = args.sample_size
    weights = [1.0] * len(CATEGORIES) # Default equal weights if nothing specified
    
    if args.category_weights:
        # Parse inputs
        assigned_weights = {}
        total_assigned = 0
        
        for item in args.category_weights:
            try:
                cat, weight = item.split('=')
                weight = float(weight)
                if cat not in CATEGORIES:
                    print(f"Warning: Unknown category '{cat}', skipping.")
                    continue
                assigned_weights[cat] = weight
                total_assigned += weight
            except ValueError:
                print(f"Error: Invalid format '{item}'. Use category=weight")
                return

        if total_assigned > 100:
            print(f"Error: Total assigned percentage ({total_assigned}%) exceeds 100%")
            return

        # Distribute remaining
        remaining_percentage = 100 - total_assigned
        unassigned_categories = [c for c in CATEGORIES if c not in assigned_weights]
        
        if unassigned_categories:
            weight_per_unassigned = remaining_percentage / len(unassigned_categories)
        else:
            weight_per_unassigned = 0 # Should effectively be 0 if all assigned, or normalized later
            
        # Construct final weights list matching CATEGORIES order
        final_weights = []
        for cat in CATEGORIES:
            if cat in assigned_weights:
                final_weights.append(assigned_weights[cat])
            else:
                final_weights.append(weight_per_unassigned)
        
        weights = final_weights
        print(f"Using calculated weights: {dict(zip(CATEGORIES, weights))}")
    
    file_path = OUTPUT_DIR / "remotion_explainer_dataset.jsonl"
    print(f"Generating {total_samples} samples to {file_path}...")
    
    with open(file_path, "w") as f:
        for i in range(total_samples):
            sample = generate_sample(i, weights)
            f.write(json.dumps(sample) + "\\n")
            if i % 10000 == 0:
                print(f"Generated {i} samples...")

if __name__ == "__main__":
    main()
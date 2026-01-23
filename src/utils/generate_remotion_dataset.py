import json
import random
import os
from pathlib import Path

# Define output directory
OUTPUT_DIR = Path("/mnt/e/data/datasets/remotion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Primitives for Flowcharts
FLOW_STEPS = [
    ("User Request", 300, 200),
    ("Auth Check", 600, 200),
    ("Process Data", 900, 200),
    ("Generate Result", 1200, 200),
    ("Send Notification", 1500, 200)
]

# Primitives for Annotations
IMAGES = [
    "https://raw.githubusercontent.com/remotion-dev/remotion/main/packages/docs/static/img/logo-colored.png",
    "https://upload.wikimedia.org/wikipedia/commons/b/b4/Periodic_table.svg",
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Human_mitochondrion_diagram.svg"
]

LABELS = ["Mitochondria", "Nucleus", "Ribosome", "Cytoplasm", "Cell Wall"]

# Existing Templates
MATH_TEMPLATES = [
    ("Explain the {concept} formula: {formula}", "<NexusMath latex='{formula}' fontSize={size} color='{color}' />"),
    ("Visualize the relationship between {var1} and {var2} in {topic}", "<NexusMath latex='{formula}' />"),
]
CONCEPTS = ["Euler's Identity", "Quadratic Equation", "Schrodinger Equation", "General Relativity", "Pythagorean Theorem"]
FORMULAS = ["e^{i\\\\pi} + 1 = 0", "ax^2 + bx + c = 0", "i\\\\hbar\\\\frac{\\\\partial}{\\\\partial t}\\\\Psi = \\\\hat{H}\\\\Psi", "G_{\\\\mu\\\\nu} + \\\\Lambda g_{\\\\mu\\\\nu} = \\\\kappa T_{\\\\mu\\\\nu}", "a^2 + b^2 = c^2"]
COLORS = ["#00f2ff", "#ff0055", "#ffffff", "#00ff88", "#ffaa00"]

FUNCTIONS = [
    ("sine", "sin(x)", "Math.sin(x)", "[-10, 10]"),
    ("exponential", "e^x", "Math.exp(x)", "[-2, 2]"),
    ("sigmoid", "1 / (1 + e^-x)", "1 / (1 + Math.exp(-x))", "[-5, 5]"),
    ("quadratic", "x^2", "x * x", "[-3, 3]"),
]

def generate_sample(id):
    cat = random.choice(["math", "graph", "flow", "annotator", "3d", "audio"])
    
    if cat == "math":
        # ... existing math logic ...
        concept = random.choice(CONCEPTS)
        formula = random.choice(FORMULAS)
        template = random.choice(MATH_TEMPLATES)
        color = random.choice(COLORS)
        size = random.randint(30, 80)
        instruction = template[0].format(concept=concept, formula=formula, topic=concept, var1="x", var2="y")
        tsx = template[1].format(formula=formula.replace("\\", "\\\\"), size=size, color=color)
        
    elif cat == "graph":
        fn = random.choice(FUNCTIONS)
        color = random.choice(COLORS)
        instruction = f"Plot the {fn[0]} function {fn[1]}"
        tsx = f"<NexusGraph fn={{(x) => {fn[2]}}} xRange={{{fn[3]}}} color='{color}' />"
        
    elif cat == "flow":
        steps = random.sample(FLOW_STEPS, k=random.randint(2, 5))
        nodes = [{"id": f"n{i}", "label": s[0], "x": s[1], "y": s[2]} for i, s in enumerate(steps)]
        edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(len(nodes)-1)]
        instruction = f"Create a flowchart explaining {random.choice(['the data pipeline', 'authentication', 'user signup'])}"
        tsx = f"<NexusFlow nodes={{{json.dumps(nodes)}}} edges={{{json.dumps(edges)}}} />"
        
    elif cat == "annotator":
        img = random.choice(IMAGES)
        ann_list = []
        for i in range(random.randint(1, 3)):
            ann_list.append({"x": random.randint(10, 80), "y": random.randint(10, 80), "text": random.choice(LABELS)})
        instruction = f"Identify and label components in this diagram."
        tsx = f"<NexusAnnotator src='{img}' annotations={{{json.dumps(ann_list)}}} />"

    elif cat == "3d":
        vectors = []
        for i in range(random.randint(1, 4)):
            vectors.append({"x": random.uniform(-3, 3), "y": random.uniform(-3, 3), "z": random.uniform(-3, 3), "color": random.choice(COLORS)})
        instruction = f"Visualize a 3D vector space with {len(vectors)} basis vectors."
        tsx = f"<Nexus3D vectors={{{json.dumps(vectors)}}} />"

    else: # Audio
        instruction = "Generate an intro video with background music."
        tsx = "<NexusAudio src='music.mp3' volume={0.5} />\n    <NexusMath latex='\\\\text{Welcome to Nexus}' />"

    return {
        "id": f"remotion_{id}",
        "instruction": instruction,
        "input": "",
        "output": f"import React from 'react';\nimport {{ NexusMath, NexusGraph, NexusFlow, NexusAnnotator, NexusAudio, Nexus3D }} from './NexusLib';\n\nexport const Scene = () => (\n  <div style={{{{ flex: 1, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}}}>\n    {tsx}\n  </div>\n);",
        "category": "remotion-explainer"
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    args = parser.parse_args()
    
    total_samples = args.sample_size
    file_path = OUTPUT_DIR / "remotion_explainer_dataset.jsonl"
    print(f"Generating {total_samples} samples to {file_path}...")
    with open(file_path, "w") as f:
        for i in range(total_samples):
            sample = generate_sample(i)
            f.write(json.dumps(sample) + "\n")
            if i % 10000 == 0:
                print(f"Generated {i} samples...")

if __name__ == "__main__":
    main()
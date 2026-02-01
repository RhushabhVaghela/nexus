import json
import argparse
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.table import Table

def validate_diversity(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Dataset not found: {path}")
        return

    console = Console()
    stats = Counter()
    total = 0
    
    console.print(f"[bold blue]Scanning dataset:[/bold blue] {path}")
    
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract category/type from the instruction or output clues
                # Since the "category" field is top-level in our generator:
                if "category" in data:
                    # Actually our generator puts specific type in metadata or we infer from output
                    # Let's check the output for components to determine the specific sub-type
                    output = data.get("output", "")
                    if "NexusTimeline" in output: stats["History (Timeline)"] += 1
                    elif "NexusChart" in output: stats["Business (Chart)"] += 1
                    elif "NexusMap" in output: stats["Geography (Map)"] += 1
                    elif "NexusScene" in output: stats["Story (Scene)"] += 1
                    elif "NexusMath" in output: stats["Math"] += 1
                    elif "NexusGraph" in output: stats["Graph"] += 1
                    elif "NexusFlow" in output: stats["Logic/Flow"] += 1
                    elif "NexusAnnotator" in output: stats["Annotation"] += 1
                    elif "Nexus3D" in output: stats["3D"] += 1
                    elif "NexusAudio" in output: stats["Audio/Narrative"] += 1
                    else: stats["Unknown"] += 1
                total += 1
            except Exception:
                pass

    # Create Table
    table = Table(title=f"Dataset Diversity Report (N={total})")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    for cat, count in stats.most_common():
        pct = (count / total) * 100
        table.add_row(cat, str(count), f"{pct:.1f}%")

    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/mnt/e/data/datasets/remotion/remotion_explainer_dataset.jsonl")
    args = parser.parse_args()
    validate_diversity(args.file)
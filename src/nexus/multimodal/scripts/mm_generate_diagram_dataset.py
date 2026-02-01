#!/usr/bin/env python3
"""
mm_generate_diagram_dataset.py

Generate multimodal dataset for architecture diagrams and flowcharts.

- Input: directory of diagram images (PNG/JPG/SVG rendered)
- Output: JSONL with messages + modalities.image for training

Categories:
- System architecture diagrams
- Database schemas
- Flowcharts
- Class diagrams
- Sequence diagrams
- Infrastructure diagrams
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_generate_diagram.log")

CONFIG = {
    "input_image_dir": "/mnt/e/data/mm_raw/diagrams",
    "output_dir": "/mnt/e/data/multimodal-fullstack-dataset/architecture_diagram",
    "samples_per_file": 50_000,
    "seed": 42,
}

# Diagram type templates
DIAGRAM_TYPES = [
    "system_architecture",
    "database_schema",
    "flowchart",
    "class_diagram",
    "sequence_diagram",
    "infrastructure",
    "network_topology",
    "data_flow",
]

USER_TEMPLATES = {
    "system_architecture": [
        "Analyze this system architecture diagram and explain the components.",
        "What are the main services shown in this architecture diagram?",
        "Describe the data flow between components in this system diagram.",
        "Identify potential bottlenecks in this architecture design.",
    ],
    "database_schema": [
        "Explain the relationships shown in this database schema.",
        "What are the primary and foreign keys in this ER diagram?",
        "Suggest improvements for this database design.",
        "Identify normalization issues in this schema diagram.",
    ],
    "flowchart": [
        "Walk through the logic in this flowchart step by step.",
        "What are the decision points in this process flow?",
        "Identify any edge cases not handled by this flowchart.",
        "Convert this flowchart to pseudocode.",
    ],
    "class_diagram": [
        "Describe the class hierarchy shown in this UML diagram.",
        "What design patterns are evident in this class diagram?",
        "Explain the relationships (inheritance, composition) shown.",
        "Suggest refactoring opportunities based on this diagram.",
    ],
    "sequence_diagram": [
        "Trace the message flow in this sequence diagram.",
        "What is the order of operations between these components?",
        "Identify synchronous vs asynchronous calls in this diagram.",
        "What happens if the third call fails?",
    ],
    "infrastructure": [
        "Describe the cloud infrastructure shown in this diagram.",
        "What AWS/GCP/Azure services are used here?",
        "Identify single points of failure in this infrastructure.",
        "How would you scale this infrastructure design?",
    ],
    "network_topology": [
        "Explain the network topology shown in this diagram.",
        "What security layers are implemented here?",
        "How does traffic flow through this network?",
        "Identify potential security vulnerabilities.",
    ],
    "data_flow": [
        "Trace the data transformation pipeline in this diagram.",
        "What are the ETL stages shown here?",
        "Where might data quality issues occur?",
        "How would you add monitoring to this pipeline?",
    ],
}

ANSWER_TEMPLATES = {
    "system_architecture": (
        "This architecture diagram shows a distributed system with the following components:\n\n"
        "Key observations:\n"
        "- Identify each service/component and its role\n"
        "- Note the communication patterns (REST, gRPC, events)\n"
        "- Recognize data stores and caching layers\n"
        "- Assess scalability and fault tolerance aspects"
    ),
    "database_schema": (
        "This database schema represents the following data model:\n\n"
        "Analysis:\n"
        "- Identify tables/entities and their relationships\n"
        "- Note primary keys, foreign keys, and indices\n"
        "- Assess normalization level (1NF, 2NF, 3NF, BCNF)\n"
        "- Consider query patterns and optimization needs"
    ),
    "flowchart": (
        "This flowchart describes the following process:\n\n"
        "Step-by-step breakdown:\n"
        "- Trace each path from start to end\n"
        "- Identify decision points and their conditions\n"
        "- Note loops and their termination criteria\n"
        "- Consider error handling and edge cases"
    ),
    "class_diagram": (
        "This class diagram shows an object-oriented design:\n\n"
        "Design analysis:\n"
        "- Identify classes, interfaces, and abstract types\n"
        "- Note inheritance hierarchies and compositions\n"
        "- Recognize design patterns (Factory, Strategy, etc.)\n"
        "- Assess SOLID principle adherence"
    ),
    "sequence_diagram": (
        "This sequence diagram illustrates the following interaction:\n\n"
        "Message flow analysis:\n"
        "- Identify actors/participants in order\n"
        "- Trace synchronous and asynchronous messages\n"
        "- Note return values and error responses\n"
        "- Consider timing and concurrency issues"
    ),
    "infrastructure": (
        "This infrastructure diagram shows the following cloud setup:\n\n"
        "Infrastructure analysis:\n"
        "- Identify compute, storage, and network resources\n"
        "- Note load balancing and auto-scaling configurations\n"
        "- Assess security boundaries and access controls\n"
        "- Consider cost optimization opportunities"
    ),
    "network_topology": (
        "This network topology diagram shows:\n\n"
        "Network analysis:\n"
        "- Identify zones (public, private, DMZ)\n"
        "- Note firewalls, load balancers, and proxies\n"
        "- Trace traffic flow and routing rules\n"
        "- Assess security posture and redundancy"
    ),
    "data_flow": (
        "This data flow diagram represents:\n\n"
        "Pipeline analysis:\n"
        "- Identify data sources and destinations\n"
        "- Note transformation and enrichment stages\n"
        "- Assess data quality checkpoints\n"
        "- Consider monitoring and alerting needs"
    ),
}


def list_images(image_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    files: List[Path] = []
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def build_sample(image_path: Path, idx: int) -> Dict:
    """Build one multimodal sample for a diagram."""
    # Pick a diagram type based on filename hints or random
    diagram_type = random.choice(DIAGRAM_TYPES)
    
    # Check filename for type hints
    filename_lower = image_path.stem.lower()
    for dtype in DIAGRAM_TYPES:
        if dtype.replace("_", "") in filename_lower or dtype in filename_lower:
            diagram_type = dtype
            break
    
    user_prompt = random.choice(USER_TEMPLATES[diagram_type])
    answer_hint = ANSWER_TEMPLATES[diagram_type]

    sample = {
        "id": f"mm_diagram_{idx:08d}",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer_hint},
        ],
        "domain": "multimodal_fullstack",
        "category": f"diagram_{diagram_type}",
        "modalities": {
            "image": [
                {
                    "path": str(image_path),
                    "type": "diagram",
                    "subtype": diagram_type,
                    "description": f"{diagram_type.replace('_', ' ').title()} diagram",
                }
            ],
            "audio": [],
            "video": [],
        },
    }
    return sample


def main():
    random.seed(CONFIG["seed"])

    image_dir = Path(CONFIG["input_image_dir"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_header(
        logger,
        "MULTIMODAL DIAGRAM DATASET GENERATOR",
        {
            "Input images": str(image_dir),
            "Output": str(output_dir),
        },
    )

    images = list_images(image_dir)
    logger.info(f"Found {len(images)} diagram images")

    samples_per_file = CONFIG["samples_per_file"]
    batch: List[Dict] = []
    batch_idx = 0
    total = 0

    for idx, img_path in enumerate(images):
        sample = build_sample(img_path, idx)
        batch.append(sample)
        total += 1

        if len(batch) >= samples_per_file:
            out_path = output_dir / f"part_{batch_idx:04d}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for s in batch:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(batch)} samples to {out_path}")
            batch = []
            batch_idx += 1

    if batch:
        out_path = output_dir / f"part_{batch_idx:04d}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in batch:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(batch)} samples to {out_path}")

    log_completion(
        logger,
        "Multimodal Diagram Dataset",
        {"Total samples": total, "Output": str(output_dir)},
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
mm_download_code_diagram_datasets.py

Download and process real code-to-diagram datasets from HuggingFace.

Datasets supported:
- PlantUML-Github (code → diagram pairs from GitHub)
- Mermaid-Charts (code → mermaid diagram pairs)
- Draw.io Diagrams (architecture diagrams + descriptions)
- ER-Diagrams (database schema → ER diagram)
- Flowchart-Code (code logic → flowchart)

This generates bidirectional training data:
- Code → Diagram (generate diagram from code)
- Diagram → Code (explain/generate code from diagram)
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_download_code_diagram.log")


# ═══════════════════════════════════════════════════════════════
# REAL HUGGINGFACE DATASETS FOR CODE-DIAGRAM PAIRS
# ═══════════════════════════════════════════════════════════════

CODE_DIAGRAM_DATASETS = {
    "plantuml_code": {
        "hf_path": "sahil2801/CodeAlpaca-20k",  # Filter for diagram-related
        "split": "train",
        "description": "Code-to-PlantUML diagram generation",
        "filter_keywords": ["diagram", "uml", "class", "sequence", "architecture"],
        "size_gb": 1,
    },
    "mermaid_charts": {
        "hf_path": "bigcode/the-stack-dedup",
        "split": "train",
        "description": "Mermaid diagram code from repositories",
        "data_files": "data/markdown/**",
        "filter_pattern": "```mermaid",
        "size_gb": 5,
        "streaming": True,
    },
    "api_schemas": {
        "hf_path": "APIs-guru/openapi-directory",
        "split": "train",
        "description": "OpenAPI schemas (can generate API diagrams)",
        "size_gb": 2,
    },
    "database_schemas": {
        "hf_path": "gretelai/synthetic_text_to_sql",
        "split": "train",
        "description": "SQL schemas for ER diagram generation",
        "size_gb": 1,
    },
    "code_explanations": {
        "hf_path": "iamtarun/python_code_instructions_18k_alpaca",
        "split": "train",
        "description": "Code with explanations (for flowchart generation)",
        "size_gb": 0.5,
    },
}

# Diagram generation templates
DIAGRAM_TASKS = {
    "code_to_class_diagram": {
        "user_templates": [
            "Generate a UML class diagram for this code:\n```{lang}\n{code}\n```",
            "Create a class diagram showing the relationships in this code.",
            "Draw a UML diagram representing the class structure.",
        ],
        "diagram_type": "class_diagram",
    },
    "code_to_sequence": {
        "user_templates": [
            "Create a sequence diagram showing the flow of this code:\n```{lang}\n{code}\n```",
            "Generate a sequence diagram for the following API flow.",
            "Show the message passing between components as a sequence diagram.",
        ],
        "diagram_type": "sequence_diagram",
    },
    "code_to_flowchart": {
        "user_templates": [
            "Create a flowchart for this algorithm:\n```{lang}\n{code}\n```",
            "Generate a flowchart showing the logic flow.",
            "Convert this function to a flowchart diagram.",
        ],
        "diagram_type": "flowchart",
    },
    "schema_to_erd": {
        "user_templates": [
            "Generate an ER diagram for this database schema:\n```sql\n{code}\n```",
            "Create an entity-relationship diagram from these tables.",
            "Draw the database relationships as an ER diagram.",
        ],
        "diagram_type": "er_diagram",
    },
    "diagram_to_code": {
        "user_templates": [
            "Generate code that implements this class diagram.",
            "Write the Python classes shown in this UML diagram.",
            "Implement the architecture shown in this diagram.",
        ],
        "diagram_type": "reverse",
    },
}

# PlantUML templates for generating diagrams
PLANTUML_TEMPLATES = {
    "class_diagram": """@startuml
{content}
@enduml""",
    "sequence_diagram": """@startuml
{content}
@enduml""",
    "flowchart": """@startuml
start
{content}
stop
@enduml""",
    "er_diagram": """@startuml
!define Table(name,desc) entity name as "name\\n(desc)"
!define Column(name,type,desc) name : type <<desc>>

{content}
@enduml""",
}

# Mermaid templates
MERMAID_TEMPLATES = {
    "class_diagram": """```mermaid
classDiagram
{content}
```""",
    "sequence_diagram": """```mermaid
sequenceDiagram
{content}
```""",
    "flowchart": """```mermaid
flowchart TD
{content}
```""",
    "er_diagram": """```mermaid
erDiagram
{content}
```""",
}


@dataclass
class CodeDiagramSample:
    """Normalized code-diagram sample."""
    id: str
    code: str
    language: str
    diagram_type: str
    diagram_content: str
    diagram_format: str  # plantuml, mermaid, or image_path
    source_dataset: str
    metadata: Dict


def extract_classes_from_code(code: str, lang: str = "python") -> List[Dict]:
    """Extract class information for generating class diagrams."""
    classes = []
    lines = code.split("\n")
    current_class = None
    
    for line in lines:
        stripped = line.strip()
        if lang == "python":
            if stripped.startswith("class "):
                # Extract class name and inheritance
                class_def = stripped[6:].split("(")[0].split(":")[0].strip()
                inheritance = ""
                if "(" in stripped:
                    inheritance = stripped.split("(")[1].split(")")[0]
                current_class = {"name": class_def, "inherits": inheritance, "methods": [], "attributes": []}
                classes.append(current_class)
            elif current_class and stripped.startswith("def "):
                method_name = stripped[4:].split("(")[0]
                if not method_name.startswith("_") or method_name.startswith("__"):
                    current_class["methods"].append(method_name)
            elif current_class and "self." in stripped and "=" in stripped:
                attr = stripped.split("self.")[1].split("=")[0].strip()
                if attr not in current_class["attributes"]:
                    current_class["attributes"].append(attr)
    
    return classes


def generate_class_diagram(classes: List[Dict], format: str = "mermaid") -> str:
    """Generate class diagram from extracted classes."""
    if format == "mermaid":
        lines = []
        for cls in classes:
            lines.append(f"    class {cls['name']} {{")
            for attr in cls["attributes"][:5]:  # Limit attributes
                lines.append(f"        +{attr}")
            for method in cls["methods"][:5]:  # Limit methods
                lines.append(f"        +{method}()")
            lines.append("    }")
            if cls["inherits"]:
                for parent in cls["inherits"].split(","):
                    parent = parent.strip()
                    if parent:
                        lines.append(f"    {parent} <|-- {cls['name']}")
        return "\n".join(lines)
    else:  # PlantUML
        lines = []
        for cls in classes:
            lines.append(f"class {cls['name']} {{")
            for attr in cls["attributes"][:5]:
                lines.append(f"  +{attr}")
            for method in cls["methods"][:5]:
                lines.append(f"  +{method}()")
            lines.append("}")
            if cls["inherits"]:
                for parent in cls["inherits"].split(","):
                    parent = parent.strip()
                    if parent:
                        lines.append(f"{parent} <|-- {cls['name']}")
        return "\n".join(lines)


def extract_sql_tables(sql: str) -> List[Dict]:
    """Extract table information for ER diagrams."""
    tables = []
    current_table = None
    
    for line in sql.upper().split("\n"):
        line = line.strip()
        if "CREATE TABLE" in line:
            # Extract table name
            parts = line.split("CREATE TABLE")[-1].strip()
            table_name = parts.split("(")[0].split()[0].strip("`\"[]")
            current_table = {"name": table_name, "columns": [], "foreign_keys": []}
            tables.append(current_table)
        elif current_table:
            if "FOREIGN KEY" in line or "REFERENCES" in line:
                # Extract relationship
                if "REFERENCES" in line:
                    ref_table = line.split("REFERENCES")[-1].split("(")[0].strip().strip("`\"[]")
                    current_table["foreign_keys"].append(ref_table)
            elif any(dtype in line for dtype in ["INT", "VARCHAR", "TEXT", "DATE", "BOOL", "FLOAT"]):
                col_name = line.split()[0].strip("`\"[],")
                current_table["columns"].append(col_name)
    
    return tables


def generate_er_diagram(tables: List[Dict], format: str = "mermaid") -> str:
    """Generate ER diagram from SQL tables."""
    if format == "mermaid":
        lines = []
        for table in tables:
            lines.append(f"    {table['name']} {{")
            for col in table["columns"][:5]:
                lines.append(f"        string {col}")
            lines.append("    }")
            for fk in table["foreign_keys"]:
                lines.append(f"    {table['name']} ||--o{{ {fk} : references")
        return "\n".join(lines)
    return ""


def normalize_code_sample(sample: Dict, idx: int, source: str) -> Optional[CodeDiagramSample]:
    """Normalize a code sample and generate diagram."""
    try:
        # Extract code based on source
        if source == "code_explanations":
            code = sample.get("output", "") or sample.get("response", "")
            if "```" in code:
                # Extract code from markdown
                code = code.split("```")[1].split("```")[0]
                if code.startswith("python"):
                    code = code[6:]
            lang = "python"
        elif source == "database_schemas":
            code = sample.get("sql", "") or sample.get("query", "")
            lang = "sql"
        else:
            code = sample.get("content", "") or sample.get("code", "")
            lang = sample.get("language", "python")
        
        if not code or len(code) < 50:
            return None
        
        # Determine diagram type
        if lang == "sql" or "CREATE TABLE" in code.upper():
            diagram_type = "er_diagram"
            tables = extract_sql_tables(code)
            if not tables:
                return None
            diagram_content = generate_er_diagram(tables)
        elif "class " in code.lower():
            diagram_type = "class_diagram"
            classes = extract_classes_from_code(code, lang)
            if not classes:
                return None
            diagram_content = generate_class_diagram(classes)
        elif "def " in code or "function" in code.lower():
            diagram_type = "flowchart"
            diagram_content = f"    A[Start] --> B[{code.split('def ')[1].split('(')[0] if 'def ' in code else 'Function'}]\n    B --> C[Process]\n    C --> D[End]"
        else:
            return None
        
        return CodeDiagramSample(
            id=f"code_diagram_{source}_{idx:08d}",
            code=code[:2000],  # Limit code length
            language=lang,
            diagram_type=diagram_type,
            diagram_content=diagram_content,
            diagram_format="mermaid",
            source_dataset=source,
            metadata={},
        )
    except Exception:
        return None


def sample_to_messages(sample: CodeDiagramSample, direction: str = "code_to_diagram") -> Dict:
    """Convert CodeDiagramSample to messages format."""
    task_key = f"code_to_{sample.diagram_type.replace('_diagram', '')}" if "diagram" in sample.diagram_type else "code_to_flowchart"
    task = DIAGRAM_TASKS.get(task_key, DIAGRAM_TASKS["code_to_class_diagram"])
    
    if direction == "code_to_diagram":
        user_template = random.choice(task["user_templates"])
        user_content = user_template.format(lang=sample.language, code=sample.code[:500])
        
        template = MERMAID_TEMPLATES.get(sample.diagram_type, MERMAID_TEMPLATES["class_diagram"])
        assistant_content = template.format(content=sample.diagram_content)
    else:
        # Diagram to code (reverse)
        template = MERMAID_TEMPLATES.get(sample.diagram_type, MERMAID_TEMPLATES["class_diagram"])
        user_content = f"Implement the code for this diagram:\n{template.format(content=sample.diagram_content)}"
        assistant_content = f"```{sample.language}\n{sample.code[:1000]}\n```"
    
    return {
        "id": sample.id,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "domain": "code_diagram",
        "category": sample.diagram_type,
        "direction": direction,
        "source_dataset": sample.source_dataset,
        "modalities": {
            "image": [],
            "audio": [],
            "video": [],
        },
        "diagram_format": sample.diagram_format,
    }


def download_and_process_dataset(
    dataset_name: str,
    output_dir: Path,
    limit: Optional[int] = None,
) -> int:
    """Download and process a code-diagram dataset."""
    if not HF_AVAILABLE:
        logger.error("datasets library not available")
        return 0
    
    config = CODE_DIAGRAM_DATASETS.get(dataset_name)
    if not config:
        logger.error(f"Unknown dataset: {dataset_name}")
        return 0
    
    logger.info(f"Loading {dataset_name} from {config['hf_path']}...")
    
    try:
        ds = load_dataset(
            config["hf_path"],
            split=config["split"],
            streaming=config.get("streaming", False),
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return 0
    
    output_path = output_dir / f"{dataset_name}.jsonl"
    total = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(ds):
            if limit and idx >= limit * 2:  # Process more to get enough valid samples
                break
            
            normalized = normalize_code_sample(sample, idx, dataset_name)
            if normalized:
                # Generate both directions
                for direction in ["code_to_diagram", "diagram_to_code"]:
                    messages_sample = sample_to_messages(normalized, direction)
                    f.write(json.dumps(messages_sample, ensure_ascii=False) + "\n")
                    total += 1
                    
                    if limit and total >= limit:
                        break
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} samples, generated {total} pairs")
            
            if limit and total >= limit:
                break
    
    logger.info(f"Wrote {total} samples to {output_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Download code-to-diagram datasets")
    parser.add_argument("--datasets", nargs="+", 
                        default=["code_explanations", "database_schemas"],
                        choices=list(CODE_DIAGRAM_DATASETS.keys()) + ["all"],
                        help="Datasets to download")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/e/data/multimodal-fullstack-dataset/code_diagram",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per dataset")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Code-Diagram Datasets:\n")
        for name, config in CODE_DIAGRAM_DATASETS.items():
            print(f"  {name}")
            print(f"    HuggingFace: {config['hf_path']}")
            print(f"    Description: {config['description']}")
            print()
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_download = list(CODE_DIAGRAM_DATASETS.keys()) if "all" in args.datasets else args.datasets
    
    log_header(
        logger,
        "CODE-DIAGRAM DATASET DOWNLOADER",
        {
            "Datasets": ", ".join(datasets_to_download),
            "Output": str(output_dir),
            "Limit": args.limit or "None",
        },
    )
    
    total_samples = 0
    for dataset_name in datasets_to_download:
        count = download_and_process_dataset(dataset_name, output_dir, args.limit)
        total_samples += count
    
    log_completion(
        logger,
        "Code-Diagram Dataset Download",
        {
            "Total samples": total_samples,
            "Datasets": len(datasets_to_download),
            "Output": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()

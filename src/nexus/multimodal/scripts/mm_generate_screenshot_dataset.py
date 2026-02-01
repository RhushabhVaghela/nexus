#!/usr/bin/env python3
"""
mm_generate_screenshot_dataset.py

Generate a simple multimodal dataset of screenshot-based Q&A samples.

- Input: a directory of screenshot images (PNG/JPG).
- Output: JSONL with:
    - messages (user + assistant)
    - domain/category
    - modalities.image entries pointing to the screenshot file

This fits the unified schema consumed by:
- 07_validate_all_datasets.py (messages-based checks)
- utils/data_mixer.py (preserves modalities block)
- multimodal decoders during training.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion  # type: ignore

logger = setup_logger(__name__, "logs/mm_generate_screenshot.log")

CONFIG = {
    "input_image_dir": "/mnt/e/data/mm_raw/screenshots",
    "output_dir": "/mnt/e/data/multimodal-fullstack-dataset/screenshot_error_log",
    "samples_per_file": 50_000,
    "seed": 42,
}

USER_TEMPLATES = [
    "Look at this screenshot and explain what is going wrong, then propose a fix.",
    "Explain the main issue visible in this screenshot and how to resolve it.",
    "What is the error shown in this screenshot, and what steps should I take?",
    "Please diagnose the problem in this screenshot and give a clear fix.",
]

ANSWER_TEMPLATES = [
    "The screenshot shows an error message in the IDE. Summarize the error text and identify the root cause. Then outline concrete steps to fix it.",
    "The screenshot likely contains a stack trace or compile error. First, restate the key error line, then provide a systematic approach to debug and resolve it.",
    "Describe what part of the UI indicates failure (e.g., red banner, console output). Then give a practical, step-by-step fix in the context of a fullstack app.",
]


def list_images(image_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files: List[Path] = []
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def build_sample(image_path: Path, idx: int) -> Dict:
    """Build one multimodal sample for a screenshot."""
    user_prompt = random.choice(USER_TEMPLATES)
    answer_hint = random.choice(ANSWER_TEMPLATES)

    # For now, assistant content is a templated hint.
    # During training, the model learns to map visual info -> full answer.
    assistant_text = (
        f"{answer_hint}\n\n"
        "Focus on:\n"
        "- The specific error message and code snippet visible.\n"
        "- Likely cause (e.g., null reference, missing dependency, bad config).\n"
        "- Concrete changes to make in code or settings."
    )

    rel_path = str(image_path)  # keep absolute or relative as you prefer

    sample = {
        "id": f"mm_screenshot_{idx:08d}",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "domain": "multimodal_fullstack",
        "category": "screenshot_error_log",
        "modalities": {
            "image": [
                {
                    "path": rel_path,
                    "type": "screenshot",
                    "description": "IDE / app screenshot with error or UI state",
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
        "MULTIMODAL SCREENSHOT DATASET GENERATOR",
        {
            "Input images": str(image_dir),
            "Output": str(output_dir),
        },
    )

    images = list_images(image_dir)
    logger.info(f"Found {len(images)} screenshot images")

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
        "Multimodal Screenshot Dataset",
        {"Total samples": total, "Output": str(output_dir)},
    )


if __name__ == "__main__":
    main()

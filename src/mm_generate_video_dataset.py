#!/usr/bin/env python3
"""
mm_generate_video_dataset.py

Generate multimodal dataset for video understanding tasks.

Categories:
- Code walkthrough videos (screen recordings)
- Tutorial videos
- Bug reproduction videos
- UI/UX demo videos
- Deployment/DevOps screencasts
- System monitoring dashboards
- Live coding sessions
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_generate_video.log")

CONFIG = {
    "input_video_dir": "/mnt/e/data/mm_raw/videos",
    "output_dir": "/mnt/e/data/multimodal-fullstack-dataset/video_understanding",
    "samples_per_file": 10_000,
    "seed": 42,
    "frame_sample_fps": 1,  # Sample 1 frame per second for analysis
}

# Video categories and their templates
VIDEO_TYPES = [
    "code_walkthrough",
    "tutorial",
    "bug_reproduction",
    "ui_demo",
    "deployment_screencast",
    "monitoring_dashboard",
    "live_coding",
    "code_review_session",
]

USER_TEMPLATES = {
    "code_walkthrough": [
        "Summarize what this code walkthrough video explains.",
        "What are the key concepts demonstrated in this video?",
        "Describe the code structure shown in this walkthrough.",
        "What programming patterns are being demonstrated?",
    ],
    "tutorial": [
        "Create a step-by-step guide from this tutorial video.",
        "What skills does this tutorial teach?",
        "List the main steps shown in this tutorial.",
        "What prerequisites would someone need before watching this?",
    ],
    "bug_reproduction": [
        "Describe the bug being reproduced in this video.",
        "What are the steps to reproduce this issue?",
        "What is the expected vs actual behavior shown?",
        "Suggest potential fixes based on this reproduction.",
    ],
    "ui_demo": [
        "Describe the user interface shown in this demo.",
        "What are the main features demonstrated?",
        "Identify usability issues visible in this demo.",
        "How would you improve the UX shown here?",
    ],
    "deployment_screencast": [
        "Summarize the deployment process shown in this video.",
        "What infrastructure is being set up?",
        "List the commands and configurations used.",
        "Identify potential security concerns in this deployment.",
    ],
    "monitoring_dashboard": [
        "Describe the metrics shown in this monitoring dashboard.",
        "What alerts or issues are visible?",
        "Analyze the system health based on these metrics.",
        "What improvements would you suggest for this dashboard?",
    ],
    "live_coding": [
        "Summarize what was built in this live coding session.",
        "What approaches did the developer try?",
        "What refactoring opportunities exist in the final code?",
        "What were the key decisions made during development?",
    ],
    "code_review_session": [
        "Summarize the feedback given in this code review.",
        "What issues were identified?",
        "What suggestions for improvement were made?",
        "Rate the quality of the code being reviewed.",
    ],
}

ANSWER_TEMPLATES = {
    "code_walkthrough": (
        "This code walkthrough demonstrates:\n\n"
        "**Main Topics:**\n"
        "- Architecture overview and file structure\n"
        "- Key components and their responsibilities\n"
        "- Data flow and state management\n"
        "- Integration points and APIs\n\n"
        "**Code Highlights:**\n"
        "- Design patterns used (e.g., Repository, Factory)\n"
        "- Error handling strategies\n"
        "- Testing approach\n\n"
        "**Key Takeaways:**\n"
        "- Best practices demonstrated\n"
        "- Areas for potential improvement"
    ),
    "tutorial": (
        "Step-by-Step Tutorial Summary:\n\n"
        "**Prerequisites:**\n"
        "- Required software and versions\n"
        "- Prior knowledge needed\n\n"
        "**Steps:**\n"
        "1. Setup and configuration\n"
        "2. Core implementation\n"
        "3. Testing and validation\n"
        "4. Deployment/output\n\n"
        "**Learning Outcomes:**\n"
        "- Skills gained\n"
        "- Concepts understood\n"
        "- Practical applications"
    ),
    "bug_reproduction": (
        "Bug Reproduction Analysis:\n\n"
        "**Issue Description:**\n"
        "- Component affected\n"
        "- Severity and impact\n\n"
        "**Reproduction Steps:**\n"
        "1. Initial state/setup\n"
        "2. Actions taken\n"
        "3. Trigger condition\n"
        "4. Observed failure\n\n"
        "**Expected vs Actual:**\n"
        "- Expected: [normal behavior]\n"
        "- Actual: [buggy behavior]\n\n"
        "**Potential Fixes:**\n"
        "- Root cause analysis\n"
        "- Suggested solutions"
    ),
    "ui_demo": (
        "UI/UX Demo Analysis:\n\n"
        "**Interface Overview:**\n"
        "- Layout and navigation structure\n"
        "- Key components and widgets\n"
        "- Visual design elements\n\n"
        "**Features Demonstrated:**\n"
        "- Core functionality\n"
        "- User interactions\n"
        "- Responsive behavior\n\n"
        "**UX Assessment:**\n"
        "- Strengths: [positive aspects]\n"
        "- Improvements: [areas to enhance]\n"
        "- Accessibility considerations"
    ),
    "deployment_screencast": (
        "Deployment Process Summary:\n\n"
        "**Infrastructure:**\n"
        "- Platform/cloud provider\n"
        "- Services configured\n"
        "- Network topology\n\n"
        "**Steps Performed:**\n"
        "1. Environment setup\n"
        "2. Configuration management\n"
        "3. Build and deploy\n"
        "4. Verification\n\n"
        "**Security Considerations:**\n"
        "- Secrets management\n"
        "- Access controls\n"
        "- Network security"
    ),
    "monitoring_dashboard": (
        "Monitoring Dashboard Analysis:\n\n"
        "**Metrics Observed:**\n"
        "- CPU, Memory, Disk usage\n"
        "- Request rates and latencies\n"
        "- Error rates and types\n"
        "- Custom business metrics\n\n"
        "**System Health:**\n"
        "- Overall status: [healthy/degraded/critical]\n"
        "- Alerts triggered\n"
        "- Anomalies detected\n\n"
        "**Recommendations:**\n"
        "- Missing metrics to add\n"
        "- Alert thresholds to adjust\n"
        "- Visualization improvements"
    ),
    "live_coding": (
        "Live Coding Session Summary:\n\n"
        "**Goal:**\n"
        "- What was being built\n"
        "- Requirements addressed\n\n"
        "**Development Process:**\n"
        "- Approaches tried\n"
        "- Challenges encountered\n"
        "- Solutions discovered\n\n"
        "**Final Implementation:**\n"
        "- Code structure\n"
        "- Design decisions\n"
        "- Tests written\n\n"
        "**Learnings:**\n"
        "- Techniques demonstrated\n"
        "- Tips and tricks shared"
    ),
    "code_review_session": (
        "Code Review Summary:\n\n"
        "**Code Under Review:**\n"
        "- Files/components reviewed\n"
        "- Purpose of changes\n\n"
        "**Issues Identified:**\n"
        "- Critical: [blocking issues]\n"
        "- Major: [significant concerns]\n"
        "- Minor: [style/nits]\n\n"
        "**Suggestions:**\n"
        "- Refactoring opportunities\n"
        "- Additional tests needed\n"
        "- Documentation gaps\n\n"
        "**Verdict:**\n"
        "- Approve / Request changes"
    ),
}


def list_video_files(video_dir: Path) -> List[Path]:
    """List all video files in directory."""
    exts = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
    files: List[Path] = []
    for p in video_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def get_video_metadata(video_path: Path) -> Dict:
    """Extract basic video metadata (duration, resolution if available)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "frame_count": int(frame_count),
        }
    except Exception:
        return {"duration_seconds": 0, "resolution": "unknown", "fps": 0, "frame_count": 0}


def get_sidecar_transcript(video_path: Path) -> Optional[str]:
    """Check for a .srt or .vtt subtitle file."""
    for ext in [".srt", ".vtt", ".txt"]:
        sidecar = video_path.with_suffix(ext)
        if sidecar.exists():
            try:
                return sidecar.read_text(encoding="utf-8")[:2000]
            except Exception:
                pass
    return None


def build_sample(video_path: Path, idx: int) -> Dict:
    """Build one multimodal sample for a video."""
    # Determine video type from filename hints
    video_type = random.choice(VIDEO_TYPES)
    filename_lower = video_path.stem.lower()
    
    for vtype in VIDEO_TYPES:
        if vtype.replace("_", "") in filename_lower or vtype in filename_lower:
            video_type = vtype
            break
    
    user_prompt = random.choice(USER_TEMPLATES[video_type])
    answer_hint = ANSWER_TEMPLATES[video_type]
    
    # Get metadata
    metadata = get_video_metadata(video_path)
    transcript = get_sidecar_transcript(video_path)

    sample = {
        "id": f"mm_video_{idx:08d}",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer_hint},
        ],
        "domain": "multimodal_fullstack",
        "category": f"video_{video_type}",
        "modalities": {
            "image": [],
            "audio": [],
            "video": [
                {
                    "path": str(video_path),
                    "type": "screencast",
                    "subtype": video_type,
                    "description": f"{video_type.replace('_', ' ').title()} video",
                    "metadata": metadata,
                    "has_transcript": transcript is not None,
                }
            ],
        },
    }
    
    if transcript:
        sample["transcript_preview"] = transcript[:1000] + "..." if len(transcript) > 1000 else transcript
    
    return sample


def main():
    random.seed(CONFIG["seed"])

    video_dir = Path(CONFIG["input_video_dir"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_header(
        logger,
        "MULTIMODAL VIDEO UNDERSTANDING DATASET GENERATOR",
        {
            "Input videos": str(video_dir),
            "Output": str(output_dir),
            "Frame sample FPS": CONFIG["frame_sample_fps"],
        },
    )

    video_files = list_video_files(video_dir)
    logger.info(f"Found {len(video_files)} video files")

    samples_per_file = CONFIG["samples_per_file"]
    batch: List[Dict] = []
    batch_idx = 0
    total = 0

    for idx, video_path in enumerate(video_files):
        sample = build_sample(video_path, idx)
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
        "Multimodal Video Understanding Dataset",
        {"Total samples": total, "Output": str(output_dir)},
    )


if __name__ == "__main__":
    main()

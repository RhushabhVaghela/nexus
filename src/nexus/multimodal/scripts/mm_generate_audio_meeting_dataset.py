#!/usr/bin/env python3
"""
mm_generate_audio_meeting_dataset.py

Generate multimodal dataset for audio meeting transcription and summarization.

- Input: directory of audio files (WAV/MP3) with optional transcript sidecars
- Output: JSONL with messages + modalities.audio for training

Categories:
- Meeting summaries
- Action item extraction
- Decision tracking
- Technical discussion analysis
- Code review discussions
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/mm_generate_audio_meeting.log")

CONFIG = {
    "input_audio_dir": "/mnt/e/data/mm_raw/audio_meetings",
    "output_dir": "/mnt/e/data/multimodal-fullstack-dataset/audio_meeting",
    "samples_per_file": 50_000,
    "seed": 42,
}

# Meeting type templates
MEETING_TYPES = [
    "standup",
    "sprint_planning",
    "code_review",
    "architecture_discussion",
    "incident_review",
    "onboarding",
    "technical_interview",
    "pair_programming",
]

USER_TEMPLATES = {
    "standup": [
        "Summarize this standup meeting and list blockers mentioned.",
        "What did each team member report in this daily standup?",
        "Extract the key updates and blockers from this standup.",
        "Create action items from this standup meeting.",
    ],
    "sprint_planning": [
        "Summarize the sprint goals discussed in this meeting.",
        "What user stories were committed to in this sprint planning?",
        "Extract story point estimates from this planning session.",
        "List the dependencies and risks identified in this meeting.",
    ],
    "code_review": [
        "Summarize the code review feedback from this discussion.",
        "What issues were identified in this code review?",
        "List the suggested improvements from this review session.",
        "What was the final decision on the code changes?",
    ],
    "architecture_discussion": [
        "Summarize the architecture decisions made in this meeting.",
        "What trade-offs were discussed in this architecture session?",
        "Extract the ADRs (Architecture Decision Records) from this discussion.",
        "What alternatives were considered and why were they rejected?",
    ],
    "incident_review": [
        "Create a post-mortem summary from this incident review.",
        "What was the root cause identified in this discussion?",
        "List the action items and preventive measures from this review.",
        "Who was responsible for each follow-up action?",
    ],
    "onboarding": [
        "Summarize the key onboarding topics covered in this session.",
        "What systems and tools were introduced in this onboarding?",
        "List the follow-up resources mentioned in this discussion.",
        "What questions did the new team member ask?",
    ],
    "technical_interview": [
        "Summarize the candidate's technical responses.",
        "What coding problems were discussed in this interview?",
        "Evaluate the candidate's system design approach.",
        "What follow-up questions would you ask based on this discussion?",
    ],
    "pair_programming": [
        "Summarize what was accomplished in this pair programming session.",
        "What approaches were tried and why did some fail?",
        "List the key learnings from this collaboration.",
        "What refactoring was performed during this session?",
    ],
}

ANSWER_TEMPLATES = {
    "standup": (
        "Standup Meeting Summary:\n\n"
        "**Team Updates:**\n"
        "- [Team member 1]: Yesterday worked on X, today will work on Y\n"
        "- [Team member 2]: Completed feature Z, starting on A\n\n"
        "**Blockers Identified:**\n"
        "- [Blocker 1]: Description and owner\n"
        "- [Blocker 2]: Description and owner\n\n"
        "**Action Items:**\n"
        "- Follow up on blocker resolution\n"
        "- Coordinate on shared dependencies"
    ),
    "sprint_planning": (
        "Sprint Planning Summary:\n\n"
        "**Sprint Goal:** [Primary objective for this sprint]\n\n"
        "**Committed Stories:**\n"
        "- Story 1 (X points): Description\n"
        "- Story 2 (Y points): Description\n\n"
        "**Total Capacity:** X story points\n"
        "**Committed Points:** Y story points\n\n"
        "**Risks & Dependencies:**\n"
        "- External API availability\n"
        "- Cross-team coordination needs"
    ),
    "code_review": (
        "Code Review Summary:\n\n"
        "**Files Reviewed:** [List of files]\n\n"
        "**Issues Found:**\n"
        "- Critical: [Security/performance issues]\n"
        "- Major: [Logic errors, missing tests]\n"
        "- Minor: [Style, naming conventions]\n\n"
        "**Suggested Improvements:**\n"
        "- Refactor X for better readability\n"
        "- Add unit tests for edge cases\n\n"
        "**Decision:** [Approved/Needs changes]"
    ),
    "architecture_discussion": (
        "Architecture Discussion Summary:\n\n"
        "**Context:** [Problem being solved]\n\n"
        "**Decision:** [Chosen approach]\n\n"
        "**Alternatives Considered:**\n"
        "- Option A: Pros/Cons\n"
        "- Option B: Pros/Cons\n\n"
        "**Trade-offs:**\n"
        "- Performance vs. maintainability\n"
        "- Cost vs. scalability\n\n"
        "**Next Steps:**\n"
        "- Create detailed design doc\n"
        "- Prototype critical components"
    ),
    "incident_review": (
        "Incident Post-Mortem Summary:\n\n"
        "**Incident:** [Brief description]\n"
        "**Duration:** [Start to resolution]\n"
        "**Impact:** [Users affected, revenue impact]\n\n"
        "**Timeline:**\n"
        "- T+0: Issue detected\n"
        "- T+X: Root cause identified\n"
        "- T+Y: Fix deployed\n\n"
        "**Root Cause:** [Technical explanation]\n\n"
        "**Action Items:**\n"
        "- [Owner]: Add monitoring for X\n"
        "- [Owner]: Improve runbook for Y"
    ),
    "onboarding": (
        "Onboarding Session Summary:\n\n"
        "**Topics Covered:**\n"
        "- Team structure and roles\n"
        "- Development environment setup\n"
        "- Key repositories and services\n"
        "- On-call procedures\n\n"
        "**Key Resources:**\n"
        "- Wiki: [link]\n"
        "- Runbooks: [link]\n"
        "- Team calendar: [link]\n\n"
        "**Follow-up:**\n"
        "- Schedule 1:1 with mentor\n"
        "- Review first ticket assignment"
    ),
    "technical_interview": (
        "Technical Interview Summary:\n\n"
        "**Coding Round:**\n"
        "- Problem: [Description]\n"
        "- Approach: [Candidate's solution]\n"
        "- Complexity: O(n) time, O(1) space\n"
        "- Execution: [Completed/Partial]\n\n"
        "**System Design:**\n"
        "- Scalability considerations\n"
        "- Trade-off discussions\n"
        "- Areas of strength/weakness\n\n"
        "**Recommendation:** [Hire/No hire/More rounds]"
    ),
    "pair_programming": (
        "Pair Programming Session Summary:\n\n"
        "**Task:** [What was being worked on]\n\n"
        "**Approach:**\n"
        "- Started with: [Initial approach]\n"
        "- Pivoted to: [Alternative if applicable]\n"
        "- Final solution: [Description]\n\n"
        "**Code Changes:**\n"
        "- Added: [New functionality]\n"
        "- Refactored: [Improved areas]\n"
        "- Tests: [Coverage added]\n\n"
        "**Learnings:**\n"
        "- Key insight shared\n"
        "- Pattern discovered"
    ),
}


def list_audio_files(audio_dir: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    files: List[Path] = []
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def get_transcript_sidecar(audio_path: Path) -> Optional[str]:
    """Check for a .txt sidecar transcript file."""
    transcript_path = audio_path.with_suffix(".txt")
    if transcript_path.exists():
        try:
            return transcript_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def build_sample(audio_path: Path, idx: int) -> Dict:
    """Build one multimodal sample for an audio meeting."""
    # Pick meeting type based on filename hints or random
    meeting_type = random.choice(MEETING_TYPES)
    
    filename_lower = audio_path.stem.lower()
    for mtype in MEETING_TYPES:
        if mtype.replace("_", "") in filename_lower or mtype in filename_lower:
            meeting_type = mtype
            break
    
    user_prompt = random.choice(USER_TEMPLATES[meeting_type])
    answer_hint = ANSWER_TEMPLATES[meeting_type]
    
    # Check for transcript sidecar
    transcript = get_transcript_sidecar(audio_path)

    sample = {
        "id": f"mm_audio_meeting_{idx:08d}",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer_hint},
        ],
        "domain": "multimodal_fullstack",
        "category": f"audio_meeting_{meeting_type}",
        "modalities": {
            "image": [],
            "audio": [
                {
                    "path": str(audio_path),
                    "type": "meeting",
                    "subtype": meeting_type,
                    "description": f"{meeting_type.replace('_', ' ').title()} meeting audio",
                    "has_transcript_sidecar": transcript is not None,
                }
            ],
            "video": [],
        },
    }
    
    # Optionally include transcript reference
    if transcript:
        sample["transcript_preview"] = transcript[:500] + "..." if len(transcript) > 500 else transcript
    
    return sample


def main():
    random.seed(CONFIG["seed"])

    audio_dir = Path(CONFIG["input_audio_dir"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_header(
        logger,
        "MULTIMODAL AUDIO MEETING DATASET GENERATOR",
        {
            "Input audio": str(audio_dir),
            "Output": str(output_dir),
        },
    )

    audio_files = list_audio_files(audio_dir)
    logger.info(f"Found {len(audio_files)} audio files")

    samples_per_file = CONFIG["samples_per_file"]
    batch: List[Dict] = []
    batch_idx = 0
    total = 0

    for idx, audio_path in enumerate(audio_files):
        sample = build_sample(audio_path, idx)
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
        "Multimodal Audio Meeting Dataset",
        {"Total samples": total, "Output": str(output_dir)},
    )


if __name__ == "__main__":
    main()

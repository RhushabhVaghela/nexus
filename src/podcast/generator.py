#!/usr/bin/env python3
"""
Podcast script generator (NotebookLM-style, 2 speakers with live interaction).

- Takes one or more documents as input (plain text).
- Generates a 2-speaker dialogue script.
- Supports mid-podcast user interruptions by generating follow-up dialogue
  that responds to the user and then naturally returns to the main topic.

This module is model-agnostic: it calls a generic `call_llm()` function that
you can wire to your local Nexus model, OpenAI-compatible endpoint, etc.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Callable


Role = Literal["Host A", "Host B", "User"]


@dataclass
class Turn:
    speaker: Role
    text: str
    vibe: str = "neutral"  # neutral, excited, thoughtful, curious, skeptical, etc.


@dataclass
class PodcastScript:
    turns: List[Turn]

    def to_dict(self) -> Dict[str, Any]:
        return {"turns": [t.__dict__ for t in self.turns]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PodcastScript":
        turns = [Turn(**t) for t in data.get("turns", [])]
        return cls(turns=turns)


# ═══════════════════════════════════════════════════════════════
# LLM CALL ADAPTER
# ═══════════════════════════════════════════════════════════════

import requests

def call_llm(messages: List[Dict[str, str]], *, model: str = "nexus-podcast") -> str:
    """
    Call a chat-completions-compatible HTTP endpoint and return the raw assistant text.

    Expected server API (OpenAI-compatible):
      POST /v1/chat/completions
      {
        "model": "nexus-podcast",
        "messages": [...],
        "temperature": 0.7
      }

    Adjust URL, headers, and JSON keys to your deployment.
    """
    url = os.getenv("NEXUS_API_URL", "http://localhost:8000/v1/chat/completions")
    api_key = os.getenv("NEXUS_API_KEY", "")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-style response
    content = data["choices"][0]["message"]["content"]
    return content



# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

BASE_SYSTEM_PROMPT = """
You are a podcast script writer for a two-host show.

Hosts:
- "Host A": more analytical, structured, explains concepts clearly.
- "Host B": more conversational, adds examples, stories, and reactions.

Write a dialogue where they discuss the provided documents.
Goals:
- Be accurate to the source material.
- Explain concepts clearly.
- Occasionally summarize and preview what comes next.
- Keep each turn a few sentences long (avoid huge monologues).
- Alternate speakers naturally: Host A, then Host B, etc.
- Do NOT include stage directions like [SFX], [pause]; only plain dialogue.

Output strictly as JSON with this structure:

{
  "turns": [
    {"speaker": "Host A", "text": "...", "vibe": "thoughtful" },
    {"speaker": "Host B", "text": "...", "vibe": "excited" }
  ]
}
""".strip()


INTERRUPT_SYSTEM_PROMPT = """
You are continuing an ongoing two-host podcast.

Hosts:
- "Host A": analytical and structured.
- "Host B": conversational and reactive.

There is an existing conversation (context).
The listener (User) has interrupted with a question or comment.

Write a short follow-up segment where the hosts:
1) Acknowledge the user's interruption.
2) Answer or react to the user's message clearly.
3) Smoothly return to the main topic based on the context.

Keep 4–8 turns, with short, natural utterances.
Only use speakers "Host A" and "Host B".
Output strictly as JSON with:

{
  "turns": [
    {"speaker": "Host A", "text": "...", "vibe": "supportive"},
    {"speaker": "Host B", "text": "...", "vibe": "curious"}
  ]
}
""".strip()


# ═══════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _parse_script_json(raw: str) -> PodcastScript:
    """
    Parse the model output into PodcastScript.

    This expects the model to return a JSON object with a "turns" list.
    If parsing fails, an error is raised so you can inspect the raw output.
    """
    try:
        data = json.loads(raw)
        return PodcastScript.from_dict(data)
    except Exception as e:
        raise ValueError(f"Failed to parse podcast script JSON: {e}\nRAW:\n{raw}") from e


def generate_podcast_script(
    documents: List[str],
    topic_hint: Optional[str] = None,
    *,
    llm: Callable[[List[Dict[str, str]]], str] = call_llm,
) -> PodcastScript:
    """
    Generate an initial 2-speaker podcast script from documents.

    Args:
        documents: List of document strings (markdown, notes, transcripts, etc.).
        topic_hint: Optional short hint like "Fullstack engineering for mobile devs".
        llm: Function(messages) -> str returning the JSON string.

    Returns:
        PodcastScript with alternating Host A / Host B turns.
    """
    docs_preview = "\n\n---\n\n".join(documents[:5])
    if len(docs_preview) > 8000:
        docs_preview = docs_preview[:8000] + "\n\n[TRUNCATED]"

    user_content_lines = ["Here are the documents for the podcast:\n", docs_preview]
    if topic_hint:
        user_content_lines.append(f"\nTopic hint: {topic_hint}")

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_content_lines)},
    ]

    raw = llm(messages)
    return _parse_script_json(raw)


def handle_user_interrupt(
    base_script: PodcastScript,
    user_message: str,
    *,
    llm: Callable[[List[Dict[str, str]]], str] = call_llm,
    max_context_turns: int = 20,
) -> PodcastScript:
    """
    Generate a follow-up segment after a user interruption.

    Args:
        base_script: The podcast script generated so far (or a slice of it).
        user_message: The user's question/comment (from text or ASR).
        llm: Function(messages) -> str returning the JSON string.
        max_context_turns: Number of trailing turns from history to include as context.

    Returns:
        PodcastScript containing only the NEW follow-up turns.
        The caller can append these to the master script.
    """
    # Take the last `max_context_turns` as context
    context_turns = base_script.turns[-max_context_turns:]
    context_text = ""
    for t in context_turns:
        context_text += f"{t.speaker}: {t.text}\n"

    messages = [
        {"role": "system", "content": INTERRUPT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the recent podcast context:\n"
                f"{context_text}\n\n"
                f"The user has interrupted and said:\n\"{user_message}\"\n\n"
                "Please continue the podcast according to the instructions."
            ),
        },
    ]

    raw = llm(messages)
    return _parse_script_json(raw)


# ═══════════════════════════════════════════════════════════════
# OMNI-ADAPTER
# ═══════════════════════════════════════════════════════════════

import torch
from transformers import AutoTokenizer
from multiprocessing import Process, Queue

# Global model cache (only used if running as script)
_MODEL_CACHE = None
_TOKENIZER_CACHE = None

def get_omni_model(model_path: str):
    global _MODEL_CACHE, _TOKENIZER_CACHE
    if _MODEL_CACHE is None:
        from multimodal.model import OmniMultimodalLM
        print(f"Loading OmniModel from {model_path}...")
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _MODEL_CACHE = OmniMultimodalLM(model_path)
    return _MODEL_CACHE, _TOKENIZER_CACHE

def call_omni_llm(messages: List[Dict[str, str]], model_path: str = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4") -> str:
    """
    Adapter that calls the actual OmniMultimodalLM logic.
    """
    model, tokenizer = get_omni_model(model_path)
    
    # 1. Apply Template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.wrapper.llm.device)
    
    # 3. Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True
        )
        
    # 4. Decode
    # Skip input tokens
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ═══════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    import sys
    from pathlib import Path
    
    # Add src to path if needed
    sys.path.append(str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Generate a 2-speaker podcast script.")
    parser.add_argument("--docs", type=str, nargs="+", required=True)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--out", type=str, default="-")
    parser.add_argument("--model", type=str, default="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")

    args = parser.parse_args()

    docs_content: List[str] = []
    for path in args.docs:
        with open(path, "r", encoding="utf-8") as f:
            docs_content.append(f.read())
            
    # Bind the model path to the callable
    def bound_llm(msgs):
        return call_omni_llm(msgs, model_path=args.model)

    try:
        script = generate_podcast_script(docs_content, topic_hint=args.topic, llm=bound_llm)
        out_data = script.to_dict()
        out_json = json.dumps(out_data, ensure_ascii=False, indent=2)

        if args.out == "-":
            sys.stdout.write(out_json + "\n")
        else:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(out_json + "\n")
                
    except Exception as e:
        print(f"❌ Error generating podcast: {e}", file=sys.stderr)



if __name__ == "__main__":
    main()

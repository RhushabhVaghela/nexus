#!/usr/bin/env python3
"""
Podcast audio synthesizer and interactive player.

Responsibilities:
- Take a PodcastScript (list of {speaker, text} turns).
- Convert each turn to audio via a TTS endpoint/CLI.
- Play audio in sequence with a small queue.
- Allow live user interaction:
    - Pause current playback.
    - Send the user message to the LLM via `handle_user_interrupt`.
    - Append the new Host A/B turns.
    - Resume playback from the new turns.

This module is agnostic to the exact TTS backend:
- You can wire it to:
    - A local CLI (e.g., `xtts-cli --speaker host_a "text"`),
    - A HTTP service (Coqui XTTS, custom TTS),
    - Or any other synthesizer.

All integration happens in `synthesize_tts()` and `play_audio()`.
"""

import os
import sys
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable, Literal

# Import generator utilities
from podcast.generator import (
    PodcastScript,
    Turn,
    handle_user_interrupt,
    call_llm,
)

SpeakerName = Literal["Host A", "Host B"]


@dataclass
class AudioTurn:
    """A single audio item in the playback queue."""
    speaker: SpeakerName
    text: str
    audio_path: Path


# ═══════════════════════════════════════════════════════════════
# TTS BACKEND ADAPTERS
# ═══════════════════════════════════════════════════════════════

def synthesize_tts(
    speaker: SpeakerName,
    text: str,
    out_dir: Path,
    tts_backend: str = "http",
    *,
    tts_url: str = "http://localhost:5002/api/tts",
    voice_map: Optional[Dict[SpeakerName, str]] = None,
) -> Path:
    """
    Synthesize speech for a given speaker + text, save to WAV/MP3 in out_dir,
    and return the audio file path.

    tts_backend:
        - "http": POST to a TTS HTTP server (e.g., Coqui XTTS, custom).
        - "cli":  invoke local CLI (must be installed on PATH).

    tts_url:
        - HTTP endpoint used when tts_backend == "http".

    voice_map:
        - Optional mapping from speaker names ("Host A", "Host B")
          to model-specific voice IDs or speaker embeddings.

    NOTE:
        This function is intentionally concrete, not a stub. It assumes:
        - HTTP backend expects JSON: {"text": "...", "voice": "..."}
          and returns raw audio bytes.
        - CLI backend is a generic example; adjust command as needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_speaker = speaker.replace(" ", "_").lower()
    ts = int(time.time() * 1000)
    out_path = out_dir / f"{ts}_{safe_speaker}.wav"

    voice_id = None
    if voice_map is not None:
        voice_id = voice_map.get(speaker)
    if voice_id is None:
        # Default to speaker name as voice id if not provided
        voice_id = safe_speaker

    if tts_backend == "http":
        import requests  # lazy import

        payload = {
            "text": text,
            "voice": voice_id,
        }
        resp = requests.post(tts_url, json=payload, timeout=60)
        resp.raise_for_status()
        audio_bytes = resp.content
        with open(out_path, "wb") as f:
            f.write(audio_bytes)

    elif tts_backend == "cli":
        # Example CLI call; adapt to your actual TTS binary
        # For example, if you have `xtts-cli --voice VOICE --output FILE "TEXT"`
        import subprocess

        cmd = [
            "xtts-cli",
            "--voice",
            voice_id,
            "--output",
            str(out_path),
            text,
        ]
        subprocess.run(cmd, check=True)

    else:
        raise ValueError(f"Unknown tts_backend: {tts_backend}")

    return out_path


def play_audio(audio_path: Path):
    """
    Play an audio file.

    This is implemented using a simple cross-platform subprocess approach:
    - On macOS: `afplay`
    - On Linux: `aplay` or `ffplay` (if installed)
    - On Windows: use `powershell` / `wmplayer` / `ffplay`

    Adjust to your environment or replace with a Python audio library.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    import subprocess
    import platform

    system = platform.system().lower()

    if "darwin" in system:
        cmd = ["afplay", str(audio_path)]
    elif "linux" in system:
        # Try aplay, fallback to ffplay
        if shutil.which("aplay"):
            cmd = ["aplay", str(audio_path)]
        else:
            cmd = ["ffplay", "-nodisp", "-autoexit", str(audio_path)]
    elif "windows" in system:
        # Simple powershell-based playback; adapt as needed
        cmd = [
            "powershell",
            "-c",
            f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync();",
        ]
    else:
        raise RuntimeError(f"Unsupported OS for audio playback: {system}")

    subprocess.run(cmd, check=True)


# ═══════════════════════════════════════════════════════════════
# PODCAST PLAYER (QUEUE + INTERACTION)
# ═══════════════════════════════════════════════════════════════

class PodcastPlayer:
    """
    Queue-based podcast player with interactive user interrupts.

    Usage:
        player = PodcastPlayer(script, audio_dir=Path("audio/"))
        player.start()  # runs in background
        ...
        player.on_user_text("Wait, can you explain X again?")
        ...
        player.stop()

    The class:
    - Converts text turns -> audio via TTS lazily.
    - Plays audio one by one.
    - On user interaction:
        - Pauses playback.
        - Calls handle_user_interrupt() to get new turns.
        - Enqueues new audio turns.
        - Resumes playback from there.
    """

    def __init__(
        self,
        script: PodcastScript,
        audio_dir: Path,
        *,
        tts_backend: str = "http",
        tts_url: str = "http://localhost:5002/api/tts",
        voice_map: Optional[Dict[SpeakerName, str]] = None,
        llm_fn: Callable = call_llm,
    ) -> None:
        self.script = script
        self.audio_dir = audio_dir
        self.tts_backend = tts_backend
        self.tts_url = tts_url
        self.voice_map = voice_map or {
            "Host A": "host_a",
            "Host B": "host_b",
        }
        self.llm_fn = llm_fn

        self._queue: "queue.Queue[AudioTurn]" = queue.Queue()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Index of next script turn to be enqueued as audio
        self._next_turn_idx = 0

    # ─────────── Public controls ───────────

    def start(self):
        """Start playback in a background thread."""
        if self._play_thread is not None and self._play_thread.is_alive():
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._play_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._play_thread.start()

    def stop(self):
        """Stop playback and exit the thread."""
        self._stop_event.set()
        self._pause_event.set()
        if self._play_thread is not None:
            self._play_thread.join(timeout=5)

    def pause(self):
        """Pause playback (used when user interrupts)."""
        self._pause_event.set()

    def resume(self):
        """Resume playback after a pause."""
        self._pause_event.clear()

    def on_user_text(self, user_message: str):
        """
        Handle a user text interrupt:

        - Pause playback.
        - Call handle_user_interrupt with the current script + user_message.
        - Append new turns to script.
        - Generate audio for them and enqueue.
        - Resume playback.
        """
        with self._lock:
            self.pause()

            # Generate follow-up segment from LLM
            follow_up = handle_user_interrupt(
                base_script=self.script,
                user_message=user_message,
                llm=self.llm_fn,
            )

            # Append to main script
            start_idx = len(self.script.turns)
            self.script.turns.extend(follow_up.turns)

            # Enqueue audio for new turns
            for turn in follow_up.turns:
                if turn.speaker not in ("Host A", "Host B"):
                    continue
                audio_path = synthesize_tts(
                    speaker=turn.speaker,
                    text=turn.text,
                    out_dir=self.audio_dir,
                    tts_backend=self.tts_backend,
                    tts_url=self.tts_url,
                    voice_map=self.voice_map,
                )
                self._queue.put(AudioTurn(turn.speaker, turn.text, audio_path))

            # Advance next_turn_idx in case playback loop needs it
            self._next_turn_idx = len(self.script.turns)

            self.resume()

    def on_user_audio_transcript(self, transcript: str):
        """Alias for on_user_text when user interacts via speech."""
        self.on_user_text(transcript)

    # ─────────── Internal loop ───────────

    def _enqueue_next_turns_if_needed(self):
        """
        Enqueue remaining script turns as audio until queue has a small buffer.

        This supports lazy synthesis so we don't TTS the whole script up-front.
        """
        BUFFER_SIZE = 3  # how many audio turns ahead we keep
        if self._queue.qsize() >= BUFFER_SIZE:
            return

        while self._next_turn_idx < len(self.script.turns) and self._queue.qsize() < BUFFER_SIZE:
            turn = self.script.turns[self._next_turn_idx]
            self._next_turn_idx += 1

            if turn.speaker not in ("Host A", "Host B"):
                continue  # ignore "User" or other roles

            audio_path = synthesize_tts(
                speaker=turn.speaker,
                text=turn.text,
                out_dir=self.audio_dir,
                tts_backend=self.tts_backend,
                tts_url=self.tts_url,
                voice_map=self.voice_map,
            )
            self._queue.put(AudioTurn(turn.speaker, turn.text, audio_path))

    def _run_loop(self):
        """Main playback loop."""
        while not self._stop_event.is_set():
            # Pause handling
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            # Ensure we have some audio queued
            with self._lock:
                self._enqueue_next_turns_if_needed()

            try:
                audio_turn = self._queue.get(timeout=0.2)
            except queue.Empty:
                # No more items and script exhausted
                if self._next_turn_idx >= len(self.script.turns):
                    # Finished playback
                    break
                continue

            # Play audio (blocking)
            try:
                play_audio(audio_turn.audio_path)
            except Exception as e:
                print(f"[PodcastPlayer] Error playing audio: {e}", file=sys.stderr)

            self._queue.task_done()


# ═══════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive podcast player.")
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to podcast script JSON (output of generator.py).",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="podcast_audio",
        help="Directory where synthesized audio files will be stored.",
    )
    parser.add_argument(
        "--tts-backend",
        type=str,
        default="http",
        choices=["http", "cli"],
        help="TTS backend type.",
    )
    parser.add_argument(
        "--tts-url",
        type=str,
        default="http://localhost:5002/api/tts",
        help="TTS HTTP endpoint when using --tts-backend=http.",
    )
    args = parser.parse_args()

    # Load script
    with open(args.script, "r", encoding="utf-8") as f:
        data = json.load(f)
    script = PodcastScript.from_dict(data)

    player = PodcastPlayer(
        script=script,
        audio_dir=Path(args.audio_dir),
        tts_backend=args.tts_backend,
        tts_url=args.tts_url,
        llm_fn=call_llm,  # wire to your actual LLM client
    )

    print("Starting podcast playback. Type messages to interrupt; Ctrl+C to exit.")
    player.start()

    try:
        while True:
            user_input = input("> ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() in {"quit", "exit", "q"}:
                break
            # Handle user interrupt
            player.on_user_text(user_input.strip())
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()


if __name__ == "__main__":
    main()

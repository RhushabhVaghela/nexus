#!/usr/bin/env python3
"""
Joint streaming orchestrator for triple modality:

- Vision stream (frames / screenshots).
- Ambient audio stream.
- User interaction (text or ASR transcript).

The goal:
- Maintain rolling buffers for each modality.
- Periodically build a unified context and call the LLM.
- Support interactive usage similar to Gemini / NotebookLM live.

This is designed to sit on top of:
- src/streaming/vision.py   (VisionStreamBuffer)
- src/streaming/audio.py    (AudioStreamBuffer / ASR)
- src/streaming/tts.py      (optional TTS for responses)
"""

import time
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any

# You would import your actual buffers here
# from streaming.vision import VisionStreamBuffer
# from streaming.audio import AudioStreamBuffer
# from streaming.tts import TTSStreamer

# For now, define small protocols (interfaces) that your real classes can satisfy.


@dataclass
class VisionFrame:
    timestamp: float
    description: str  # e.g., "screenshot of game HUD" or caption from vision model
    path: Optional[str] = None  # optional image path


@dataclass
class AudioChunk:
    timestamp: float
    transcript: Optional[str]  # ASR transcript or None if not speech
    summary: Optional[str] = None  # optional short description


@dataclass
class UserEvent:
    timestamp: float
    text: str  # user chat text or ASR from mic


class VisionStreamBuffer:
    """Example interface; replace with your actual implementation."""

    def __init__(self, max_seconds: float = 30.0):
        self.max_seconds = max_seconds
        self._frames: List[VisionFrame] = []
        self._lock = threading.Lock()

    def add_frame(self, frame: VisionFrame):
        with self._lock:
            self._frames.append(frame)
            cutoff = time.time() - self.max_seconds
            self._frames = [f for f in self._frames if f.timestamp >= cutoff]

    def get_recent_frames(self) -> List[VisionFrame]:
        with self._lock:
            return list(self._frames)


class AudioStreamBuffer:
    """Example interface; replace with your actual implementation."""

    def __init__(self, max_seconds: float = 30.0):
        self.max_seconds = max_seconds
        self._chunks: List[AudioChunk] = []
        self._lock = threading.Lock()

    def add_chunk(self, chunk: AudioChunk):
        with self._lock:
            self._chunks.append(chunk)
            cutoff = time.time() - self.max_seconds
            self._chunks = [c for c in self._chunks if c.timestamp >= cutoff]

    def get_recent_chunks(self) -> List[AudioChunk]:
        with self._lock:
            return list(self._chunks)


class UserEventBuffer:
    """Buffer for user text/ASR events."""

    def __init__(self, max_events: int = 50):
        self.max_events = max_events
        self._events: List[UserEvent] = []
        self._lock = threading.Lock()

    def add_event(self, event: UserEvent):
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events :]

    def get_recent_events(self) -> List[UserEvent]:
        with self._lock:
            return list(self._events)


# LLM call adapter; wire to OmniMultimodalLM or HTTP endpoint.
LLMFn = Callable[[List[Dict[str, str]]], str]


def call_llm(messages: List[Dict[str, str]]) -> str:
    raise RuntimeError("Replace call_llm in streaming/joint.py with your real client.")


class JointStreamingOrchestrator:
    """
    Orchestrates triple-modality streaming:

    - Maintains rolling buffers for vision, audio, user events.
    - Periodically builds a textual summary + passes raw modalities if needed.
    - Calls the LLM and yields responses.

    You can:
    - Connect this to a UI (VR, mobile app).
    - Attach TTS to read responses aloud.
    """

    def __init__(
        self,
        vision_buffer: VisionStreamBuffer,
        audio_buffer: AudioStreamBuffer,
        user_buffer: UserEventBuffer,
        llm_fn: LLMFn = call_llm,
        interval_sec: float = 5.0,
        tts_engine: Optional[Any] = None,
    ) -> None:
        self.vision_buffer = vision_buffer
        self.audio_buffer = audio_buffer
        self.user_buffer = user_buffer
        self.llm_fn = llm_fn
        self.interval_sec = interval_sec
        self.tts_engine = tts_engine
        
        # Voice Persona State
        self.active_voice = "NATM1"  # Default PersonaPlex voice
        self.active_vibe = "neutral"  # Default VibeVoice vibe

        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_llm_response: Optional[Callable[[str], None]] = None

    def switch_voice(self, voice_id: str):
        """Dynamic Voice Switching Hook."""
        print(f"🔄 Switching voice persona to: {voice_id}")
        self.active_voice = voice_id

    def set_vibe(self, vibe_name: str):
        """Dynamic Vibe Modulation Hook."""
        print(f"🎚️ Setting acoustic vibe to: {vibe_name}")
        self.active_vibe = vibe_name

    def start(self):
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)

    def _build_context_text(self) -> str:
        """
        Build a textual context summary from the three streams.

        This version uses descriptive text, which is easiest to debug.
        Later, you can switch to feeding raw tensors via your multimodal model.
        """
        frames = self.vision_buffer.get_recent_frames()
        audio_chunks = self.audio_buffer.get_recent_chunks()
        events = self.user_buffer.get_recent_events()

        lines: List[str] = []

        # Vision summary
        if frames:
            lines.append("Recent visual context:")
            for f in frames[-5:]:
                lines.append(f"- [Vision @ {f.timestamp:.0f}] {f.description}")
        else:
            lines.append("No recent visual context.")

        # Audio summary
        if audio_chunks:
            lines.append("\nRecent ambient audio context:")
            for c in audio_chunks[-5:]:
                if c.transcript:
                    lines.append(f"- [Audio @ {c.timestamp:.0f}] transcript: {c.transcript}")
                elif c.summary:
                    lines.append(f"- [Audio @ {c.timestamp:.0f}] summary: {c.summary}")
                else:
                    lines.append(f"- [Audio @ {c.timestamp:.0f}] (non-speech audio)")
        else:
            lines.append("\nNo recent audio context.")

        # User events
        if events:
            lines.append("\nRecent user interactions:")
            for e in events[-5:]:
                lines.append(f"- [User @ {e.timestamp:.0f}] {e.text}")
        else:
            lines.append("\nNo recent user interactions.")

        return "\n".join(lines)

    def _run_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval_sec)

            context_text = self._build_context_text()

            # The 'Brain' (Omni Model) configuration
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an always-on assistant observing a live session.\n"
                        "Respond concisely. Include vibe markers like [excited] or [thoughtful] in your text."
                    ),
                },
                {"role": "user", "content": context_text},
            ]

            try:
                # 1. Brain Processing
                # In the real implementation, we would also capture the 'hidden_states' here
                reply = self.llm_fn(messages)
                
                # 2. Intelligence Sharing: Parse 'Mental State' from tags
                detected_vibe = self.active_vibe
                if "[" in reply and "]" in reply:
                    import re
                    match = re.search(r"\[([a-zA-Z]+)\]", reply)
                    if match:
                        detected_vibe = match.group(1).lower()
                        reply = reply.replace(f"[{match.group(1)}]", "").strip()

                if self.on_llm_response:
                    self.on_llm_response(reply)
                    
                # 3. Synchronized Voice Synthesis
                if self.tts_engine and reply:
                    # We pass the persona DNA and the detected mental 'vibe' as a single packet
                    self.tts_engine.synthesize(
                        text=reply,
                        voice=self.active_voice,
                        vibe=detected_vibe,
                        sync_mode="high_fidelity" # Signals to use 100% model capabilities
                    )
            except Exception as e:
                print(f"[JointStreaming] LLM error: {e}", file=sys.stderr)

    # Exposed helpers to push data into buffers

    def add_vision_frame(self, description: str, path: Optional[str] = None):
        ts = time.time()
        self.vision_buffer.add_frame(VisionFrame(timestamp=ts, description=description, path=path))

    def add_audio_chunk(
        self,
        transcript: Optional[str] = None,
        summary: Optional[str] = None,
    ):
        ts = time.time()
        self.audio_buffer.add_chunk(AudioChunk(timestamp=ts, transcript=transcript, summary=summary))

    def add_user_event(self, text: str):
        ts = time.time()
        self.user_buffer.add_event(UserEvent(timestamp=ts, text=text))


# ═══════════════════════════════════════════════════════════════
# LIVE OMNI-MODEL ADAPTER
# ═══════════════════════════════════════════════════════════════

def get_live_model(model_path: str):
    # Re-use the factory logic from podcast generator or import directly
    # Ideally should be a shared utility, but duplicating for standalone script stability
    from multimodal.model import OmniMultimodalLM
    from transformers import AutoTokenizer
    
    print(f"⚡ Loading Live OmniModel: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = OmniMultimodalLM(model_path)
    return model, tokenizer

def run_live_inference(messages: List[Dict[str, str]], model, tokenizer) -> str:
    import torch
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.wrapper.llm.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, # Short responses for streaming
            temperature=0.7,
            do_sample=True
        )
        
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ═══════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Tri-Streaming Orchestrator")
    parser.add_argument("--model", type=str, default="/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    vision_buf = VisionStreamBuffer(max_seconds=30.0)
    audio_buf = AudioStreamBuffer(max_seconds=30.0)
    user_buf = UserEventBuffer(max_events=50)

    # Load Model ONCE
    try:
        model, tokenizer = get_live_model(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    def live_llm_adapter(messages: List[Dict[str, str]]) -> str:
        return run_live_inference(messages, model, tokenizer)

    orchestrator = JointStreamingOrchestrator(
        vision_buffer=vision_buf,
        audio_buffer=audio_buf,
        user_buffer=user_buf,
        llm_fn=live_llm_adapter,
        interval_sec=args.interval,
    )

    orchestrator.on_llm_response = lambda r: print(f"\n🤖 [Omni]: {r}\n")

    orchestrator.start()
    print("🚀 Joint Streaming Active. Type to interact (simulates User Event). Ctrl+C to stop.")


    orchestrator.on_llm_response = lambda r: print(f"\n[LLM] {r}\n")

    orchestrator.start()
    print("Joint streaming demo. Type text to simulate user events; Ctrl+C to exit.")

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                break

            # Voice/Vibe Commands
            if text.startswith("/voice "):
                voice_id = text.split(" ")[1]
                orchestrator.switch_voice(voice_id)
                continue
            elif text.startswith("/vibe "):
                vibe_name = text.split(" ")[1]
                orchestrator.set_vibe(vibe_name)
                continue

            # For demo: every user text also updates vision/audio summaries
            orchestrator.add_user_event(text)
            orchestrator.add_vision_frame(description=f"User typed: {text}")
            orchestrator.add_audio_chunk(summary="Ambient game audio present.")

    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()


if __name__ == "__main__":
    main()

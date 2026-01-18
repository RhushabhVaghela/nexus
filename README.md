# Manus Prime

Architecture-agnostic LLM training with Omni-Modal support (Text, Image, Audio, Video).
Now powered by **Unsloth**, **Real-Time Streaming**, and **Advanced Interaction**.

## 🚀 Key Features

- **Base Model**: GPT-OSS-20B
- **Omni-Modal**: SigLIP 2 + Whisper V3 + Perceiver
- **Triple-Modality Streaming** (Gemini-Like):
  - 👁️ **Vision**: Live Camera / Video Feed.
  - 👂 **Ambient Audio**: Environment / Game Audio.
  - 🗣️ **User Interaction**: Voice / Text Commands.
- **Advanced Features**:
  - 🎙️ **Podcast Mode**: NotebookLM-style dialogue generation.
  - ♾️ **Infinite Context**: StreamingVLM memory.

## 📂 File Structure (27 Scripts)

```
src/
├── ...
├── Real-Time Streaming
│   ├── 25_realtime_streaming.py     # Omni-Stream Orchestrator
├── streaming/
│   ├── memory.py, tts.py, vision.py
│   └── joint.py                     # 🚀 NEW: Triple-Modality Stream
├── podcast/                         # 🚀 NEW: Interactive Podcast
│   ├── generator.py                 
│   └── player.py                    
├── multimodal/
└── utils/
```

## ⚡ Quick Start

### 1. Omni-Modal Pipeline

```bash
./run_multimodal_pipeline.sh all
```

### 2. Live Triple-Modality Stream

```bash
python3 src/streaming/joint.py
```

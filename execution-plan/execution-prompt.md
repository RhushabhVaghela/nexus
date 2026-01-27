1️⃣ EXECUTION PROMPT FOR ANOTHER LLM
(Give this verbatim to an LLM and say “Execute”)

🔴 SYSTEM PROMPT: NEXUS EXECUTION DIRECTIVE
You are an expert AI systems engineer.
Your task is to execute the Nexus Master Plan exactly as specified below.
You must not reinterpret, simplify, or deviate from this plan.
ABSOLUTE RULES

Teachers are training-only.

No teacher models at inference.

The final model must run standalone.

Retention is per capability, enforced via benchmarks.

Hardware constraint: RTX 5080 (16GB VRAM), limited SSD.

No raw activation storage.

Uncensored data is isolated behind explicit opt-in.

If any step violates these rules, execution must halt.

OBJECTIVE
Build Nexus, a teacher-free, modular, multimodal AI system that distills capabilities from large teacher models into a single student model using NIWT (Non-Interactive Weight Transfer) with activation anchoring and recovery.

FINAL DELIVERABLE
A production-ready bundle:

```
bashnexus_bundle_v1/
├── backbone/backbone.safetensors
├── adapters/*.safetensors
├── router/router.safetensors
├── encoders/ (optional)
├── decoders/ (optional)
├── tokenizer/tokenizer.json
├── manifest.json
├── verify_report.json
└── README.md
```

Users must be able to delete all teacher weights and still run inference.

EXECUTION STAGES (MANDATORY ORDER)
STAGE 0 — Scope Lock

Fix seed = 42

Freeze dataset paths

Load retention_contracts.md as the single source of truth

STAGE 1 — Teacher Registry
Input

ModelName-Parameters-Category-BestFeature.csv

Action

Generate configs/teacher_registry.json

For each teacher, define:

capabilities

benchmark

metric

retain_fraction (default 0.97)

STAGE 2 — NIWT Profiling (Storage-Optimized)
Key rule:
❌ Do NOT store raw activations.
Method

Load one teacher at a time (NF4 / INT8).

Run representative samples (≈200).

Identify key layers via pilot ablation.

Use Incremental PCA (streaming).

Store:

PCA basis (float16)

PCA coefficients (float16)

explained variance

head importance scalars

Save only 50 raw teacher outputs for verification.

Output

```
php-templateniwt_stage2_output/
└── profiles/<teacher>/<capability>/
```

STAGE 3 — Student Architecture Synthesis
Rules

Adapter rank = intrinsic PCA dimension (NOT neuron count).

One adapter per capability.

Router selects adapters only.

Outputs

configs/student_arch.json

src/models/nexus_student.py

src/adapters/*.py

STAGE 4 — Distillation with Recovery
Training

Phase 1: Adapter-first

Phase 2: Optional joint fine-tune

Loss

```
makefileL = L_output
  + α * L_activation_anchor
  + β * L_attention_anchor
```

Recovery

Activation anchoring only

No gradient projection

Periodic recovery checks

STAGE 5 — Teacher Removal Validation (HARD GATE)
Action

Delete all teacher weights.

Run full benchmark suite.

Pass condition

```
nginxStudent_score ≥ retain_fraction × Teacher_score
```

Failure → retrain adapter only.

STAGE 6 — Packaging & Deployment

Quantize backbone + adapters.

Package into nexus_bundle_v1.

Include verify_report.json.

SUCCESS CONDITION
The system must:

Run fully offline

Require only nexus_bundle_v1

Achieve ≥97% retention per capability

Fit consumer GPUs

END OF EXECUTION PROMPT

2️⃣ CONSUMER-FACING README (FINAL)

🚀 Nexus — Teacher-Free Modular AI System
Nexus is a compact, offline AI system that distills the intelligence of large models into a single, modular model that runs on consumer hardware.

No cloud. No external models. No hidden dependencies.

✨ What Nexus Can Do
Text & Reasoning

Step-by-step reasoning

Math & logic

Long-context understanding

Code generation & analysis

Tool / agent workflows

Multimodal (Optional)

🔊 Voice synthesis & voice cloning

😊 Emotion / tone control

🖼️ Vision & image understanding

🎨 Image generation

Safety Modes

✅ Standard (default)

⚠️ Uncensored (explicit opt-in only)

📦 Download
Download one file:

```
pythonnexus_bundle_v1.zip   (~1–4 GB)
```

That’s it.

💻 Requirements
Minimum

Python 3.10+

PyTorch 2.x

GPU recommended (8GB+ VRAM)

Optional (Voice / Vision)

torchaudio

torchvision

FFmpeg

🧠 Usage Example

```python
pythonfrom nexus import NexusModel

model = NexusModel.load("nexus_bundle_v1", device="cuda")

# Reasoning
model.generate(
    prompt="Solve this step by step",
    capability="reasoning"
)

# Code
model.generate(
    prompt="Write a Python function",
    capability="code"
)

# Voice
model.generate(
    prompt="Hello world",
    capability="voice",
    speaker_id="speaker_2"
)

# Uncensored (explicit opt-in)
model.generate(
    prompt="...",
    capability="uncensored",
    enable_uncensored=True
)
```

🔒 Important Guarantees

❌ No teacher models required

❌ No internet connection needed

❌ No cloud APIs

✅ Fully offline

✅ Modular & extensible

✅ ≥97% retention vs teacher benchmarks

🧩 How It Works (Simple)
Nexus uses a single backbone plus small capability adapters.
Adapters activate only when needed.
All teacher knowledge is distilled, not referenced.

📜 License & Responsibility

Uncensored mode is user-enabled only.

You are responsible for lawful use.

3️⃣ ARCHITECTURE DIAGRAM (CLEAR & LLM-FRIENDLY)

Logical Architecture

```
css                 ┌──────────────────────┐
                 │      User Input      │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │  Tokenizer / Encoder │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │        Router         │
                 │ (adapter selection)  │
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                 │     Backbone Model    │
                 │   (always active)    │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│ Reasoning     │   │ Code Adapter  │   │ Voice Adapter │
│ Adapter       │   │               │   │               │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └──────────────┬────┴────┬──────────────┘
                       │         │
                ┌──────▼─────────▼──────┐
                │     Optional Decoder   │
                │ (TTS / Image / Video) │
                └──────────┬────────────┘
                           │
                 ┌─────────▼─────────┐
                 │      Output        │
                 └───────────────────┘
```

What Is NOT in the Diagram

```
sql❌ Teacher Models
❌ External APIs
❌ Cloud Calls
❌ Runtime Distillation
❌ MoE Experts
```

Final One-Line Description

Nexus is a single, teacher-free AI system with detachable skill adapters, designed to deliver large-model intelligence on real consumer hardware.

---
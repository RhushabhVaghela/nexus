# Nexus Self-Driving Pipeline - Operator's Manual

> **System:** Nexus Unconstrained (Tier 3 Architecture)
> **Pipeline:** Automated Distillation & Training
> **Hardware Target:** Consumer High-End (RTX 5080 Laptop, 16GB VRAM)

---

## 🎯 The Philosophy

Nexus is not just a model; it's a **self-driving distillation factory**.
Instead of running manual scripts like `python train.py`, you maintain a **Registry** of teacher models. The pipeline creates the student model by extracting knowledge from these teachers.

---

## 🚀 3-Step Execution

### 1. Register Teachers

Edit `src/nexus_final/registry.json`.
Only teachers marked `"status": "ready"` will be processed.

```json
[
  {
    "teacher_id": "DeepSeek-V3",
    "params": "671B",
    "path": "/mnt/d/models/deepseek-v3",
    "status": "ready" 
  },
  {
    "teacher_id": "Step3-VL",
    "params": "10B",
    "path": "/path/to/step3",
    "status": "ready"
  }
]
```

### 2. Activate Environment

```bash
conda activate nexus
```

### 3. Launch the Pipeline

```bash
# Standard Run (all models, all stages)
./run_nexus_master.sh

# Target specific teachers and datasets
./run_nexus_master.sh --models "DeepSeek-V3" --datasets "reasoning/AI4Math_IneqMath,code/python-stack"

# High-Scale Dataset Auto-Discovery
./run_nexus_master.sh --datasets all

# Reset and fresh start (clears previous artifacts)
./run_nexus_master.sh --reset
```

---

## 🔄 The Pipeline Stages

* **Resume:** If you interrupt the process (Ctrl+C), simply run `./run_nexus_master.sh` again. It will skip completed stages and resume instantly.
* **Reset:** If you want to force a re-run or clear disk space, use `--reset`. This will delete the saved state file, all previous extraction shards in `memory/`, checkpoints in `checkpoints/`, and the final model in `nexus-release-v1/`.

| Stage | What it does | Magic component |
| :--- | :--- | :--- |
| **1. Profiling** | Analyzes teachers to find "Critical Circuits". | `NIWTCore` |
| **1.5 Extraction** | Extracts knowledge shards to SSD. | `Librarian` (SLI) |
| **2. Training** | Distills knowledge into the Student. | `NexusTrainer` |
| **3. Router** | Trains the intent router. | `train_router.py` |

---

## 🛠️ Advanced Operations

### Handling Massive Models (SLI)

If a teacher has **>60B Parameters** (like DeepSeek or massive encoder-decoders), or if the **VRAM Headroom** analysis indicates insufficient memory, the pipeline **automatically** switches to **SLI Mode** (Sequential Layer Ingestion).

* **Layer Mapping:** Uses `SequentialLayerIntegrator` to stream layers one-by-one.

* **No Config Needed.** The system detects the size.
* **Disk Requirement:** ~20GB free space (it deletes shards as it goes).

### Thermal Protection

The system monitors your GPU temperature.

* **> 83°C**: Auto-Pause for 30 seconds.
* **OOM (Out of Memory)**: Auto-Recovery (clears cache, skips batch).

### Manual Checkpoints

While training is running, you can signal the process:

* `kill -USR1 <pid>`: Pause and Save State.
* `kill -USR2 <pid>`: Save Immediate Checkpoint (don't stop).

---

## 📂 File Structure

* `scripts/nexus_pipeline.py`: The Boss. Runs everything.
* `src/nexus_final/registry.json`: The Roster.
* `src/nexus_final/sli_integrator.py`: The Heavy Lifter (for 1T models).
* `results/`: All outputs (profiles, checkpoints).

---

## ❓ FAQ

**Q: Can I add a new teacher mid-run?**
A: Yes. Pause the pipeline, add to `registry.json`, and restart. The Extraction stage scans for *new* teachers.

**Q: I have a 12GB Laptop, can I run this?**
A: Yes. The SLI system is designed for 16GB but scales down. For 12GB, ensure you don't run browser tabs in background.

**Q: Where is the student model saved?**
A: `checkpoints/checkpoint_latest.pt` (during training) and `results/router_weights/` (final).

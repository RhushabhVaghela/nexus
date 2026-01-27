# Nexus Production Packaging Specification

## 1. Directory Structure: `nexus_bundle_v1/`

This structure is designed to be immutable and self-contained for production deployment.

```
nexus_bundle_v1/
├── manifest.json              # Signed metadata and integrity checksums
├── README.md                  # Quickstart guide
├── LICENSE                    # Licensing information
├── bin/
│   └── nexus_cli              # Main executable/entry point
├── model/
│   ├── student_model.onnx     # Optimized Student Model (NO TEACHER WEIGHTS)
│   └── tokenizer/             # Tokenizer artifacts
│       ├── tokenizer.json
│       └── vocab.txt
├── config/
│   └── runtime.yaml           # default inference settings
└── src/                       # Minimized inference-only source code
    ├── __init__.py
    ├── inference.py           # Inference engine wrapper
    └── utils.py
```

## 2. Teacher Removal Validation (TRV) Protocol

The "Teacher-Free" invariant is enforced via a strictly subtractive packaging process followed by an adversarial verification step.

### A. Packaging Script Logic (`scripts/build_bundle.sh`)
1.  **Stage**: Create a clean temporary directory.
2.  **Export**: Export **only** the student model weights (e.g., convert to ONNX/TensorRT).
3.  **Filter**: Copy inference source code. grep/sed check to remove any imports of `teacher_model` or `training_modules`.
4.  **Assemble**: Move assets into the `nexus_bundle_v1` structure.

### B. Verification Script Logic (`tests/verify_teacher_free.py`)
This script runs in the CI/CD pipeline *after* packaging and *before* deployment.

**Checks:**
1.  **File Allow-list**: Recursively scan `nexus_bundle_v1/`. If any file named `*teacher*`, `*checkpoint*` (raw training ckpt), or `*optimizer*` exists -> **FAIL**.
2.  **Code Inspection**: Scan all Python/Config files for string literals "teacher", "distillation", "soft_targets". If found -> **FAIL** (or warn if benign).
3.  **Weight Inspection**: Load `student_model.onnx`. Inspect metadata. If metadata refers to "Teacher" -> **FAIL**.
4.  **Size Heuristic**: If `student_model.onnx` size > defined threshold (implying accidental inclusion of larger model) -> **WARN/FAIL**.
5.  **Functional Test**: Run inference in an isolated environment where *only* the bundle exists. If it tries to import missing training libraries -> **FAIL**.

## 3. Manifest Schema (`manifest.json`)

The manifest acts as the bill of materials and integrity check.

```json
{
  "schema_version": "1.0",
  "bundle_id": "nexus_bundle_v1",
  "build_timestamp": "ISO8601_DATE",
  "model_metadata": {
    "architecture": "StudentArchitectureName",
    "parameters": "1.2B",
    "format": "ONNX",
    "quantization": "int8"
  },
  "invariants": {
    "teacher_free": true,
    "training_code_excluded": true
  },
  "files": [
    {
      "path": "model/student_model.onnx",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "size_bytes": 12345678
    },
    {
      "path": "config/runtime.yaml",
      "sha256": "...",
      "size_bytes": 1024
    }
  ],
  "signature": "..."
}
```

"""
Nexus Models — model loading, quantization, diffusion, video, and SLI subsystems.

Sub-packages:
    - sli: Selective Layer Inference (caching, quantization, I/O optimization)
    - omni: Multi-modal OmniModel loading and inference
    - diffusion: Image generation pipelines
    - video: Video generation pipelines
    - gguf: GGUF model loading/conversion
    - tensorrt: TensorRT acceleration
    - utils: Memory estimation utilities
"""

__version__ = "1.0.0"

# Lazy imports to avoid heavy torch/CUDA loading at import time
_LAZY_IMPORTS = {
    # SLI
    "UniversalSLIIntegrator": ".sli",
    "AdvancedSLIIntegrator": ".sli",
    "ArchitectureRegistry": ".sli",
    # Omni
    "OmniModelLoader": ".omni",
    # Diffusion
    "ImagePipeline": ".diffusion",
    "DiffusionPipelineLoader": ".diffusion",
    # Video
    "VideoPipeline": ".video",
    # GGUF
    "GGUFLoader": ".gguf",
    "GGUFConverter": ".gguf",
    # TensorRT
    "TRTEngine": ".tensorrt",
    "TensorRTBackend": ".tensorrt",
    # --- Library modules (previously orphaned) ---
    # auditor.py — memorization auditing
    "DistillationReport": ".auditor",
    "MemorizationClassifier": ".auditor",
    "MemorizationAuditor": ".auditor",
    "create_auditor": ".auditor",
    # config.py — teacher/model configs
    "TeacherConfig": ".config",
    "NexusConfig": ".config",
    # data_loader.py — data loading + memorization filtering
    "MemorizationFilter": ".data_loader",
    "UniversalDataLoader": ".data_loader",
    # decoders.py — video/multimodal decoders
    "VideoDecoder": ".decoders",
    "NexusDecoders": ".decoders",
    # loss_functions.py — custom loss functions
    "ActivationAnchoringLoss": ".loss_functions",
    "RecoveryStepLoss": ".loss_functions",
    # memory_projector.py — activation-guided projection
    "ActivationGuidedProjector": ".memory_projector",
    # optimization_suite.py — training optimization monitors
    "ThermalWatchdog": ".optimization_suite",
    "GradNormMonitor": ".optimization_suite",
    "SynergyMonitor": ".optimization_suite",
    "compute_optimal_batch_size": ".optimization_suite",
    # speculative_decoding.py — speculative decoding
    "SpeculativeDecoder": ".speculative_decoding",
    # architect.py — neural architecture search
    "NeuralArchitect": ".architect",
    "NexusBridge": ".architect",
    "NexusStudent": ".architect",
    "build_student": ".architect",
    # registry.py — teacher model registry
    "TeacherRegistry": ".registry",
    # export.py — model export
    "NexusExporter": ".export",
    # export_model.py — multi-format model export
    "ModelExporter": ".export_model",
    "export_all_formats": ".export_model",
    # alignment.py — cross-modal alignment
    "CrossModalAlignment": ".alignment",
    # distill.py — knowledge distillation trainer
    "NexusTrainer": ".distill",
    # knowledge.py — knowledge tower + memory
    "FileMemoryManager": ".knowledge",
    "KnowledgeTower": ".knowledge",
    # profiler.py — streaming PCA profiling
    "StreamingPCAProfiler": ".profiler",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())

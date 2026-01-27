# Conceptual Design: StreamingDataLoader for NIWT Profiling

## 1. Overview
The `StreamingDataLoader` is a critical component for the "Storage-Optimized" stage of NIWT. Its primary goal is to feed model inputs to the inference engine batch-by-batch, allowing activations to be computed, fed into `IncrementalPCA`, and immediately discarded. It strictly enforces capability mapping and safety protocols for uncensored data.

## 2. Architecture

### Core Class: `StreamingDataLoader`
The loader operates as a generator, yielding batches of data while maintaining a small buffer for teacher calibration.

```python
class StreamingDataLoader:
    def __init__(self, 
                 model_name: str, 
                 capabilities_csv_path: str, 
                 benchmarks_root: str,
                 batch_size: int = 32):
        self.model_metadata = self._load_metadata(model_name, capabilities_csv_path)
        self.root = benchmarks_root
        self.batch_size = batch_size
        self.teacher_buffer = [] # Stores approx 200 representative samples
        
    def stream(self, dataset_type: str = "auto", allow_uncensored: bool = False):
        """
        Main generator method.
        Yields: Batch of inputs (Tensor or Dict)
        """
        adapter = self._select_adapter(dataset_type, allow_uncensored)
        
        # Reservoir sampling logic for teacher_buffer
        count = 0
        reservoir_target = 200
        
        for batch in adapter.load_batches(self.batch_size):
            # Update teacher samples (Reservoir Sampling or First-N)
            self._update_teacher_buffer(batch, count, reservoir_target)
            count += len(batch)
            
            yield batch
            
            # Explicit cleanup hint (though Python GC handles ref counts)
            del batch 
```

## 3. Data Loading Strategy

### A. Capabilities Mapping
The loader maps the `Category` from `ModelName-Parameters-Category-BestFeature.csv` to specific adapters and data paths found in `benchmarks-structure.txt`.

| Model Category | Adapter Type | Target Benchmark Paths |
| :--- | :--- | :--- |
| **Language model** | `TextAdapter` | `./benchmarks/general/cais_mmlu`<br>`./benchmarks/general/google_xtreme` |
| **Agent (LLM-based)** | `CodeAdapter` | `./benchmarks/code/MiniMaxAI_OctoCodingBench`<br>`./benchmarks/code/princeton-nlp_SWE-bench` |
| **Vision-language** | `MultimodalAdapter` | `./benchmarks/code/MiniMaxAI_VIBE` |
| **Audio (ASR/TTS)** | `AudioAdapter` | *(Requires mapping to specific audio datasets, not currently in structure list)* |
| **Image/Video Gen** | `VisualGenAdapter` | *(Requires specific visual prompts dataset)* |

### B. Streaming Mechanism
*   **Format Handling:**
    *   **Parquet:** Uses `pyarrow.parquet.ParquetFile` with `iter_batches()` to read chunks without loading the full table.
    *   **JSONL:** Standard line-by-line reading generator.
    *   **HuggingFace Cache:** If detecting `.cache/huggingface`, attempts to load via `datasets.load_dataset(..., streaming=True)`.

### C. Teacher Sample Extraction (Representative 200)
To ensure the "Teacher" model (used for KD or comparison) gets a representative view without storing the whole dataset:
*   **Strategy:** Reservoir Sampling.
*   **Implementation:**
    1.  Initialize `buffer = []`.
    2.  For the first 200 items, append directly.
    3.  For item `i > 200`, replace an element in `buffer` with probability `200/i`.
*   **Output:** The `teacher_buffer` is accessible as a property `loader.representative_samples` at any time, stabilizing as the stream progresses.

## 4. Uncensored Data Isolation
This requirement mandates strict separation of sensitive/uncensored data.

*   **Mechanism:** Explicit Opt-in Flag.
*   **Implementation:**
    *   The `stream()` method requires `allow_uncensored=True`.
    *   **Secure Adapter:** A specialized `UncensoredAdapter` is registered.
    *   **Path Isolation:** Uncensored datasets are assumed to reside in a specific isolated path (e.g., `./benchmarks/restricted/` or flagged in metadata).
    *   **Guardrails:**
        ```python
        if adapter.is_restricted and not allow_uncensored:
            raise SecurityError("Attempted to access restricted adapter without explicit opt-in.")
        ```

## 5. Integration with IncrementalPCA
The workflow for NIWT profiling using this loader:

1.  **Initialize:** `loader = StreamingDataLoader(...)`
2.  **Loop:**
    ```python
    pca = IncrementalPCA(n_components=...)
    
    for batch_inputs in loader.stream():
        # 1. Forward Pass (No Grad)
        activations = model(batch_inputs) 
        
        # 2. Update PCA
        pca.partial_fit(activations)
        
        # 3. Discard
        del activations # Immediate memory release
    
    # 4. Teacher Verification (Optional)
    teacher_inputs = loader.get_representative_samples()
    # ... run verification ...
    ```

## 6. Implementation Notes relative to File Structure
*   **MMLU Handling:** The structure shows deeply nested parquet files (`.cache/...`). The `TextAdapter` must recursively crawl `./benchmarks/general/cais_mmlu` to find `.parquet` files if standard HF loading fails.
*   **VIBE Handling:** Located at `./benchmarks/code/MiniMaxAI_VIBE/vibe_queries.parquet`. The `MultimodalAdapter` must handle image paths referenced in the parquet/jsonl if applicable, or text-only queries.

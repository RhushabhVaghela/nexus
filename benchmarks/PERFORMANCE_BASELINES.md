# Performance Baselines Documentation

This document defines the performance baselines for all benchmark suites in the Nexus project.

## Overview

The benchmark suite measures performance across 4 key implementations:

1. **Multimodal Architecture** - Embedding injection and fusion performance
2. **Video Decoder** - Video generation and export performance
3. **TTS Engine** - Text-to-speech synthesis performance
4. **Multi-Agent Orchestration** - Multi-agent system performance

## Running Benchmarks

### All Benchmarks

```bash
# Run all benchmarks
python benchmarks/test_multimodal_architect_benchmark.py --all
python benchmarks/test_video_decoder_benchmark.py --all
python benchmarks/test_tts_benchmark.py --all
python benchmarks/test_multi_agent_benchmark.py --all
```

### Specific Categories

```bash
# Run specific categories
python benchmarks/test_multimodal_architect_benchmark.py --category projection
python benchmarks/test_video_decoder_benchmark.py --category generation
python benchmarks/test_tts_benchmark.py --category latency
python benchmarks/test_multi_agent_benchmark.py --category initialization
```

### Environment Variables

```bash
export BENCHMARK_ITERATIONS=100  # Number of iterations per benchmark
export BENCHMARK_WARMUP=10       # Number of warmup iterations
export ENABLE_GPU_BENCHMARKS=1   # Enable GPU benchmarks (if available)
```

---

## 1. Multimodal Architecture Benchmarks

**File:** [`benchmarks/test_multimodal_architect_benchmark.py`](test_multimodal_architect_benchmark.py)

### Baseline Targets

#### Embedding Projection Latency

| Dimension | Target Mean (ms) | Max Acceptable (ms) |
|-----------|------------------|---------------------|
| 512→768   | 0.5              | 2.0                 |
| 768→1024  | 0.8              | 3.0                 |
| 1024→2048 | 1.5              | 5.0                 |
| 2048→4096 | 3.0              | 10.0                |

#### Multimodal Fusion Throughput

| Hidden Dim | Target FPS | Min FPS |
|------------|------------|---------|
| 512        | 1000       | 500     |
| 768        | 800        | 400     |
| 1024       | 600        | 300     |
| 2048       | 300        | 150     |

#### Attention Mask Computation

| Sequence Length | Target (ms) | Max (ms) |
|-----------------|-------------|----------|
| 128             | 0.1         | 0.5      |
| 512             | 0.3         | 1.0      |
| 1024            | 0.6         | 2.0      |
| 4096            | 2.5         | 8.0      |

#### Memory Usage During Injection

| Configuration | Target (MB) | Max (MB) |
|---------------|-------------|----------|
| Small         | 50          | 100      |
| Medium        | 200         | 400      |
| Large         | 500         | 1000     |

### Regression Thresholds

- **Critical (>20% slower):** Immediate investigation required
- **Warning (10-20% slower):** Monitor closely
- **Acceptable (<10%):** Normal variance

---

## 2. Video Decoder Benchmarks

**File:** [`benchmarks/test_video_decoder_benchmark.py`](test_video_decoder_benchmark.py)

### Baseline Targets

#### Video Generation Time

| Resolution | Frames | Target (s) | Max (s) |
|------------|--------|------------|---------|
| 256x256    | 16     | 2.0        | 5.0     |
| 512x512    | 16     | 8.0        | 20.0    |
| 1024x1024  | 16     | 30.0       | 60.0    |

#### Frame Generation Rate (FPS)

| Resolution | Target FPS | Min FPS |
|------------|------------|---------|
| 256x256    | 8          | 4       |
| 512x512    | 2          | 1       |
| 1024x1024  | 0.5        | 0.25    |

#### Memory Usage Peak

| Resolution | Frames | Target (GB) | Max (GB) |
|------------|--------|-------------|----------|
| 256x256    | 16     | 2           | 4        |
| 512x512    | 16     | 6           | 12       |
| 1024x1024  | 16     | 16          | 32       |

#### Export Format Performance (per second of video)

| Format   | Target (ms) | Max (ms) |
|----------|-------------|----------|
| MP4 H264 | 10          | 50       |
| MP4 HEVC | 20          | 100      |
| WebM VP9 | 30          | 150      |
| GIF      | 50          | 250      |

#### VAE Optimization

| Resolution | Without Optimization | With Slicing | With Tiling |
|------------|---------------------|--------------|-------------|
| 256x256    | 100%                | 110%         | 105%        |
| 512x512    | 100%                | 95%          | 90%         |
| 1024x1024  | 100%                | 85%          | 80%         |

### Regression Thresholds

- **Critical (>30% slower):** Immediate investigation
- **Warning (15-30% slower):** Monitor closely
- **Acceptable (<15%):** Normal variance

---

## 3. TTS Engine Benchmarks

**File:** [`benchmarks/test_tts_benchmark.py`](test_tts_benchmark.py)

### Baseline Targets

#### Synthesis Latency

| Text Length | Characters | Target (ms) | Max (ms) |
|-------------|------------|-------------|----------|
| Short       | < 50       | 50          | 200      |
| Medium      | ~200       | 200         | 800      |
| Long        | ~1000      | 800         | 3000     |

#### Real-Time Factor (RTF)

| Configuration | Target RTF | Max RTF |
|---------------|------------|---------|
| Streaming     | 0.1x       | 0.3x    |
| Batch         | 0.05x      | 0.15x   |
| Voice Cloning | 0.2x       | 0.5x    |

#### Voice Cloning Setup

| Reference Audio | Target (ms) | Max (ms) |
|-----------------|-------------|----------|
| 3 seconds       | 100         | 500      |
| 10 seconds      | 300         | 1000     |
| 30 seconds      | 800         | 3000     |

#### Cache Performance

| Cache Type | Hit Latency (ms) | Miss Latency (ms) | Speedup |
|------------|------------------|-------------------|---------|
| Memory     | 0.01             | 100               | 10000x  |
| Disk       | 1.0              | 100               | 100x    |

#### Streaming Throughput

| Chunk Size (phonemes) | Target (chunks/sec) | Min (chunks/sec) |
|-----------------------|---------------------|------------------|
| 1                     | 50                  | 20               |
| 5                     | 20                  | 8                |
| 10                    | 10                  | 4                |
| 20                    | 5                   | 2                |

#### Language Comparison (relative to English)

| Language | Target Multiplier | Max Multiplier |
|----------|-------------------|----------------|
| English  | 1.0x              | 1.0x           |
| Chinese  | 1.1x              | 1.5x           |
| Japanese | 1.2x              | 1.6x           |
| Spanish  | 1.0x              | 1.3x           |
| German   | 1.0x              | 1.3x           |
| French   | 1.0x              | 1.3x           |

#### Audio Format Conversion

| Format      | Target (ms/sec) | Max (ms/sec) |
|-------------|-----------------|--------------|
| WAV PCM16   | 1               | 5            |
| WAV PCM24   | 1               | 5            |
| WAV FLOAT   | 2               | 10           |
| MP3 128k    | 5               | 25           |
| MP3 192k    | 8               | 40           |
| MP3 320k    | 15              | 75           |
| OGG Vorbis  | 10              | 50           |
| FLAC        | 3               | 15           |

### Regression Thresholds

- **Critical (>25% slower):** Immediate investigation
- **Warning (10-25% slower):** Monitor closely
- **Acceptable (<10%):** Normal variance

---

## 4. Multi-Agent Orchestration Benchmarks

**File:** [`benchmarks/test_multi_agent_benchmark.py`](test_multi_agent_benchmark.py)

### Baseline Targets

#### Agent Initialization Time

| Agent Type | Target (ms) | Max (ms) |
|------------|-------------|----------|
| Planning   | 50          | 200      |
| Backend    | 30          | 120      |
| Frontend   | 30          | 120      |
| Review     | 20          | 80       |

#### Multi-Agent Initialization Scaling

| Number of Agents | Target (ms) | Max (ms) |
|------------------|-------------|----------|
| 5                | 100         | 400      |
| 10               | 200         | 800      |
| 20               | 400         | 1600     |
| 50               | 1000        | 4000     |

#### Planning Agent Response

| Complexity | Target (ms) | Max (ms) |
|------------|-------------|----------|
| Simple     | 100         | 400      |
| Medium     | 500         | 2000     |
| Complex    | 1500        | 6000     |

#### Code Generation Latency

| Task Size | Backend (ms) | Frontend (ms) |
|-----------|--------------|---------------|
| Small     | 100          | 100           |
| Medium    | 500          | 500           |
| Large     | 2000         | 2000          |

#### Full Workflow Execution

| Workflow Type | Steps | Target (ms) | Max (ms) |
|---------------|-------|-------------|----------|
| Simple        | 2     | 150         | 600      |
| Standard      | 3     | 400         | 1600     |
| Complex       | 6     | 1000        | 4000     |

#### Concurrent Execution

| Concurrent Agents | Target (ms) | Throughput (ops/sec) |
|-------------------|-------------|----------------------|
| 2                 | 30          | 66                   |
| 4                 | 35          | 114                  |
| 8                 | 45          | 178                  |
| 16                | 65          | 246                  |

#### Context Passing Overhead

| Context Size | Target (ms) | Max (ms) |
|--------------|-------------|----------|
| Small        | 0.01        | 0.1      |
| Medium       | 0.1         | 1.0      |
| Large        | 1.0         | 10.0     |

#### Context Buildup (per handoff)

| Handoffs | Target (ms) | Max (ms) |
|----------|-------------|----------|
| 2        | 20          | 80       |
| 5        | 50          | 200      |
| 10       | 100         | 400      |
| 20       | 200         | 800      |

#### Retry Mechanism Overhead

| Scenario        | Target (ms) | Max (ms) |
|-----------------|-------------|----------|
| No Retry        | 20          | 80       |
| Success First   | 20          | 80       |
| Retry Once      | 35          | 140      |
| Retry Twice     | 50          | 200      |
| Max Retries     | 65          | 260      |

#### Memory Usage

| Metric                | Target (MB/agent) | Max (MB/agent) |
|-----------------------|-------------------|----------------|
| Agent Overhead        | 1                 | 5              |
| Workflow (2 steps)    | 10                | 50             |
| Workflow (5 steps)    | 25                | 125            |
| Workflow (10 steps)   | 50                | 250            |

### Regression Thresholds

- **Critical (>20% slower):** Immediate investigation
- **Warning (10-20% slower):** Monitor closely
- **Acceptable (<10%):** Normal variance

---

## CI/CD Integration

### Running Benchmarks in CI

```yaml
# Example GitHub Actions workflow
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install numpy
      
      - name: Run Multimodal Benchmarks
        run: |
          python benchmarks/test_multimodal_architect_benchmark.py \
            --category regression \
            --output results/multimodal.json
      
      - name: Run Video Decoder Benchmarks
        run: |
          python benchmarks/test_video_decoder_benchmark.py \
            --category regression \
            --output results/video.json
      
      - name: Run TTS Benchmarks
        run: |
          python benchmarks/test_tts_benchmark.py \
            --category regression \
            --output results/tts.json
      
      - name: Run Multi-Agent Benchmarks
        run: |
          python benchmarks/test_multi_agent_benchmark.py \
            --category regression \
            --output results/multi_agent.json
      
      - name: Compare with Baseline
        run: |
          # Compare results against baselines
          python benchmarks/compare_baselines.py results/
```

### Regression Detection

The benchmark suite supports automatic regression detection:

```bash
# Compare current results against baseline
python benchmarks/test_multimodal_architect_benchmark.py \
  --baseline baselines/multimodal_baseline.json \
  --output results/current.json
```

Output indicators:

- 🟢 **Green:** Performance improved or within 5% of baseline
- 🟡 **Yellow:** Performance degraded 5-10%
- 🔴 **Red:** Performance degraded >10%

---

## Updating Baselines

To update baseline values after legitimate performance improvements:

1. Run full benchmark suite
2. Review results for consistency
3. Update this document with new targets
4. Commit updated baseline JSON files

```bash
# Generate new baselines
python benchmarks/test_multimodal_architect_benchmark.py --all \
  --output baselines/multimodal_baseline.json
python benchmarks/test_video_decoder_benchmark.py --all \
  --output baselines/video_baseline.json
python benchmarks/test_tts_benchmark.py --all \
  --output baselines/tts_baseline.json
python benchmarks/test_multi_agent_benchmark.py --all \
  --output baselines/multi_agent_baseline.json
```

---

## Performance Optimization Tips

### Multimodal Architecture

- Use mixed precision (FP16/BF16) for 2x speedup
- Enable gradient checkpointing for large models
- Batch embedding projections when possible

### Video Decoder

- Enable VAE tiling for high resolutions
- Use lower precision for inference
- Batch frame generation when memory allows

### TTS Engine

- Use streaming mode for real-time applications
- Enable caching for repeated phrases
- Batch phoneme processing

### Multi-Agent Orchestration

- Initialize agents concurrently
- Use connection pooling for external APIs
- Implement intelligent retry backoff

---

## Troubleshooting

### High Variance in Results

- Increase number of iterations
- Ensure system is idle during benchmarks
- Disable CPU frequency scaling
- Use dedicated benchmarking hardware

### Out of Memory Errors

- Reduce batch sizes
- Enable gradient checkpointing
- Use memory profiling to identify leaks
- Close unnecessary applications

### Slow Benchmarks

- Check for thermal throttling
- Verify GPU is being used (if applicable)
- Disable unnecessary logging
- Use SSD for disk cache benchmarks

---

## Contact

For questions about benchmarks or to report performance issues:

- Create an issue in the project repository
- Tag with `performance` and `benchmark` labels
- Include benchmark output and system specifications

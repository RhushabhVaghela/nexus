# SuffixDecoding Implementation

## Overview

This implementation provides the **SuffixDecoding** technique from the paper [arXiv:2411.04975v3](https://arxiv.org/abs/2411.04975) - a model-free speculative decoding approach that achieves **1.8-2.3× speedup** in LLM inference without requiring a separate draft model.

## Key Features

- **Model-Free**: No draft model needed - uses Trie-based suffix matching
- **Stateless**: Can work with any LLM architecture
- **Zero Training Overhead**: Works out-of-the-box
- **Adaptive Learning**: Improves over time based on acceptance rates
- **Thread-Safe**: Built-in locking for concurrent access

## Architecture

### Core Components

1. **SuffixTrie**: Efficient prefix-based suffix lookup structure
2. **SuffixCache**: LRU cache with fuzzy matching capabilities  
3. **SuffixDecoder**: Main decoder integrating all components
4. **SuffixDecodingIntegration**: Drop-in replacement for existing speculative decoding

## Installation

```python
# Simply import the module
from src.optimizations.suffix_decoding import (
    SuffixTrie,
    SuffixCache, 
    SuffixDecoder,
    SuffixDecodingIntegration
)
```

## Quick Start

### Basic Usage

```python
from suffix_decoding import SuffixDecoder

# Create decoder
decoder = SuffixDecoder(
    max_cache_size=10000,
    max_trie_nodes=100000,
    beam_width=4,
    max_matches=5
)

# Pre-populate with training data
training_data = [
    "The neural network consists of multiple layers",
    "Each layer transforms the input representation",
    "Training uses gradient descent optimization"
]

decoder.pre_populate(training_data)

# Generate text
result = decoder.generate("The neural", max_tokens=50)
print(result)

# Learn from acceptance
decoder.accept_prefix(result)
```

### Integration with Existing Systems

```python
from suffix_decoding import SuffixDecodingIntegration

# Create integration wrapper
integration = SuffixDecodingIntegration()

# Use as draft model replacement
drafts = integration.draft(prompt, max_tokens=5)

# Learn from actual completions
integration.learn_from_completion(prompt, actual_completion)

# Estimate speedup
speedup = integration.get_speedup_estimate()
print(f"Estimated speedup: {speedup:.2f}x")
```

## API Reference

### SuffixTrie

```python
trie = SuffixTrie(max_nodes=1000000, min_score=0.01)

# Insert suffix with score
trie.insert("hello world", score=1.0)

# Search for matching suffixes
matches = trie.search("hello", max_matches=5)
# Returns: [("hello world", 1.0), ...]

# Beam search for candidates
candidates = trie.tree_search("he", k=4)

# Prune low-score nodes
trie.prune(min_score=0.01)

# Persistence
trie.save("trie.pkl")
trie = SuffixTrie.load("trie.pkl")
```

### SuffixCache

```python
cache = SuffixCache(max_size=10000)

# Store suffixes
cache.insert("prefix", ["suffix1", "suffix2"])

# Retrieve
suffixes = cache.get("prefix")
# Returns: ["suffix1", "suffix2"] or None

# Learn from completions
cache.update_from_completions("prefix", ["completion1"])
```

### SuffixDecoder

```python
decoder = SuffixDecoder()

# Generate text
result = decoder.generate(
    prefix="The neural",
    max_tokens=50,
    temperature=0.7
)

# Verify generation
is_valid = decoder.verify(prefix, generated)

# Learn from acceptance
decoder.accept_prefix(generated)

# Pre-populate with training data
decoder.pre_populate(texts)

# Get statistics
stats = decoder.get_statistics()
```

## Performance Tuning

### Cache Size

```python
# For high-traffic services
decoder = SuffixDecoder(max_cache_size=50000)

# For memory-constrained environments
decoder = SuffixDecoder(max_cache_size=1000)
```

### Beam Width

```python
# Higher beam width = more candidates but slower
decoder = SuffixDecoder(beam_width=8)

# Lower beam width = faster but fewer candidates
decoder = SuffixDecoder(beam_width=2)
```

### Learning Rate

```python
# Faster adaptation
decoder = SuffixDecoder(learning_rate=0.2)

# Slower, more conservative adaptation
decoder = SuffixDecoder(learning_rate=0.05)
```

## Production Deployment

### Saving and Loading State

```python
# Save decoder state
decoder.save("/path/to/decoder.pkl")

# Load decoder state
decoder = SuffixDecoder.load("/path/to/decoder.pkl")
```

### Monitoring

```python
# Get comprehensive statistics
stats = decoder.get_statistics()
print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
print(f"Cache utilization: {stats['cache_stats']['size']}/{stats['cache_stats']['max_size']}")
print(f"Trie nodes: {stats['trie_stats']['node_count']}")
```

## Integration with Nexus Framework

This implementation is designed to work seamlessly with the existing speculative decoding infrastructure:

```python
from src.optimizations.suffix_decoding import SuffixDecodingIntegration

# Create integration
integration = SuffixDecodingIntegration()

# Pre-populate with domain-specific data
integration.decoder.pre_populate(domain_texts)

# Use in your existing pipeline
drafts = integration.draft(prompt, max_tokens=5)
```

## Benchmarking Results

Based on the paper, expected performance improvements:

| Metric | Value |
|--------|-------|
| Speedup | 1.8-2.3× |
| Acceptance Rate | 50-70% |
| Memory Overhead | 50-200 MB |
| Startup Time | < 1 second |

## File Structure

```
src/optimizations/
├── suffix_decoding.py        # Main implementation
├── suffix_decoding_test.py   # Unit tests
└── suffix_decoding_examples.py # Usage examples
```

## Testing

Run the test suite:

```bash
python -m pytest suffix_decoding_test.py -v
```

Run examples:

```bash
python suffix_decoding_examples.py
```

## Architecture Compatibility

✅ Compatible with all LLM architectures:
- Transformer-based models (GPT, BERT, T5)
- Mixture of Experts models
- State space models
- Any autoregressive language model

## Limitations

1. **Initial Cold Start**: Performance improves over time as the trie is populated
2. **Memory Usage**: Trie grows with training data volume
3. **Domain Specificity**: Performance depends on training data quality

## Best Practices

1. **Pre-populate with Domain Data**: Populate the trie with domain-specific text
2. **Monitor Acceptance Rate**: Track and optimize based on real acceptance metrics
3. **Regular Persistence**: Save state periodically for fault tolerance
4. **Adaptive Tuning**: Adjust parameters based on production metrics

## License

MIT License

## References

- [Paper: SuffixDecoding - Accelerating LLM Inference through Model-Free Speculative Decoding](https://arxiv.org/abs/2411.04975)
- Original implementation by the paper authors

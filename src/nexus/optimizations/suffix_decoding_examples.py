"""
Usage examples for SuffixDecoding.

This module demonstrates how to use the SuffixDecoding implementation
for accelerating LLM inference without requiring a draft model.
"""

import logging
from suffix_decoding import (
    SuffixTrie,
    SuffixCache,
    SuffixDecoder,
    SuffixDecodingIntegration,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_trie():
    """Example 1: Basic Trie operations."""
    logger.info("=" * 50)
    logger.info("Example 1: Basic Trie Operations")
    logger.info("=" * 50)

    # Create a trie
    trie = SuffixTrie(max_nodes=1000, min_score=0.01)

    # Insert suffixes
    suffixes = [
        ("hello world", 1.0),
        ("machine learning", 0.9),
        ("deep neural networks", 0.85),
        ("natural language processing", 0.8),
    ]

    for suffix, score in suffixes:
        trie.insert(suffix, score)
        logger.info(f"Inserted: '{suffix}' with score {score}")

    # Search for matches
    logger.info("\nSearching for 'deep':")
    matches = trie.search("deep", max_matches=5)
    for suffix, score in matches:
        logger.info(f"  Found: '{suffix}' (score: {score})")

    # Get statistics
    stats = trie.get_statistics()
    logger.info(f"\nTrie statistics: {stats}")


def example_beam_search():
    """Example 2: Beam search for candidate generation."""
    logger.info("\n" + "=" * 50)
    logger.info("Example 2: Beam Search")
    logger.info("=" * 50)

    trie = SuffixTrie()

    # Pre-populate with common English phrases
    phrases = [
        "the quick brown fox",
        "the lazy dog",
        "the weather is nice",
        "the stock market",
        "the economy is growing",
        "the government announced",
        "the company reported",
        "the results show",
        "the analysis indicates",
        "the data suggests",
    ]

    for phrase in phrases:
        trie.insert(phrase, score=1.0)

    # Use beam search to generate candidates
    prefix = "the"
    candidates = trie.tree_search(prefix, k=4)

    logger.info(f"Beam search candidates for '{prefix}':")
    for i, candidate in enumerate(candidates, 1):
        logger.info(f"  {i}. '{candidate}'")

    logger.info(f"Selected candidate: '{prefix + candidates[0]}'")


def example_cache_operations():
    """Example 3: LRU Cache with suffix learning."""
    logger.info("\n" + "=" * 50)
    logger.info("Example 3: Cache Operations")
    logger.info("=" * 50)

    cache = SuffixCache(max_size=100)

    # Store suffixes for common prefixes
    prefixes_and_suffixes = {
        "hello": ["world", "there", "everyone"],
        "machine": ["learning", "vision", "translation"],
        "deep": ["learning", "neural networks", "understanding"],
    }

    for prefix, suffixes in prefixes_and_suffixes.items():
        cache.insert(prefix, suffixes)
        logger.info(f"Stored {len(suffixes)} suffixes for prefix '{prefix}'")

    # Retrieve
    logger.info("\nRetrieving suffixes:")
    result = cache.get("hello")
    logger.info(f"  hello -> {result}")

    # Learn from completions
    logger.info("\nLearning from completions:")
    cache.update_from_completions("new prefix", ["new suffix 1", "new suffix 2"])

    stats = cache.get_statistics()
    logger.info(f"Cache statistics: {stats}")


def example_full_decoder():
    """Example 4: Full SuffixDecoder workflow."""
    logger.info("\n" + "=" * 50)
    logger.info("Example 4: Full Decoder Workflow")
    logger.info("=" * 50)

    # Create decoder with custom settings
    decoder = SuffixDecoder(
        max_cache_size=1000, max_trie_nodes=100000, beam_width=4, max_matches=5
    )

    # Pre-populate with training data
    training_data = [
        "The neural network architecture consists of multiple layers",
        "Each layer performs transformations on the input data",
        "The training process uses gradient descent optimization",
        "Backpropagation computes gradients efficiently",
        "Regularization techniques prevent overfitting",
        "Batch normalization stabilizes the learning process",
    ]

    logger.info(f"Pre-populating with {len(training_data)} training samples...")
    decoder.pre_populate(training_data)

    # Generate text
    prefixes = ["The neural", "The training", "Backpropagation computes"]

    for prefix in prefixes:
        result = decoder.generate(prefix, max_tokens=10)
        logger.info(f"  Input: '{prefix}' -> Output: '{result}'")

        # Simulate acceptance
        decoder.accept_prefix(result)

    # Get statistics
    stats = decoder.get_statistics()
    logger.info(f"\nDecoder statistics:")
    logger.info(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    logger.info(f"  Total generated: {stats['total_generated']}")
    logger.info(f"  Trie nodes: {stats['trie_stats']['node_count']}")

    # Save and load
    logger.info("\nSaving decoder state...")
    decoder.save("/tmp/suffix_decoder.pkl")

    logger.info("Loading decoder state...")
    loaded_decoder = SuffixDecoder.load("/tmp/suffix_decoder.pkl")

    logger.info("State loaded successfully!")


def example_integration():
    """Example 5: Integration with existing speculative decoding."""
    logger.info("\n" + "=" * 50)
    logger.info("Example 5: Integration with Existing Systems")
    logger.info("=" * 50)

    # Create integration wrapper
    integration = SuffixDecodingIntegration()

    # Pre-populate
    integration.decoder.pre_populate(
        ["Once upon a time", "In a far away land", "Long long ago", "Once upon a dream"]
    )

    # Use as draft model
    prompt = "Once"
    drafts = integration.draft(prompt, max_tokens=5)

    logger.info(f"Drafts for '{prompt}':")
    for i, draft in enumerate(drafts, 1):
        logger.info(f"  {i}. '{draft}'")

    # Learn from actual completion
    actual_completion = "Once upon a midnight dreary"
    integration.learn_from_completion(prompt, actual_completion)
    logger.info(f"Learned from completion: '{actual_completion}'")

    # Estimate speedup
    speedup = integration.get_speedup_estimate()
    logger.info(f"Estimated speedup: {speedup:.2f}x")

    stats = integration.get_statistics()
    logger.info(f"Integration statistics: {stats}")


def example_production_workflow():
    """Example 6: Production-ready workflow."""
    logger.info("\n" + "=" * 50)
    logger.info("Example 6: Production Workflow")
    logger.info("=" * 50)

    # Initialize decoder with production settings
    decoder = SuffixDecoder(
        max_cache_size=50000,
        max_trie_nodes=500000,
        beam_width=4,
        max_matches=5,
        learning_rate=0.1,
    )

    # Load from pre-trained data
    try:
        decoder.load("/tmp/suffix_decoder.pkl")
        logger.info("Loaded pre-trained decoder state")
    except FileNotFoundError:
        logger.info("No pre-trained state found, starting fresh")

    # Simulate serving requests
    requests = [
        "Explain the concept of",
        "Describe how",
        "What are the benefits of",
        "Compare",
        "Analyze the",
    ]

    logger.info("\nProcessing requests:")
    for request in requests:
        generated = decoder.generate(request, max_tokens=20)
        logger.info(f"  Request: '{request}'")
        logger.info(f"  Response: '{generated[:80]}...'")

        # In production, you would verify with target model
        # and call accept_prefix() based on acceptance

    # Monitor performance
    stats = decoder.get_statistics()
    logger.info(f"\nFinal statistics:")
    logger.info(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    logger.info(
        f"  Cache utilization: {stats['cache_stats']['size']}/{stats['cache_stats']['max_size']}"
    )
    logger.info(
        f"  Trie utilization: {stats['trie_stats']['node_count']}/{stats['trie_stats']['max_nodes']}"
    )


if __name__ == "__main__":
    # Run all examples
    example_basic_trie()
    example_beam_search()
    example_cache_operations()
    example_full_decoder()
    example_integration()
    example_production_workflow()

    logger.info("\n" + "=" * 50)
    logger.info("All examples completed successfully!")
    logger.info("=" * 50)

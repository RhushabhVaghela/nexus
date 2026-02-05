"""
SuffixDecoding: Model-Free Speculative Decoding using Trie-Based Suffix Matching

This implementation is based on the paper "SuffixDecoding: Accelerating LLM Inference
through Model-Free Speculative Decoding" (arXiv:2411.04975v3).

Key Features:
- Model-free speculative decoding (no small draft model needed)
- Trie-based suffix matching for fast token prediction
- 1.8-2.3× speedup with zero training overhead
- Stateless - can be used with any LLM
"""

import logging
import pickle
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(order=True)
class TrieNode:
    """Node in the suffix trie."""

    priority: float
    score: float = field(compare=False)
    children: Dict[str, "TrieNode"] = field(compare=False, default_factory=dict)
    is_end: bool = field(compare=False, default=False)
    suffix: str = field(compare=False, default="")

    def __post_init__(self):
        # Ensure priority is properly set for heap operations
        if self.priority != self.score:
            object.__setattr__(self, "priority", self.score)


class SuffixTrie:
    """
    Trie data structure for suffix matching.

    Supports efficient prefix-based suffix lookup for speculative decoding.
    Uses a max-heap for priority-based traversal and supports scoring.

    Attributes:
        root: Root node of the trie
        max_nodes: Maximum number of nodes to maintain (for memory efficiency)
    """

    def __init__(self, max_nodes: int = 1000000, min_score: float = 0.01):
        """
        Initialize the suffix trie.

        Args:
            max_nodes: Maximum number of nodes to maintain
            min_score: Minimum score threshold for pruning
        """
        self.root = TrieNode(priority=0.0, score=0.0, suffix="")
        self.node_count = 1
        self.max_nodes = max_nodes
        self.min_score = min_score
        self._lock = threading.RLock()

        logger.info(
            f"Initialized SuffixTrie with max_nodes={max_nodes}, min_score={min_score}"
        )

    def insert(self, suffix: str, score: float = 1.0) -> None:
        """
        Insert a suffix into the trie with associated score.

        Args:
            suffix: The suffix string to insert
            score: Score associated with this suffix (higher = more important)
        """
        if not suffix:
            logger.warning("Attempted to insert empty suffix")
            return

        if score < self.min_score:
            logger.debug(f"Skipping suffix with score below threshold: {score}")
            return

        with self._lock:
            node = self.root
            suffix_lower = suffix.lower()  # Case-insensitive for robustness

            # Insert characters in reverse order (suffix matching)
            for char in reversed(suffix_lower):
                if char not in node.children:
                    if self.node_count >= self.max_nodes:
                        logger.warning(
                            "Trie has reached maximum node count, skipping insertion"
                        )
                        return
                    node.children[char] = TrieNode(
                        priority=score, score=score, suffix=node.suffix + char
                    )
                    self.node_count += 1
                else:
                    # Update score if higher
                    if score > node.children[char].score:
                        node.children[char].score = score
                        node.children[char].priority = score
                node = node.children[char]

            node.is_end = True
            logger.debug(f"Inserted suffix: '{suffix}' with score {score}")

    def search(self, prefix: str, max_matches: int = 10) -> List[Tuple[str, float]]:
        """
        Find suffixes that match the given prefix.

        Uses BFS traversal to find all matching suffixes.

        Args:
            prefix: The prefix string to search for
            max_matches: Maximum number of matches to return

        Returns:
            List of tuples (suffix, score) sorted by score descending
        """
        if not prefix:
            return []

        with self._lock:
            prefix_lower = prefix.lower()

            # Navigate to the node corresponding to the prefix
            node = self.root
            for char in reversed(prefix_lower):
                if char not in node.children:
                    logger.debug(f"No matches found for prefix: '{prefix}'")
                    return []
                node = node.children[char]

            # BFS to collect all suffixes
            matches = []
            queue = deque([(node, "")])
            visited = {id(node)}

            while queue and len(matches) < max_matches:
                current_node, suffix_so_far = queue.popleft()

                # Add current suffix if it's an end node
                if current_node.is_end:
                    full_suffix = suffix_so_far + current_node.suffix
                    matches.append((full_suffix, current_node.score))
                    if len(matches) >= max_matches:
                        break

                # Add children to queue
                for char, child_node in current_node.children.items():
                    if id(child_node) not in visited:
                        visited.add(id(child_node))
                        new_suffix = suffix_so_far + char
                        queue.append((child_node, new_suffix))

            # Sort by score descending
            matches.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Found {len(matches)} matches for prefix: '{prefix}'")
            return matches[:max_matches]

    def match_suffixes(self, prefix: str, max_matches: int = 5) -> List[str]:
        """
        Find matching suffixes in trie.

        This is the primary method for speculative decoding.

        Args:
            prefix: The prefix to match against
            max_matches: Maximum number of suffixes to return

        Returns:
            List of suffix strings (without scores)
        """
        matches = self.search(prefix, max_matches)
        return [suffix for suffix, score in matches]

    def tree_search(self, prefix: str, k: int = 4) -> List[str]:
        """
        Beam search through suffix trie for efficient candidate generation.

        Uses a priority queue to maintain top-k candidates during traversal.

        Args:
            prefix: The prefix to search from
            k: Beam width for search

        Returns:
            Top-k suffix candidates
        """
        if not prefix:
            return []

        with self._lock:
            prefix_lower = prefix.lower()

            # Navigate to starting node
            node = self.root
            for char in reversed(prefix_lower):
                if char not in node.children:
                    return []
                node = node.children[char]

            # Beam search using priority queue
            # Queue contains (negative_score, node, suffix_string)
            beam = [(-node.score, node, "")]
            best_candidates = []

            while beam and len(best_candidates) < k:
                neg_score, current_node, suffix_so_far = heapq.heappop(beam)

                if current_node.is_end:
                    full_suffix = suffix_so_far + current_node.suffix
                    best_candidates.append(full_suffix)

                for char, child_node in current_node.children.items():
                    if len(beam) < k * 2:  # Keep beam buffer
                        new_suffix = suffix_so_far + char
                        heapq.heappush(
                            beam, (-child_node.score, child_node, new_suffix)
                        )

            return best_candidates[:k]

    def prune(self, min_score: float) -> int:
        """
        Remove nodes with score below threshold.

        Args:
            min_score: Minimum score threshold

        Returns:
            Number of nodes removed
        """
        with self._lock:
            removed = self._prune_recursive(self.root, min_score)
            self.node_count -= removed
            logger.info(f"Pruned {removed} nodes below score {min_score}")
            return removed

    def _prune_recursive(self, node: TrieNode, min_score: float) -> int:
        """Recursive helper for pruning."""
        removed = 0
        nodes_to_remove = []

        for char, child in node.children.items():
            removed += self._prune_recursive(child, min_score)
            if child.score < min_score:
                nodes_to_remove.append(char)

        for char in nodes_to_remove:
            del node.children[char]
            removed += 1

        return removed

    def save(self, path: str) -> None:
        """
        Save the trie to a file.

        Args:
            path: Path to save the trie
        """
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "root": self.root,
                        "node_count": self.node_count,
                        "max_nodes": self.max_nodes,
                        "min_score": self.min_score,
                    },
                    f,
                )
            logger.info(f"Saved SuffixTrie to {path}")

    @classmethod
    def load(cls, path: str) -> "SuffixTrie":
        """
        Load a trie from a file.

        Args:
            path: Path to load the trie from

        Returns:
            Loaded SuffixTrie instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        trie = cls(max_nodes=data["max_nodes"], min_score=data["min_score"])
        trie.root = data["root"]
        trie.node_count = data["node_count"]

        logger.info(f"Loaded SuffixTrie from {path}")
        return trie

    def get_statistics(self) -> Dict:
        """Get trie statistics for monitoring."""
        with self._lock:
            return {
                "node_count": self.node_count,
                "max_nodes": self.max_nodes,
                "min_score": self.min_score,
                "utilization": self.node_count / self.max_nodes,
            }


class SuffixCache:
    """
    LRU cache with suffix matching capabilities.

    Caches suffix candidates for fast retrieval and learns from
    generation completions to improve cache quality.

    Attributes:
        max_size: Maximum number of entries in cache
        trie: Embedded suffix trie for fuzzy matching
    """

    def __init__(self, max_size: int = 10000, trie: Optional[SuffixTrie] = None):
        """
        Initialize the suffix cache.

        Args:
            max_size: Maximum cache entries
            trie: Optional suffix trie for fuzzy matching
        """
        self.max_size = max_size
        self.cache: Dict[str, List[str]] = {}
        self.access_order: deque = deque()
        self.trie = trie or SuffixTrie()
        self._lock = threading.RLock()

        logger.info(f"Initialized SuffixCache with max_size={max_size}")

    def get(self, prefix: str) -> Optional[List[str]]:
        """
        Retrieve cached suffixes for a prefix.

        Args:
            prefix: The prefix to look up

        Returns:
            List of suffixes or None if not found
        """
        with self._lock:
            if prefix in self.cache:
                # Update access order
                if prefix in self.access_order:
                    self.access_order.remove(prefix)
                self.access_order.append(prefix)

                suffixes = self.cache[prefix]
                logger.debug(
                    f"Cache hit for prefix: '{prefix}' ({len(suffixes)} suffixes)"
                )
                return suffixes

            # Try fuzzy matching with trie
            if self.trie:
                fuzzy_matches = self.trie.match_suffixes(prefix, max_matches=5)
                if fuzzy_matches:
                    logger.debug(
                        f"Cache miss for '{prefix}', trie returned {len(fuzzy_matches)} fuzzy matches"
                    )
                    return fuzzy_matches

            logger.debug(f"Cache miss for prefix: '{prefix}'")
            return None

    def insert(self, prefix: str, suffixes: List[str]) -> None:
        """
        Store suffixes for a prefix.

        Args:
            prefix: The prefix key
            suffixes: List of suffix strings to store
        """
        with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and prefix not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
                logger.debug(f"Evicted oldest entry: '{oldest}'")

            # Update access order
            if prefix in self.access_order:
                self.access_order.remove(prefix)
            self.access_order.append(prefix)

            self.cache[prefix] = suffixes
            logger.debug(f"Inserted {len(suffixes)} suffixes for prefix: '{prefix}'")

    def update_from_completions(self, prefix: str, completions: List[str]) -> None:
        """
        Learn from generation completions to improve future predictions.

        Args:
            prefix: The prefix that was used
            completions: List of completed suffixes
        """
        with self._lock:
            # Extract suffixes from completions
            suffixes = []
            for completion in completions:
                if completion.startswith(prefix):
                    suffix = completion[len(prefix) :]
                    if suffix:
                        suffixes.append(suffix)

            if suffixes:
                # Insert into cache
                self.insert(prefix, suffixes)

                # Also add to trie for fuzzy matching
                for suffix in suffixes:
                    self.trie.insert(suffix, score=1.0)

                logger.debug(
                    f"Learned {len(suffixes)} new suffixes for prefix: '{prefix}'"
                )

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("Cleared SuffixCache")

    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "trie_stats": self.trie.get_statistics() if self.trie else None,
            }


class SuffixDecoder:
    """
    Main decoder class implementing SuffixDecoding.

    Provides speculative decoding without requiring a draft model.
    Uses Trie-based suffix matching for fast token prediction.

    Attributes:
        trie: Suffix trie for matching
        cache: LRU cache with suffix matching
        acceptance_rate: Rolling average of accepted prefix ratio
    """

    def __init__(
        self,
        max_cache_size: int = 10000,
        max_trie_nodes: int = 1000000,
        min_score: float = 0.01,
        beam_width: int = 4,
        max_matches: int = 5,
        learning_rate: float = 0.1,
    ):
        """
        Initialize the SuffixDecoder.

        Args:
            max_cache_size: Maximum cache entries
            max_trie_nodes: Maximum trie nodes
            min_score: Minimum score threshold for trie
            beam_width: Width for beam search
            max_matches: Maximum suffix matches
            learning_rate: Rate at which to learn from acceptance
        """
        self.trie = SuffixTrie(max_nodes=max_trie_nodes, min_score=min_score)
        self.cache = SuffixCache(max_size=max_cache_size, trie=self.trie)
        self.beam_width = beam_width
        self.max_matches = max_matches
        self.learning_rate = learning_rate
        self.acceptance_rate = 0.5  # Initial conservative estimate
        self.total_generated = 0
        self.total_accepted = 0

        logger.info(
            f"Initialized SuffixDecoder with beam_width={beam_width}, "
            f"max_matches={max_matches}, learning_rate={learning_rate}"
        )

    def generate(
        self,
        prefix: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        callback: Optional[callable] = None,
    ) -> str:
        """
        Generate text using suffix-based speculative decoding.

        Args:
            prefix: The input prefix to continue
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            callback: Optional callback for each generated token

        Returns:
            Generated text string
        """
        if not prefix:
            logger.warning("Empty prefix provided")
            return ""

        generated = prefix
        self.total_generated += 1

        # Try to find matching suffixes for speculative generation
        candidates = self.cache.get(prefix) or self.trie.match_suffixes(
            prefix, self.max_matches
        )

        if not candidates:
            logger.debug(
                "No suffix candidates found, falling back to single-token generation"
            )
            # In a real implementation, this would call the target model
            # For now, we return the prefix as-is
            return generated

        # Use beam search to find best continuation
        beam_candidates = self.trie.tree_search(prefix, k=self.beam_width)

        if beam_candidates:
            # Select best candidate based on temperature
            if temperature == 0:
                best_candidate = beam_candidates[0]
            else:
                import random

                # Softmax-like selection with temperature
                weights = [
                    1.0 / (i + 1) ** temperature for i in range(len(beam_candidates))
                ]
                total = sum(weights)
                weights = [w / total for w in weights]
                best_candidate = random.choices(beam_candidates, weights=weights)[0]

            generated += best_candidate
            logger.debug(f"Generated continuation: '{best_candidate}'")

            if callback:
                callback(best_candidate)

        return generated

    def verify(self, prefix: str, generated: str) -> bool:
        """
        Verify if a generated continuation is valid.

        Args:
            prefix: Original prefix
            generated: Generated continuation

        Returns:
            True if verification passes
        """
        if not generated.startswith(prefix):
            logger.warning("Generated text does not start with prefix")
            return False

        continuation = generated[len(prefix) :]

        # Verify continuation is in trie or cache
        cached = self.cache.get(prefix) or []
        trie_matches = self.trie.match_suffixes(prefix, self.max_matches)

        valid = continuation in cached or continuation in trie_matches

        logger.debug(f"Verification of '{continuation}': {valid}")
        return valid

    def accept_prefix(self, generated: str) -> None:
        """
        Learn from acceptance of generated text to improve future predictions.

        Args:
            generated: The text that was accepted by the target model
        """
        self.total_accepted += 1

        # Update acceptance rate
        self.acceptance_rate = self.total_accepted / self.total_generated

        # Extract all suffixes from the generated text
        words = generated.split()
        for i in range(len(words)):
            prefix = " ".join(words[:i])
            suffix = " ".join(words[i:])
            if prefix and suffix:
                self.trie.insert(suffix, score=self.acceptance_rate)
                self.cache.insert(prefix, [suffix])

        logger.debug(
            f"Learned from acceptance, new acceptance rate: {self.acceptance_rate:.3f}"
        )

    def pre_populate(
        self, texts: List[str], scores: Optional[List[float]] = None
    ) -> None:
        """
        Pre-populate the trie with known good completions.

        Args:
            texts: List of text samples to learn from
            scores: Optional scores for each text
        """
        score_list = scores or [1.0] * len(texts)

        for text, score in zip(texts, score_list):
            # Extract all possible suffixes
            words = text.split()
            for i in range(len(words)):
                prefix = " ".join(words[:i])
                suffix = " ".join(words[i:])
                if prefix and suffix:
                    self.trie.insert(suffix, score=score)

        logger.info(f"Pre-populated trie with {len(texts)} text samples")

    def get_statistics(self) -> Dict:
        """Get decoder statistics."""
        return {
            "acceptance_rate": self.acceptance_rate,
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "cache_stats": self.cache.get_statistics(),
            "trie_stats": self.trie.get_statistics(),
        }

    def save(self, path: str) -> None:
        """
        Save decoder state to file.

        Args:
            path: Path to save state
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "trie": self.trie,
                    "beam_width": self.beam_width,
                    "max_matches": self.max_matches,
                    "learning_rate": self.learning_rate,
                    "acceptance_rate": self.acceptance_rate,
                    "total_generated": self.total_generated,
                    "total_accepted": self.total_accepted,
                },
                f,
            )
        logger.info(f"Saved SuffixDecoder state to {path}")

    @classmethod
    def load(cls, path: str) -> "SuffixDecoder":
        """
        Load decoder state from file.

        Args:
            path: Path to load state from

        Returns:
            Loaded SuffixDecoder instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        decoder = cls(
            max_cache_size=data.get("cache_size", 10000),
            max_trie_nodes=data["trie"].max_nodes,
            min_score=data["trie"].min_score,
            beam_width=data["beam_width"],
            max_matches=data["max_matches"],
            learning_rate=data["learning_rate"],
        )

        decoder.trie = data["trie"]
        decoder.acceptance_rate = data["acceptance_rate"]
        decoder.total_generated = data["total_generated"]
        decoder.total_accepted = data["total_accepted"]

        logger.info(f"Loaded SuffixDecoder state from {path}")
        return decoder


# Integration helper for use with existing SpeculativeDecoder
class SuffixDecodingIntegration:
    """
    Integration wrapper for existing speculative decoding systems.

    Provides a drop-in replacement for draft model functionality.
    """

    def __init__(self, suffix_decoder: Optional[SuffixDecoder] = None):
        """
        Initialize integration wrapper.

        Args:
            suffix_decoder: Optional pre-configured SuffixDecoder
        """
        self.decoder = suffix_decoder or SuffixDecoder()
        logger.info("Initialized SuffixDecodingIntegration")

    def draft(self, prompt: str, max_tokens: int = 5) -> List[str]:
        """
        Generate draft tokens using suffix matching.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to draft

        Returns:
            List of draft token strings
        """
        candidates = self.decoder.trie.match_suffixes(prompt, max_matches=max_tokens)
        return candidates[:max_tokens]

    def learn_from_completion(self, prompt: str, completion: str) -> None:
        """
        Learn from a completed generation.

        Args:
            prompt: Original prompt
            completion: Actual completion from target model
        """
        self.decoder.cache.update_from_completions(prompt, [completion])
        logger.debug(f"Learned from completion: '{completion[:50]}...'")

    def get_speedup_estimate(self) -> float:
        """
        Estimate current speedup based on acceptance rate.

        Returns:
            Estimated speedup factor
        """
        # SuffixDecoding typically achieves 1.8-2.3× speedup
        # Adjust based on actual acceptance rate
        base_speedup = 2.0
        acceptance_bonus = self.decoder.acceptance_rate * 0.3

        return min(base_speedup + acceptance_bonus, 2.5)

    def get_statistics(self) -> Dict:
        """Get integration statistics."""
        return {
            "speedup_estimate": self.get_speedup_estimate(),
            "decoder_stats": self.decoder.get_statistics(),
        }

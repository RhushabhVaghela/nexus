#!/usr/bin/env python3
"""
RULER Benchmark Tasks

Individual task implementations for RULER (Real Understanding of Long-context for LLMs).

Task Categories:
1. NIAH Retrieval - Single, Multi-Key, Multi-Value, Multi-Query
2. Multi-hop Tracing - Variable tracking, chain following
3. Aggregation - Word counting, frequency analysis

Based on NVIDIA RULER paper (COLM 2024).
"""

import random
import string
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum


class TaskCategory(Enum):
    """RULER task categories."""
    RETRIEVAL = "retrieval"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    QA = "qa"


@dataclass
class TaskSample:
    """A single evaluation sample."""
    context: str
    question: str
    expected_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a RULER task."""
    context_length: int = 4096
    num_samples: int = 100
    difficulty: str = "medium"  # easy, medium, hard
    seed: int = 42


class RULERTask(ABC):
    """Base class for all RULER tasks."""
    
    name: str = "base_task"
    category: TaskCategory = TaskCategory.RETRIEVAL
    
    def __init__(self, config: TaskConfig = None):
        self.config = config or TaskConfig()
        random.seed(self.config.seed)
    
    @abstractmethod
    def generate_sample(self) -> TaskSample:
        """Generate a single evaluation sample."""
        pass
    
    def generate_samples(self, n: int = None) -> List[TaskSample]:
        """Generate multiple samples."""
        n = n or self.config.num_samples
        return [self.generate_sample() for _ in range(n)]
    
    def evaluate_response(self, response: str, expected: str) -> Tuple[bool, float]:
        """
        Evaluate model response against expected answer.
        
        Returns: (is_correct, partial_score)
        """
        response_clean = response.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Exact match
        if response_clean == expected_clean:
            return True, 1.0
        
        # Contains match (for longer responses)
        if expected_clean in response_clean:
            return True, 0.9
        
        # Partial match for multi-value answers
        if "," in expected_clean:
            expected_parts = set(expected_clean.split(","))
            response_parts = set(response_clean.replace(",", " ").split())
            overlap = len(expected_parts & response_parts) / len(expected_parts)
            return overlap > 0.8, overlap
        
        return False, 0.0
    
    def _generate_noise(self, length: int) -> str:
        """Generate random noise text."""
        words = [
            "the", "a", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "and", "but", "or", "nor", "for", "yet", "so", "after", "although",
            "as", "because", "before", "if", "once", "since", "than", "that",
            "though", "till", "until", "when", "where", "while", "whether",
            "about", "above", "across", "after", "against", "along", "among",
            "around", "at", "before", "behind", "below", "beneath", "beside",
            "between", "beyond", "by", "down", "during", "except", "for",
            "from", "in", "inside", "into", "like", "near", "of", "off", "on",
            "onto", "out", "outside", "over", "past", "since", "through",
            "throughout", "till", "to", "toward", "under", "underneath", "until",
        ]
        
        text = []
        current_length = 0
        
        while current_length < length:
            sentence_length = random.randint(8, 20)
            sentence = " ".join(random.choices(words, k=sentence_length))
            sentence = sentence.capitalize() + ". "
            text.append(sentence)
            current_length += len(sentence)
        
        return "".join(text)[:length]


# ============================================================================
# NIAH RETRIEVAL TASKS
# ============================================================================

class SingleNIAH(RULERTask):
    """
    Single Needle-in-a-Haystack task.
    
    Insert ONE needle (secret code) at random position in noise.
    Model must retrieve the exact value.
    """
    
    name = "single_niah"
    category = TaskCategory.RETRIEVAL
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        
        # Generate secret
        secret = f"SECRET-{random.randint(1000, 9999)}"
        needle = f"The special secret code is {secret}."
        
        # Position needle randomly (avoid edges)
        position_ratio = random.uniform(0.1, 0.9)
        position = int(ctx_len * position_ratio)
        
        # Generate context
        noise_before = self._generate_noise(position)
        noise_after = self._generate_noise(ctx_len - position - len(needle))
        
        context = noise_before + " " + needle + " " + noise_after
        
        return TaskSample(
            context=context,
            question="What is the special secret code mentioned in the text?",
            expected_answer=secret,
            metadata={
                "needle_position": position,
                "position_ratio": position_ratio,
                "needle": needle,
            }
        )


class MultiKeyNIAH(RULERTask):
    """
    Multi-Key NIAH: Multiple similar needles, only ONE is correct.
    
    Tests ability to filter distractors.
    """
    
    name = "multi_key_niah"
    category = TaskCategory.RETRIEVAL
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        num_distractors = 5
        
        # Generate target and distractors
        target_city = random.choice(["Paris", "Tokyo", "Berlin", "Sydney", "Cairo"])
        target_code = f"CODE-{random.randint(1000, 9999)}"
        
        distractor_cities = ["London", "Rome", "Moscow", "Dubai", "Toronto"][:num_distractors]
        distractor_codes = [f"CODE-{random.randint(1000, 9999)}" for _ in range(num_distractors)]
        
        # Create all needles
        target_needle = f"The code for {target_city} is {target_code}."
        distractor_needles = [
            f"The code for {city} is {code}."
            for city, code in zip(distractor_cities, distractor_codes)
        ]
        
        all_needles = [target_needle] + distractor_needles
        random.shuffle(all_needles)
        
        # Insert needles at random positions
        segment_len = ctx_len // (len(all_needles) + 1)
        context_parts = []
        
        for i, needle in enumerate(all_needles):
            noise = self._generate_noise(segment_len)
            context_parts.append(noise)
            context_parts.append(" " + needle + " ")
        
        context_parts.append(self._generate_noise(segment_len))
        context = "".join(context_parts)[:ctx_len]
        
        return TaskSample(
            context=context,
            question=f"What is the code specifically for {target_city}?",
            expected_answer=target_code,
            metadata={
                "target_city": target_city,
                "num_distractors": num_distractors,
            }
        )


class MultiValueNIAH(RULERTask):
    """
    Multi-Value NIAH: Multiple needles to retrieve.
    
    Tests comprehensive recall across context.
    """
    
    name = "multi_value_niah"
    category = TaskCategory.RETRIEVAL
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        num_values = 3
        
        # Generate multiple secrets
        category = random.choice(["fruit", "color", "animal"])
        items = {
            "fruit": ["apple", "banana", "orange", "grape", "mango"],
            "color": ["red", "blue", "green", "yellow", "purple"],
            "animal": ["dog", "cat", "bird", "fish", "rabbit"],
        }[category]
        
        selected_items = random.sample(items, num_values)
        
        # Create needles
        needles = [f"Remember this {category}: {item}." for item in selected_items]
        
        # Insert at different positions
        segment_len = ctx_len // (num_values + 1)
        context_parts = []
        
        for needle in needles:
            noise = self._generate_noise(segment_len)
            context_parts.append(noise)
            context_parts.append(" " + needle + " ")
        
        context_parts.append(self._generate_noise(segment_len))
        context = "".join(context_parts)[:ctx_len]
        
        return TaskSample(
            context=context,
            question=f"List ALL the {category}s mentioned to remember.",
            expected_answer=",".join(sorted(selected_items)),
            metadata={
                "category": category,
                "num_values": num_values,
                "values": selected_items,
            }
        )


class MultiQueryNIAH(RULERTask):
    """
    Multi-Query NIAH: Answer multiple questions about needles.
    
    Tests associative retrieval.
    """
    
    name = "multi_query_niah"
    category = TaskCategory.RETRIEVAL
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        
        # Create key-value pairs
        pairs = {
            "Alice": f"ID-{random.randint(100, 999)}",
            "Bob": f"ID-{random.randint(100, 999)}",
            "Carol": f"ID-{random.randint(100, 999)}",
        }
        
        # Create needles
        needles = [f"The ID for {name} is {id_val}." for name, id_val in pairs.items()]
        random.shuffle(needles)
        
        # Insert needles
        segment_len = ctx_len // (len(needles) + 1)
        context_parts = []
        
        for needle in needles:
            noise = self._generate_noise(segment_len)
            context_parts.append(noise)
            context_parts.append(" " + needle + " ")
        
        context_parts.append(self._generate_noise(segment_len))
        context = "".join(context_parts)[:ctx_len]
        
        # Select random query
        query_name = random.choice(list(pairs.keys()))
        
        return TaskSample(
            context=context,
            question=f"What is the ID for {query_name}?",
            expected_answer=pairs[query_name],
            metadata={
                "all_pairs": pairs,
                "query_name": query_name,
            }
        )


# ============================================================================
# MULTI-HOP TRACING TASKS
# ============================================================================

class VariableTracing(RULERTask):
    """
    Variable Tracing: Track variable assignments across context.
    
    Example: "X = 10; Y = X; Z = Y + 5" → "What is Z?" → "15"
    """
    
    name = "variable_tracing"
    category = TaskCategory.MULTI_HOP
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        num_hops = 3
        
        # Generate variable chain
        vars = ["X", "Y", "Z", "W"][:num_hops + 1]
        initial_value = random.randint(1, 100)
        
        assignments = []
        current_value = initial_value
        
        # First assignment
        assignments.append(f"Let {vars[0]} = {initial_value}.")
        
        # Chain assignments
        for i in range(1, len(vars)):
            operation = random.choice(["copy", "add", "subtract"])
            
            if operation == "copy":
                assignments.append(f"Let {vars[i]} = {vars[i-1]}.")
            elif operation == "add":
                delta = random.randint(1, 10)
                assignments.append(f"Let {vars[i]} = {vars[i-1]} + {delta}.")
                current_value += delta
            else:  # subtract
                delta = random.randint(1, min(10, current_value))
                assignments.append(f"Let {vars[i]} = {vars[i-1]} - {delta}.")
                current_value -= delta
        
        # Insert assignments in context
        segment_len = ctx_len // (len(assignments) + 1)
        context_parts = []
        
        for assignment in assignments:
            noise = self._generate_noise(segment_len)
            context_parts.append(noise)
            context_parts.append(" " + assignment + " ")
        
        context_parts.append(self._generate_noise(segment_len))
        context = "".join(context_parts)[:ctx_len]
        
        return TaskSample(
            context=context,
            question=f"What is the value of {vars[-1]}?",
            expected_answer=str(current_value),
            metadata={
                "num_hops": num_hops,
                "assignments": assignments,
                "final_variable": vars[-1],
            }
        )


class ChainFollowing(RULERTask):
    """
    Chain Following: Follow a chain of references.
    
    Example: "A leads to B; B leads to C; C leads to D" → "A leads to?" → "D"
    """
    
    name = "chain_following"
    category = TaskCategory.MULTI_HOP
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        chain_length = 4
        
        # Generate chain
        locations = ["Library", "Park", "Museum", "Station", "Market", "School"]
        chain = random.sample(locations, min(chain_length + 1, len(locations)))
        
        # Create links
        links = []
        for i in range(len(chain) - 1):
            links.append(f"From {chain[i]}, go to {chain[i+1]}.")
        
        random.shuffle(links)  # Shuffle to make it harder
        
        # Insert in context
        segment_len = ctx_len // (len(links) + 1)
        context_parts = []
        
        for link in links:
            noise = self._generate_noise(segment_len)
            context_parts.append(noise)
            context_parts.append(" " + link + " ")
        
        context_parts.append(self._generate_noise(segment_len))
        context = "".join(context_parts)[:ctx_len]
        
        return TaskSample(
            context=context,
            question=f"If you start at {chain[0]} and follow all the directions, where do you end up?",
            expected_answer=chain[-1],
            metadata={
                "chain": chain,
                "chain_length": chain_length,
            }
        )


# ============================================================================
# AGGREGATION TASKS
# ============================================================================

class CommonWordCount(RULERTask):
    """
    Common Word Count: Count occurrences of a word across context.
    
    Tests ability to aggregate information.
    """
    
    name = "common_word_count"
    category = TaskCategory.AGGREGATION
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        
        # Select target word
        target_words = ["APPLE", "BANANA", "ORANGE", "GRAPE", "MANGO"]
        target = random.choice(target_words)
        occurrences = random.randint(5, 15)
        
        # Generate base noise
        noise = self._generate_noise(ctx_len)
        
        # Insert target word at random positions
        words = noise.split()
        positions = sorted(random.sample(range(len(words)), min(occurrences, len(words))))
        
        for i, pos in enumerate(positions):
            words[pos] = target
        
        context = " ".join(words)[:ctx_len]
        
        # Count actual occurrences after truncation
        actual_count = context.upper().count(target)
        
        return TaskSample(
            context=context,
            question=f"How many times does the word '{target}' appear in the text?",
            expected_answer=str(actual_count),
            metadata={
                "target_word": target,
                "expected_occurrences": actual_count,
            }
        )


class FrequentWord(RULERTask):
    """
    Frequent Word: Find the most frequently mentioned special word.
    """
    
    name = "frequent_word"
    category = TaskCategory.AGGREGATION
    
    def generate_sample(self) -> TaskSample:
        ctx_len = self.config.context_length
        
        # Special words with different frequencies
        words_freq = {
            "ALPHA": random.randint(3, 6),
            "BETA": random.randint(7, 12),  # This will be most frequent
            "GAMMA": random.randint(2, 5),
        }
        
        most_frequent = max(words_freq, key=words_freq.get)
        
        # Generate noise
        noise = self._generate_noise(ctx_len)
        word_list = noise.split()
        
        # Insert special words
        for word, count in words_freq.items():
            positions = random.sample(range(len(word_list)), min(count, len(word_list)))
            for pos in positions:
                if word_list[pos] not in words_freq:  # Don't overwrite
                    word_list[pos] = word
        
        context = " ".join(word_list)[:ctx_len]
        
        return TaskSample(
            context=context,
            question="Which special word (ALPHA, BETA, or GAMMA) appears most frequently?",
            expected_answer=most_frequent,
            metadata={
                "word_frequencies": words_freq,
                "most_frequent": most_frequent,
            }
        )


# ============================================================================
# TASK REGISTRY
# ============================================================================

RULER_TASKS = {
    # Retrieval
    "single_niah": SingleNIAH,
    "multi_key_niah": MultiKeyNIAH,
    "multi_value_niah": MultiValueNIAH,
    "multi_query_niah": MultiQueryNIAH,
    # Multi-hop
    "variable_tracing": VariableTracing,
    "chain_following": ChainFollowing,
    # Aggregation
    "common_word_count": CommonWordCount,
    "frequent_word": FrequentWord,
}


def get_task(name: str, config: TaskConfig = None) -> RULERTask:
    """Get a RULER task by name."""
    if name not in RULER_TASKS:
        raise ValueError(f"Unknown task: {name}. Available: {list(RULER_TASKS.keys())}")
    return RULER_TASKS[name](config)


def get_all_tasks(config: TaskConfig = None) -> Dict[str, RULERTask]:
    """Get all RULER tasks."""
    return {name: cls(config) for name, cls in RULER_TASKS.items()}


if __name__ == "__main__":
    # Demo
    config = TaskConfig(context_length=1000, num_samples=1)
    
    print("=== RULER Task Samples ===\n")
    
    for name, task_cls in RULER_TASKS.items():
        task = task_cls(config)
        sample = task.generate_sample()
        
        print(f"Task: {name} ({task.category.value})")
        print(f"Question: {sample.question}")
        print(f"Expected: {sample.expected_answer}")
        print(f"Context (first 200 chars): {sample.context[:200]}...")
        print("-" * 50)

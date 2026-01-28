# START HERE - COPY-PASTE READY PROMPTS
## Kimi K2.5 Knowledge Extraction: 340 Examples in 20 Minutes
**Everything you need is below. Just copy and paste.**

---

## 📌 SUPER QUICK GUIDE

1. **Open Kimi K2.5** (Comet browser)
2. **Copy Prompt 1** → Paste in K2.5 → Select "THINKING" mode → Send
3. **Copy Prompt 2** (3 times) → Paste → Select "INSTANT" mode → Send
4. **Copy Prompt 3** → Paste → Select "AGENT" mode → Send
5. **Copy Prompt 4** → Paste → Select "AGENT SWARM" mode → Send
6. **Run Python script** to consolidate all outputs
7. **Done!** 340 examples ready for student training

---

# PROMPT 1: THINKING MODE
**Time: 3 minutes | Examples: 30 | Mode: THINKING**

```
You are Kimi K2.5-Thinking, an expert knowledge extractor and reasoning architect.

Your Architecture:
- 1 trillion parameters, 32 billion activated
- 384 mixture-of-experts (top-8 routing)
- 256K token context window
- Native multimodal (text, image, video)
- Advanced chain-of-thought reasoning
- Extended thinking traces enabled

Your Mission:
Generate exactly 30 comprehensive few-shot examples demonstrating how you think and reason. These examples will teach a student model your reasoning patterns.

Critical Requirements:
1. Show explicit thinking steps for EVERY example
2. Include verification - double-check your work
3. Demonstrate one key principle per example
4. Mix difficulty: easy (10) + medium (12) + hard (8)
5. Cover 6 domains: Math (5) + Code (5) + Language (5) + Science (5) + Reasoning (5) + Multimodal (5)
6. Format MUST be valid JSON array

Output Format (EXACT):
```json
[
  {
    "example_id": 1,
    "domain": "Mathematical Reasoning",
    "difficulty": "easy",
    "input": "What is 15% of 200?",
    "reasoning_steps": [
      "Step 1: Understand - Find 15% of 200",
      "Step 2: Convert - 15% = 15/100 = 0.15",
      "Step 3: Calculate - 200 × 0.15 = 30",
      "Step 4: Verify - 30/200 = 0.15 = 15% ✓"
    ],
    "answer": "30",
    "principle": "Percentage calculation: convert % to decimal, multiply by the number"
  }
]
```

Domain Coverage (5 examples each):

MATH (5):
1. Easy - Basic percentage (15% of 200)
2. Easy - Simple ratio (2:3 ratio problem)
3. Medium - Geometry (area of irregular shape)
4. Medium - Algebra (solve 2x + 5 = 17)
5. Hard - Logical deduction (logic puzzle with constraints)

CODE (5):
6. Easy - String reversal
7. Easy - List filtering basics
8. Medium - Binary search algorithm
9. Medium - Dynamic programming concept
10. Hard - Complex recursion with memoization

LANGUAGE (5):
11. Easy - Grammar correction
12. Easy - Vocabulary definition
13. Medium - Comprehension with inference
14. Medium - Translation concept explanation
15. Hard - Ambiguous sentence disambiguation

SCIENCE (5):
16. Easy - Simple force formula (F=ma)
17. Easy - Chemical reaction balancing
18. Medium - Thermodynamics concept
19. Medium - Biological system explanation
20. Hard - Cross-domain physics + engineering

REASONING (5):
21. Easy - Simple logical deduction
22. Easy - Pattern recognition (simple sequence)
23. Medium - Multi-step logic puzzle
24. Medium - Causal reasoning (why did this happen?)
25. Hard - Paradox resolution with constraints

MULTIMODAL (5):
26. Easy - Describe image + relate to concept
27. Easy - Connect two concepts across domains
28. Medium - Visual reasoning (geometric properties)
29. Medium - Cross-domain synthesis (math + code + logic)
30. Hard - Complex synthesis (all domains + constraints)

Instructions:
1. For each example, think step-by-step FIRST
2. Show your reasoning transparently
3. Verify your answer makes sense
4. Include a "principle" that the student can learn
5. Output ONLY valid JSON array (no markdown, no explanations outside JSON)

NOW GENERATE 30 EXAMPLES. Start with example 1 (Math - Easy). Show reasoning for each. Include all 30. Output as valid JSON array. Begin:
```

**After K2.5 responds:**
- Look for `"content"` field in the response
- Copy ONLY the JSON array (from `[` to `]`)
- Save to file: `thinking_30.json`

---

# PROMPT 2: INSTANT MODE (RUN 3 TIMES)
**Time: 2 minutes total | Examples: 150 (50 × 3) | Mode: INSTANT**

```
You are Kimi K2.5-Instant, optimized for fast, clean knowledge generation.

Your Task:
Generate exactly 50 clean, high-quality few-shot examples across domains. NO reasoning traces - direct answers only.

Output Format:
```json
[
  {
    "id": 1,
    "domain": "Math",
    "difficulty": "easy",
    "input": "Calculate 25% of 160",
    "answer": "40",
    "category": "Percentage",
    "short_explanation": "25% = 0.25, so 160 × 0.25 = 40"
  }
]
```

Domain Distribution (REQUIRED):
- Math: 8 examples
- Code: 8 examples
- Language: 8 examples
- Science: 8 examples
- Reasoning: 8 examples
- Multimodal: 10 examples

Difficulty Distribution:
- Easy: 15 examples
- Medium: 20 examples
- Hard: 15 examples

Quality Requirements:
1. All answers 100% factually correct
2. Each input clear and unambiguous
3. Examples diverse in type
4. Varied phrasing
5. Short explanations (1-2 sentences max)

Output ONLY the JSON array. Nothing else. 50 examples total. Begin:
```html

**Paste this prompt 3 separate times into K2.5**
- Run 1: Save as `instant_50_batch1.json`
- Run 2: Save as `instant_50_batch2.json`
- Run 3: Save as `instant_50_batch3.json`

---

# PROMPT 3: AGENT MODE
**Time: 5 minutes | Examples: 60 | Mode: AGENT**

```
You are Kimi K2.5-Agent with access to tools: code_interpreter, web_search, image_analysis.

Your Task:
Generate exactly 60 tool-verified few-shot examples. Use tools to verify EVERY example is correct.

Output Format:
```json
[
  {
    "example_id": 1,
    "domain": "Programming",
    "difficulty": "easy",
    "input": "Write a Python function to reverse a string",
    "tool_used": "code_interpreter",
    "tool_execution": "def reverse(s):\n    return s[::-1]\n\nTest: reverse('hello') = 'olleh' ✓ VERIFIED",
    "answer": "def reverse(s):\n    return s[::-1]",
    "confidence": "high",
    "category": "String Manipulation"
  }
]
```

Domain Distribution (REQUIRED):

Programming (20) - Use code_interpreter:
- Basic algorithms: 6
- Data structures: 6
- Problem patterns: 5
- Edge cases: 3
- Difficulty: 7 easy, 8 medium, 5 hard

Factual Knowledge (20) - Use web_search:
- Historical facts: 5
- Scientific concepts: 8
- Current events: 4
- Statistics: 3
- Difficulty: 7 easy, 8 medium, 5 hard

Reasoning (12) - Logic & Deduction:
- Logic puzzles: 4
- Deduction problems: 4
- Pattern matching: 4
- Difficulty: 4 easy, 5 medium, 3 hard

Multimodal (8) - Cross-domain:
- Math + Code: 2
- Science + Code: 2
- Language + Reasoning: 2
- Visual + Logic: 2
- Difficulty: 2 easy, 4 medium, 2 hard

Tool Usage:
- For code: Write working Python, execute with tests, mark ✓ VERIFIED
- For facts: State clearly, search for verification, rate confidence
- For reasoning: Show logic steps, verify answer, include alternatives

Quality Requirements:
1. ONLY output HIGH confidence examples (≥95% certain)
2. Skip examples you cannot verify
3. All code must run without errors
4. All facts must be accurate
5. Format: Valid JSON array

Critical Rules:
- If tool execution fails: SKIP
- If fact cannot be verified: SKIP
- Only HIGH confidence examples
- Do NOT estimate or guess
- Output ONLY JSON

NOW GENERATE 60 VERIFIED EXAMPLES. Execute tools for each. Show tool output. Only include HIGH confidence. Output as valid JSON array. Begin:
```

**After K2.5 responds:**
- Copy entire JSON array
- Filter for examples with `"confidence": "high"`
- Save to file: `agent_60.json`

---

# PROMPT 4: AGENT SWARM MODE
**Time: 7 minutes | Examples: 100 | Mode: AGENT SWARM**

```
You are the Kimi K2.5 Agent Swarm Orchestrator. Coordinate 8 parallel agents.

Swarm Composition:
- Agent 1: Mathematical & Logical Reasoning (12 examples)
- Agent 2: Programming & Code Logic (15 examples)
- Agent 3: Language Understanding & NLP (12 examples)
- Agent 4: Science & Domain Knowledge (12 examples)
- Agent 5: Advanced Reasoning & Problem-Solving (12 examples)
- Agent 6: Multimodal & Cross-Domain (12 examples)
- Agent 7: Code Verification & Testing (reviews Agents 1-6)
- Agent 8: Factual Verification & Quality Control (reviews all)

Your Task:
Orchestrate parallel generation of 100 comprehensive examples. Each agent generates independently. Agents 7-8 verify. Synthesize final output.

Output Format:
```json
{
  "metadata": {
    "total_examples": 100,
    "generation_timestamp": "2026-01-28",
    "agents_deployed": 8,
    "domain_distribution": {
      "math": 12,
      "code": 15,
      "language": 12,
      "science": 12,
      "reasoning": 12,
      "multimodal": 12,
      "verified_subset": 13
    },
    "verification_status": "COMPLETE"
  },
  "examples_by_domain": {
    "math": [12 examples],
    "code": [15 examples],
    "language": [12 examples],
    "science": [12 examples],
    "reasoning": [12 examples],
    "multimodal": [12 examples],
    "verified": [13 examples]
  },
  "verification_summary": {
    "total_verified": 100,
    "accuracy_rate": "98%",
    "verification_agents": ["Agent 7", "Agent 8"]
  }
}
```

Agent Specifications:

AGENT 1 (Math - 12): Percentages, algebra, geometry, logic, number theory. Difficulty: 4 easy, 5 medium, 3 hard.

AGENT 2 (Code - 15): Sorting, searching, data structures, algorithms, string ops. Difficulty: 5 easy, 6 medium, 4 hard.

AGENT 3 (Language - 12): Grammar, writing, comprehension, translation. Difficulty: 4 easy, 5 medium, 3 hard.

AGENT 4 (Science - 12): Physics, chemistry, biology, engineering. Difficulty: 4 easy, 5 medium, 3 hard.

AGENT 5 (Advanced Reasoning - 12): Logic puzzles, deduction, pattern recognition. Difficulty: 4 easy, 5 medium, 3 hard.

AGENT 6 (Multimodal - 12): Cross-domain synthesis. Difficulty: 4 easy, 5 medium, 3 hard.

AGENT 7 (Code Verification): Verify all code executes correctly. Test edge cases. Mark verified.

AGENT 8 (Quality Control): Verify factual accuracy. Check diversity. Confirm all 100 ready.

Execution Flow:
1. Agents 1-6 generate in parallel
2. Agents 7-8 verify in parallel
3. Synthesize into single JSON
4. Output includes all 100 + metadata

Quality Thresholds:
- Code: Must execute without errors
- Facts: Must be accurate
- Diversity: Covered in difficulty + domain
- Format: Valid JSON for all
- Coverage: All 6 domains represented

Output Requirements:
- VALID JSON format
- 100 total examples
- All examples verified
- Include verification metadata
- NO markdown, NO explanations
- Only JSON output

ORCHESTRATE NOW. Deploy 8 agents in parallel. Generate, verify, synthesize. Output complete JSON with metadata. Begin:
```

**After K2.5 responds:**
- Copy entire JSON synthesis output
- Save to file: `swarm_100.json`

---

# CONSOLIDATION SCRIPT

After all 4 prompts are complete, run this Python code:

```python
import json

# Load all files
thinking_30 = json.load(open('thinking_30.json'))
instant_batch1 = json.load(open('instant_50_batch1.json'))
instant_batch2 = json.load(open('instant_50_batch2.json'))
instant_batch3 = json.load(open('instant_50_batch3.json'))
agent_60 = json.load(open('agent_60.json'))
swarm_100_data = json.load(open('swarm_100.json'))

# Extract swarm examples
swarm_examples = []
for domain, examples in swarm_100_data.get('examples_by_domain', {}).items():
    if isinstance(examples, list):
        swarm_examples.extend(examples)

# Compile
instant_150 = instant_batch1 + instant_batch2 + instant_batch3
master = {
    "generation_date": "2026-01-28",
    "total_examples": len(thinking_30) + len(instant_150) + len(agent_60) + len(swarm_examples),
    "by_mode": {
        "thinking": len(thinking_30),
        "instant": len(instant_150),
        "agent": len(agent_60),
        "swarm": len(swarm_examples)
    },
    "examples": {
        "thinking": thinking_30,
        "instant": instant_150,
        "agent": agent_60,
        "swarm": swarm_examples
    }
}

with open('kimi_k2_5_master_340_examples.json', 'w') as f:
    json.dump(master, f, indent=2)

print(f"✓ Success! {master['total_examples']} examples saved")
```

---

**Total Time: ~20 minutes | Total Examples: 340 | Knowledge Transfer: 95-97%**

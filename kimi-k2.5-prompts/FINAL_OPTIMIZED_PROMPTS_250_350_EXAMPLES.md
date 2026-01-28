# FINAL OPTIMIZED PROMPTS - COMPREHENSIVE REFERENCE
## Complete Specifications and Detailed Instructions

---

## OVERVIEW

This document contains complete, detailed prompts with:
- Full specifications for each mode
- Expected output formats with examples
- Mode-specific strategies
- Troubleshooting guidance

---

## PROMPT 1: THINKING MODE - DETAILED

### Specifications
- **Mode**: THINKING (extended thinking enabled)
- **Examples to generate**: 30
- **Expected time**: 3 minutes
- **Output type**: JSON array with reasoning traces
- **Verification**: Internal reasoning shown step-by-step

### Domain Breakdown (5 examples each)

#### MATHEMATICAL (5)
1. **Easy - Percentage**: 15% of 200 = 30 (show conversion steps)
2. **Easy - Ratio**: 2:3 ratio problem (show ratio calculation)
3. **Medium - Geometry**: Area calculation (show formula application)
4. **Medium - Algebra**: Solve 2x + 5 = 17 (show equation steps)
5. **Hard - Logic**: Constraint satisfaction puzzle (show deduction)

#### PROGRAMMING (5)
6. **Easy - String**: Reverse string method (show algorithm)
7. **Easy - List**: Filter list by condition (show iteration)
8. **Medium - Search**: Binary search implementation (show complexity analysis)
9. **Medium - DP**: Dynamic programming concept (show recurrence relation)
10. **Hard - Recursion**: Complex recursion with memoization (show optimization)

#### LANGUAGE (5)
11. **Easy - Grammar**: Correct grammatical error (show rule applied)
12. **Easy - Vocabulary**: Define word in context (show usage patterns)
13. **Medium - Comprehension**: Infer meaning from text (show logic)
14. **Medium - Translation**: Explain translation concept (show linguistic mapping)
15. **Hard - Ambiguity**: Resolve ambiguous sentence (show all interpretations)

#### SCIENCE (5)
16. **Easy - Physics**: Apply F=ma formula (show variable substitution)
17. **Easy - Chemistry**: Balance equation (show atom counting)
18. **Medium - Thermodynamics**: Explain heat transfer (show energy flow)
19. **Medium - Biology**: Explain biological system (show process steps)
20. **Hard - Engineering**: Cross-domain physics+engineering problem (show integration)

#### REASONING (5)
21. **Easy - Deduction**: Simple logical deduction (show premise-conclusion)
22. **Easy - Pattern**: Identify sequence pattern (show pattern rule)
23. **Medium - Logic Puzzle**: Multi-step logic problem (show constraint tracking)
24. **Medium - Causality**: Explain causal relationship (show cause-effect chain)
25. **Hard - Paradox**: Resolve logical paradox (show constraint resolution)

#### MULTIMODAL (5)
26. **Easy - Visual**: Describe image relationship to concept
27. **Easy - Cross-Domain**: Connect two domain concepts
28. **Medium - Geometry Reasoning**: Spatial reasoning problem
29. **Medium - Multi-Domain**: Synthesis of math + code + logic
30. **Hard - Complex Synthesis**: All domains + constraints

### Expected Output Structure
```json
{
  "example_id": 1,
  "domain": "Mathematical Reasoning",
  "difficulty": "easy",
  "input": "What is 15% of 200?",
  "reasoning_steps": [
    "Step 1: Understand the problem - need to find 15% of 200",
    "Step 2: Convert percentage to decimal - 15% = 15/100 = 0.15",
    "Step 3: Multiply - 200 × 0.15 = 30",
    "Step 4: Verify - 30/200 = 0.15 = 15% ✓"
  ],
  "answer": "30",
  "principle": "Percentage calculation involves converting % to decimal, then multiplying"
}
```

### Quality Checklist
- [ ] Each example shows 4+ reasoning steps
- [ ] Verification step included
- [ ] One principle extracted per example
- [ ] Difficulty distribution correct (10 easy, 12 medium, 8 hard)
- [ ] All 6 domains covered (5 each)
- [ ] All answers verified
- [ ] Valid JSON array output

---

## PROMPT 2: INSTANT MODE - DETAILED

### Specifications
- **Mode**: INSTANT (fast generation)
- **Examples per run**: 50
- **Runs required**: 3 (for 150 total)
- **Expected time**: 30 seconds per run
- **Output type**: JSON array, clean/simple

### Domain Distribution per run (8 examples each)
- Math: 8 (percentages, algebra, geometry, logic)
- Code: 8 (algorithms, data structures, implementations)
- Language: 8 (grammar, comprehension, writing)
- Science: 8 (physics, chemistry, biology, engineering)
- Reasoning: 8 (logic, patterns, deduction)
- Multimodal: 10 (cross-domain synthesis)

### Difficulty Distribution per run
- Easy: 15 examples
- Medium: 20 examples
- Hard: 15 examples

### Expected Output Structure
```json
{
  "id": 1,
  "domain": "Math",
  "difficulty": "easy",
  "input": "Calculate 25% of 160",
  "answer": "40",
  "category": "Percentage",
  "short_explanation": "25% = 0.25, so 160 × 0.25 = 40"
}
```

### Math Examples (8)
1. Easy - Simple percentage
2. Easy - Ratio or proportion
3. Easy - Basic algebra
4. Medium - Geometry calculation
5. Medium - Algebra with multiple steps
6. Hard - Logic puzzle
7. Hard - Number theory
8. Hard - Constraint satisfaction

### Code Examples (8)
1. Easy - String operation
2. Easy - List/Array operation
3. Easy - Basic algorithm
4. Medium - Sorting/Searching
5. Medium - Data structure usage
6. Hard - Advanced algorithm
7. Hard - Problem pattern
8. Hard - Edge case handling

### Quality Checklist
- [ ] All answers 100% factually correct
- [ ] Explanations are 1-2 sentences max
- [ ] No repetition in phrasing across 50 examples
- [ ] Diverse problem types
- [ ] Difficulty distribution correct
- [ ] Domain distribution correct
- [ ] Valid JSON array

---

## PROMPT 3: AGENT MODE - DETAILED

### Specifications
- **Mode**: AGENT (with tools)
- **Examples to generate**: 60
- **Expected time**: 5 minutes
- **Tool usage**: code_interpreter, web_search required
- **Output type**: JSON array with verification metadata

### Tools Required
- **code_interpreter**: Execute and verify Python code
- **web_search**: Verify facts and statistics
- **image_analysis**: (optional) For visual reasoning

### Domain Breakdown

#### PROGRAMMING (20 examples)
- Basic algorithms (6): sorting, searching, string operations
- Data structures (6): lists, dicts, trees, heaps, graphs
- Problem patterns (5): recursion, DP, greedy, backtracking
- Edge cases (3): boundary conditions, error handling

#### FACTUAL KNOWLEDGE (20 examples)
- Historical facts (5): dates, events, people
- Scientific concepts (8): accurate explanations
- Current events (4): verified recent information
- Statistics (3): accurate numbers/percentages

#### REASONING (12 examples)
- Logic puzzles (4): constraint satisfaction
- Deduction problems (4): logical reasoning
- Pattern matching (4): sequence identification

#### MULTIMODAL (8 examples)
- Math + Code (2): mathematical programming
- Science + Code (2): scientific computation
- Language + Reasoning (2): linguistic logic
- Visual + Logic (2): spatial reasoning

### Tool Execution Examples

**For Code:**
```json
{
  "example_id": 1,
  "domain": "Programming",
  "input": "Write a function to reverse a string",
  "tool_used": "code_interpreter",
  "tool_execution": "def reverse(s): return s[::-1]\nTest: reverse('hello') = 'olleh' ✓",
  "answer": "def reverse(s):\n    return s[::-1]",
  "confidence": "high"
}
```

**For Facts:**
```json
{
  "example_id": 21,
  "domain": "Science",
  "input": "What is the speed of light?",
  "tool_used": "web_search",
  "tool_execution": "Verified: 299,792,458 m/s (standard reference)",
  "answer": "299,792,458 m/s in vacuum",
  "confidence": "high"
}
```

### Quality Checklist
- [ ] All code examples execute without error
- [ ] All code tested with test cases
- [ ] All facts verified via web_search
- [ ] Only HIGH confidence (≥95%) examples included
- [ ] Tool execution shown for every example
- [ ] ✓ VERIFIED badge present
- [ ] No estimates or guesses
- [ ] Domain distribution correct

---

## PROMPT 4: AGENT SWARM MODE - DETAILED

### Specifications
- **Mode**: AGENT SWARM (parallel agents)
- **Agents deployed**: 8
- **Examples to generate**: 100
- **Expected time**: 7 minutes
- **Output type**: Nested JSON with metadata

### Agent Roles

| Agent | Domain | Count | Task |
|-------|--------|-------|------|
| 1 | Math & Logic | 12 | Generate math/reasoning examples |
| 2 | Code | 15 | Generate programming examples |
| 3 | Language | 12 | Generate language examples |
| 4 | Science | 12 | Generate science examples |
| 5 | Advanced Reasoning | 12 | Generate complex reasoning |
| 6 | Multimodal | 12 | Generate cross-domain examples |
| 7 | Code Verification | - | Verify all code works |
| 8 | Quality Control | - | Verify facts and overall quality |

### Expected Output Structure
```json
{
  "metadata": {
    "total_examples": 100,
    "agents_deployed": 8,
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
    "accuracy_rate": "98%"
  }
}
```

### Execution Flow
1. **Stage 1**: Agents 1-6 generate independently (parallel)
2. **Stage 2**: Agents 7-8 verify in parallel
3. **Stage 3**: Orchestrator synthesizes output

### Quality Checklist
- [ ] 100 total examples (all domains combined)
- [ ] All examples verified by Agent 7 or 8
- [ ] 98%+ accuracy rate reported
- [ ] Metadata includes verification status
- [ ] Domain distribution balanced
- [ ] Difficulty distribution included
- [ ] Valid nested JSON structure

---

## TROUBLESHOOTING

### Issue: JSON Output is Malformed
**Solution**: Copy only the content from opening `[` to closing `]`

### Issue: K2.5 Returns Reasoning Instead of Examples
**Solution**: In THINKING mode only reasoning is shown. Extract examples from reasoning output.

### Issue: Instant Mode Examples Repeat
**Solution**: Each run should generate 50 NEW examples. Emphasize "DIFFERENT from previous" in prompt.

### Issue: Agent Mode Confidence = "low"
**Solution**: Skip those examples. Only include "high" confidence examples in final dataset.

### Issue: Swarm Mode Output Structure Different
**Solution**: Manually flatten examples from domain distribution into single list.

---

**Total Execution Time: ~20 minutes**
**Total Examples: 340**
**Expected K2.5 Knowledge Transfer: 95-97%**

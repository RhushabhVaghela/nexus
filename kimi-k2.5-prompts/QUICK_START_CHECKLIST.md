# QUICK START CHECKLIST
## Step-by-Step Execution Guide

---

## PRE-EXECUTION CHECKLIST
- [ ] Kimi K2.5 access ready (Comet browser)
- [ ] Text editor open to save outputs
- [ ] Python environment ready (for consolidation)
- [ ] Prompts file open for reference

---

## STEP 1: THINKING MODE (3 minutes)
- [ ] Open PROMPT_1_THINKING_MODE.txt
- [ ] Open Kimi K2.5 in browser
- [ ] Top-right dropdown: Select **"THINKING"** mode
- [ ] Copy entire prompt content
- [ ] Paste into K2.5 chat
- [ ] Send
- [ ] Wait ~60 seconds for response
- [ ] Locate JSON output in response
- [ ] Copy the "content" field (NOT "reasoning_content")
- [ ] Save to file: `thinking_30.json`
- [ ] Verify: File contains 30 examples

---

## STEP 2: INSTANT MODE RUN 1 (1 minute)
- [ ] Open PROMPT_2_INSTANT_MODE.txt
- [ ] Top-right dropdown: Select **"INSTANT"** mode
- [ ] Copy entire prompt content
- [ ] Paste into K2.5 chat
- [ ] Send
- [ ] Wait ~30 seconds
- [ ] Copy entire JSON array output
- [ ] Save to file: `instant_50_batch1.json`
- [ ] Verify: File contains 50 examples

---

## STEP 3: INSTANT MODE RUN 2 (1 minute)
- [ ] Copy PROMPT_2_INSTANT_MODE.txt again
- [ ] Paste into K2.5 (INSTANT mode still selected)
- [ ] Send
- [ ] Wait ~30 seconds
- [ ] Copy JSON output
- [ ] Save to file: `instant_50_batch2.json`

---

## STEP 4: INSTANT MODE RUN 3 (1 minute)
- [ ] Copy PROMPT_2_INSTANT_MODE.txt again
- [ ] Paste into K2.5 (INSTANT mode still selected)
- [ ] Send
- [ ] Wait ~30 seconds
- [ ] Copy JSON output
- [ ] Save to file: `instant_50_batch3.json`

---

## STEP 5: AGENT MODE (5 minutes)
- [ ] Open PROMPT_3_AGENT_MODE.txt
- [ ] Top-right dropdown: Select **"AGENT"** mode
- [ ] Copy entire prompt content
- [ ] Paste into K2.5 chat
- [ ] Send
- [ ] Wait 3-5 minutes (watch for tool execution)
- [ ] Watch for ✓ VERIFIED badges
- [ ] Copy entire JSON array output
- [ ] Filter for examples with "confidence": "high"
- [ ] Save to file: `agent_60.json`
- [ ] Verify: File contains 60+ examples (filtered)

---

## STEP 6: AGENT SWARM MODE (7 minutes)
- [ ] Open PROMPT_4_SWARM_MODE.txt
- [ ] Top-right dropdown: Select **"AGENT SWARM"** mode
- [ ] Copy entire prompt content
- [ ] Paste into K2.5 chat
- [ ] Send
- [ ] Wait 5-7 minutes (parallel agents working)
- [ ] Watch as 8 agents execute in parallel
- [ ] Wait for Agent 7 & 8 verification
- [ ] Copy entire JSON output
- [ ] Verify structure includes "examples_by_domain"
- [ ] Check "verification_summary" shows 98%+ accuracy
- [ ] Save to file: `swarm_100.json`

---

## STEP 7: CONSOLIDATION (2 minutes)
- [ ] Verify all JSON files saved in same directory:
  - [ ] thinking_30.json
  - [ ] instant_50_batch1.json
  - [ ] instant_50_batch2.json
  - [ ] instant_50_batch3.json
  - [ ] agent_60.json
  - [ ] swarm_100.json
- [ ] Open Python console/terminal
- [ ] Run: `python consolidation_script.py`
- [ ] Wait for completion message
- [ ] Verify: `kimi_k2_5_master_340_examples.json` created

---

## POST-EXECUTION VALIDATION
- [ ] All JSON files valid (can open in text editor)
- [ ] Master file shows 340 total examples
- [ ] Examples span all 6 domains
- [ ] Difficulty mix: easy, medium, hard
- [ ] Agent/Swarm examples marked as verified
- [ ] No JSON syntax errors

---

## SUCCESS INDICATORS ✅
When complete, you should have:
- ✓ 340 total examples
- ✓ 30 Thinking mode examples
- ✓ 150 Instant mode examples
- ✓ 60 Agent mode examples
- ✓ 100 Swarm mode examples
- ✓ Master JSON file created
- ✓ All examples across 6 domains
- ✓ Expected K2.5 knowledge transfer: 95-97%

---

## NEXT STEPS
1. Use master dataset for student model training
2. Few-shot prompting: Add examples to prompts
3. Soft prompt tuning: Train 0.1% of parameters
4. Ensemble inference: K2.5 + student verification

**Total Time: ~20 minutes**
**Result: 340 production-ready examples**

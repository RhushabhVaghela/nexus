# Tightening Phase Plan

## Overview
Finish the "Tightening" phase by updating documentation and configuration to match the new "Activation Anchoring" terminology and "Tiered" capability structure.

## Project Type
**TYPE:** `BACKEND` (Documentation & Configuration)

## Success Criteria
1. `retention_contracts.md` reflects the new Tiered Table structure.
2. `nexus-implementation.md` uses "Activation Anchoring" instead of "Gradient Surgery".
3. `manifest.json` includes the new `capabilities` schema with Tiers.

## Tech Stack
- Markdown
- JSON

## Task Breakdown

### Task 1: Update Retention Contracts
- **Goal:** Replace legacy capability tables with the new Tiered structure.
- **Input:** `retention_contracts.md` lines 9-29.
- **Output:** Updated file with Tier 1/2/3 tables.
- **Verify:** Read file and check for "Tier 1: Text & Reasoning".

### Task 2: Update Implementation Doc
- **Goal:** Global replace of obsolete terminology.
- **Input:** `nexus-implementation.md`.
- **Action:** Replace "Gradient Surgery" with "Activation Anchoring".
- **Verify:** `grep "Gradient Surgery"` returns 0 matches.

### Task 3: Update Manifest
- **Goal:** Add capability tiers to the bundle manifest.
- **Input:** `dist/nexus_bundle_v1/manifest.json`.
- **Action:** Insert `capabilities` object with tiers.
- **Verify:** `cat manifest.json` shows "tiers" key.

## Phase X: Verification Checklist
- [ ] `retention_contracts.md` has Tiered tables.
- [ ] `nexus-implementation.md` has NO "Gradient Surgery".
- [ ] `manifest.json` has `capabilities` with Tiers.

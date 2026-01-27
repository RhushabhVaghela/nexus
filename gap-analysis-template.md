# Gap Analysis: [New Feature Set Name]

**Objective:** Map new feature requirements to the existing Nexus Implementation Roadmap to ensure strategic integration without disrupting invariants.
**Base Roadmap:** `implementation_roadmap.md`
**Status:** Draft

---

## 1. Feature Inventory
*List the incoming new features or requirements below. Assign an ID to each.*

| ID | Feature Name | Description | Source |
|----|--------------|-------------|--------|
| F-01 | [Feature Name] | [Description] | [User Request/Doc] |
| F-02 | ... | ... | ... |

---

## 2. Categorization Matrix
*Map each feature to its integration strategy.*

| Feature ID | Category | Logic/Rationale |
|------------|----------|-----------------|
| F-01 | [Immediate / New Stage / Extension] | ... |

**Categories:**
*   **Immediate Updates:** Modifications to existing stages (e.g., tweaking Loss Function, changing config).
*   **New Stages:** Entirely new pipeline steps (e.g., Pre-training phase, new data processing step).
*   **Extensions:** Post-V1 features to be backlogged.

---

## 3. Integration Plan

### 3.1 Immediate Updates (Modifications)
*Updates to existing `implementation_roadmap.md` tasks.*

#### Stage [X]: [Stage Name]
*   **Task [X.Y]:** [Existing Task Name]
    *   **Current:** [Current definition]
    *   **Modification:** [New definition/constraint]
    *   **Action:** Update task description/subtasks.

### 3.2 New Stages (Additions)
*New stages to be inserted into the pipeline.*

#### **NEW** Stage [X.5]: [New Stage Name]
*   **Insertion Point:** After Stage [X] and before Stage [Y]
*   **Goal:** [Measurable goal]
*   **Tasks:**
    *   [ ] Task [X.5.1]: [Task Name]
    *   [ ] Task [X.5.2]: [Task Name]

### 3.3 Extensions (Post-V1)
*Features deferred to V2 or later.*

*   **Feature:** [Name]
*   **Reason:** [Why it's not V1]
*   **Location:** Added to `backlog.md` / `future_roadmap.md`

---

## 4. Conflict & Invariant Check
*Verify that changes do not violate Global Invariants.*

- [ ] **Pure Distillation Only:** Do new features require teachers at inference?
- [ ] **Hardware Constraints:** Does this push VRAM > 16GB?
- [ ] **Storage:** Does this require storing raw activations?

---

## 5. Execution Actions
1.  [ ] Apply "Immediate Updates" to `implementation_roadmap.md`.
2.  [ ] Insert "New Stages" into `implementation_roadmap.md`.
3.  [ ] Create specific plan files for complex new stages (e.g., `pretraining-plan.md`).

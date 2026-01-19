# Advanced Dataset Generators Plan

To create a state-of-the-art "Replica" model (Application Specialist), we need to close the gap between "knowing code" and "building products".

## 1. ✅ The "Coder" (Script 01 - Running)

- **Role**: Execute commands, write React/Node code, fix errors.
- **Status**: Running (200M samples).

## 2. ✅ The "Instruction Follower" (Script 03 - Running)

- **Role**: Follow repetitive/tedious queries accurately.
- **Status**: Running (200M samples, 50+ generators).

## 3. 🧠 The "Architect" (Script 11 - New)

- **Role**: High-level reasoning, stack selection, database schema design.
- **Input**: "I need a scalable CRM."
- **Output**: `<think>Requirement analysis...</think>` + Blueprint (Next.js, Postgres, etc.).
- **Value**: Teaches the "Why" behind the code.

## 4. 🧪 The "QA Engineer" (Script 12 - Proposed)

- **Role**: Generate test cases, identify edge cases, security audit.
- **Input**: "Here is a React form component."
- **Output**: "Here are 5 unit tests (Jest) and 3 security vulnerabilities (XSS) to fix."
- **Value**: Replica apps must be production-ready, not just prototypes.

## 5. 🎨 The "UI/UX Designer" (Script 13 - Proposed)

- **Role**: Translate abstract requirements into concrete UI components (Tailwind classes).
- **Input**: "Make it look like a modern SaaS dashboard."
- **Output**: "Use Sidebar navigation, Card widgets with `shadow-sm border-gray-200`, and `text-slate-900` typography."
- **Value**: Code is useless if it looks ugly. This teaches "aesthetic intelligence".

## 6. 🛠️ The "DevOps Engineer" (Script 14 - Proposed)

- **Role**: Deployment, CI/CD, Dockerization.
- **Input**: "Deploy this Next.js app to AWS."
- **Output**: Dockerfile + GitHub Actions YAML + Terraform/CDK.
- **Value**: The "Last Mile" of app creation.

---

## 🏗️ Implementation Plan

I will implement these in order, starting with the **Architect** (Script 11) as requested.

### Shared Infrastructure

- All new generators will use `utils/logging_config.py`.
- All will output to `E:/[dataset_name]`.
- All will use `DeduplicatedGenerator` for 0 redundancy.
- All will use **Native Schema** (OpenAI tool calling format where applicable).

# Deep Research: Fullstack "Replica" Architectures

## Objective

To generate synthetic training data that mimics the creation of high-end AI development platforms (Manus, Lovable, Replit). The goal is to train a model that can "replicate" these experiences locally.

## 1. Manus (Advanced Agentic IDE)

**Core Philosophy**: "Autonomous execution with human-in-the-loop oversight."
**Key Features**:

- **Artifact System**: Creating standalone documents (plans, code, diffs) distinct from chat.
- **Environment**: Linux-based container (sandboxed).
- **Tooling**: Bash, Python, Browser, File Editing.

**Replica Schema for Trajectory Generation**:

- **User Prompt**: "Build a Manus-like agentic interface where I can chat and seeing code generated in real-time."
- **Tech Stack**:
  - **Frontend**: React (Vite) + Tailwind CSS + Radix UI (for polished accessibility).
  - **State Management**: Zustand or Jotai (granular updates).
  - **Backend**: Python (FastAPI) or Node.js (WebSocket server) to stream tool outputs.
  - **Sandboxing**: Docker containers via `docker-py` or WebVM (WASM).
- **Key Components**:
  - `ChatPanel.tsx`: Streaming tokens, rendering markdown.
  - `ArtifactView.tsx`: Tabbed interface for code/preview/terminal.
  - `AgentOrchestrator.ts`: Handling tool calling loops.

## 2. Lovable (Text-to-Fullstack App)

**Core Philosophy**: "production-ready React apps from prompts."
**Key Features**:

- **Design Systems**: Shadcn/UI integration by default.
- **Database**: Supabase (Postgres) auth & realtime.
- **Visual Editing**: "Click to edit" mixed with AI prompts.

**Replica Schema for Trajectory Generation**:

- **User Prompt**: "Create a Lovable clone that generates React apps with visual tweaking."
- **Tech Stack**:
  - **Frontend**: Next.js 14 (App Router).
  - **UI Library**: Shadcn/UI + Lucide Icons.
  - **Database**: Supabase (Auth, Row Level Security).
  - **AI Integration**: Vercel AI SDK for streaming component generation.
- **Key Components**:
  - `PromptInput.tsx`: Large, central input with suggestions.
  - `LivePreview.tsx`: Iframe or isolated renderer for generated code.
  - `PropertyEditor.tsx`: Side panel to tweak colors/spacing manually.

## 3. Replit (Cloud IDE & Ghostwriter)

**Core Philosophy**: "Instant dev environment."
**Key Features**:

- **Multiplayer**: OT/CRDTs for real-time collaboration.
- **Universal Runtime**: Nix-based environments.
- **Deployment**: One-click hosted apps.

**Replica Schema for Trajectory Generation**:

- **User Prompt**: "Build a collaborative cloud IDE with AI assistance."
- **Tech Stack**:
  - **Frontend**: React + XTerm.js (terminal) + Monaco Editor.
  - **Backend**: Go or Rust (performance) for WebSocket handling.
  - **Collaboration**: Yjs (CRDT library) + WebSockets.
  - **Infrastructure**: Kubernetes/Firecracker microVMs.
- **Key Components**:
  - `EditorWorkspace.tsx`: Split panes (Code, Preview, Console).
  - `CollaboratorCursor.tsx`: Rendering remote user positions.
  - `RunnerService.ts`: Managing ephemeral execution environments.

## Integration Plan for `02_generate_trajectories.py`

We will replace the simple generic `generate_fullstack_steps` with a `ReplicaGenerator` class that:

1. **Selects a Archetype**: (Manus, Lovable, Replit, or Custom SaaS).
2. **Generates Specific Steps**:
    - *Planning*: "Analyzing requirements for [Archetype]..."
    - *Setup*: "Initializing [Stack]..."
    - *Component Construction*: "Creating [KeyComponent] with [Library]..."
    - *Integration*: "Connecting [Frontend] to [Backend]..."
    - *Refinement*: "Adding [Feature] like [ReferenceApp]..."
3. **Simulates "Deep Research"**:
    - The model will "search" for "latest shadcn docs" or "Supabase auth patterns" as part of the trajectory.

## 4. Coding Capabilities & Logic Evaluation

We conducted a comprehensive "needle-in-a-haystack" style evaluation of coding capabilities by prompting the model with highly specific, advanced constraints in various domains. The model `/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4` demonstrated exceptional proficiency.

### 4.1 Systems & Low-Level Programming

- **CUDA (C++)**: Correctly implemented a Tiled Matrix Multiplication kernel using shared memory to minimize global memory traffic. Included proper synchronization (`__syncthreads()`) and error checking.
- **Rust**: Implemented an `async` TCP server with `tokio`, custom `Stream` implementation for parsing, and zero-copy parsing techniques.
- **x86-64 Assembly**: Produced correct NASM syntax for AES-256 CBC encryption using AES-NI instructions (`AESENC`, `AESDEC`) and implemented constant-time comparison to prevent timing attacks.
- **WebAssembly (WAT)**: Successfully wrote raw WAT text format for an image convolution filter using the SIMD proposal (`v128`) and bulk memory operations.

### 4.2 Modern Backend & Infrastructure

- **Spring Boot 3 (Java)**: Correctly configured Virtual Threads (Project Loom) for high concurrency, GraalVM native image compatibility, and R2DBC for reactive database interactions.
- **.NET 8 (C#)**: Implemented an Event Sourcing system using EventStoreDB, MediatR with source generators (crucial for AOT), and SignalR. Correctly used C# 12 primary constructors.
- **Elixir (Phoenix LiveView)**: Created a real-time collaborative editor using CRDTs (RGA algorithm), binary WebSocket encoding for performance, and proper OTP supervision trees.

### 4.3 Frontend & Mobile

- **React**: Implemented a virtualized infinite scroll list with `IntersectionObserver`, generic types for TypeScript components, and sophisticated state management.
- **iOS (SwiftUI)**: Built a health dashboard using pure SwiftUI, `HealthKit` integration, `CoreData` with `CloudKit` syncing, and `Combine` for reactive updates.
- **Android (Kotlin)**: Developed a background music player service using `MediaSessionCompat` and Hilt for dependency injection, adhering to Modern Android Development (MAD) guidelines.

### 4.4 Data Science & Database

- **PyTorch**: Implemented a Vision Transformer (ViT) from scratch, including `FlashAttention` logic and Mixed Precision training (`autocast`).
- **PostgreSQL**: Wrote complex Recursive CTEs for hierarchical data, recursive GIN index strategies for JSONB, and Row-Level Security (RLS) policies.
- **Julia**: Solved ODEs with adaptive step sizes and event detection, showcasing idiomatic usage of the `DifferentialEquations.jl` ecosystem patterns (even if implemented from scratch).

### 4.5 Graphics & Game Dev

- **GLSL Shaders**: Implemented a full PBR rendering pipeline including Cook-Torrance BRDF, IBL with HDR cubemaps, SSAO, Cascaded Shadow Maps, and TAA.
- **Volumetric Fog**: Correctly implemented ray-marching based fog in a fragment shader.

### 4.6 Prompt Engineering

- **Prompt Repetition**: Validated technique (arXiv 2512.14982) where duplicating the input query (`<QUERY><QUERY>`) approximates bidirectional attention, significantly improving performance on lookup/retrieval tasks.

### Conclusion on Coding Logic

The model avoids "tutorial-level" code and defaults to production-grade patterns (e.g., Dependency Injection, AOT correctness, SIMD usage, Async flows). It handles conflicting constraints well (e.g., "High concurrency" + "Native Image").

### 4.5 Multimodal Capabilities

- **Status**: Ready for Testing
- **Test Suite**: `multimodal_test_prompts.json` (7 scenarios)
- **Coverage**:
  - **Vision**: UI Screenshot to Code, Architecture Diagram to Terraform, Whiteboard to Python.
  - **Video**: Bug Reproduction Analysis, User Flow Mapping.
  - **Audio**: Code Review Transcription, Sprint Planning Summary.

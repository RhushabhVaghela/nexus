# GPT-OSS-20B Multimodal Capability Test Prompts

These prompts are designed to evaluate the model's ability to process non-text inputs (Image, Audio, Video) and generate code or technical analysis.

## 🖼️ Vision (Image-to-Code)

### 1. UI Screenshot to React Component

**Input**: `assets/test_images/dashboard_ui.png` (A dashboard with charts, sidebar, and profile)
**Prompt**:

```
Analyze this UI screenshot.
1. Identify the color palette (hex codes).
2. List the breakdown of components (Sidebar, Header, MainContent, Cards).
3. Generate a pixel-perfect React component using Tailwind CSS that replicates this design.
4. Use Lucide-React for icons where appropriate.
```

### 2. Architecture Diagram to Terraform

**Input**: `assets/test_images/aws_architecture.png` (Diagram showing ALB, EC2 Auto Scaling, RDS, and S3)
**Prompt**:

```
Based on this architecture diagram:
1. Identify the AWS services and their relationships.
2. Write a complete Terraform configuration (`main.tf`) to deploy this infrastructure.
3. Include security groups allowing HTTP/HTTPS to ALB and specific ports to EC2/RDS.
4. Use variables for region and instance types.
```

### 3. Whiteboard Algorithm to Python

**Input**: `assets/test_images/binary_tree_whiteboard.jpg` (Hand-drawn binary tree inversion logic)
**Prompt**:

```
Transcribe the algorithm shown on this whiteboard.
1. Explain the logic in plain text.
2. Implement the solution in Python 3.
3. specific any edge cases noted in the drawing (e.g., null nodes).
```

---

## 📹 Video (Video-to-Code/Text)

### 4. Bug Reproduction Analysis

**Input**: `assets/test_videos/glitch_repro.mp4` (Screen recording showing a UI flickering issue on button hover)
**Prompt**:

```
Watch this screen recording of a bug.
1. Describe the exact sequence of events leading to the glitch.
2. Hypothesize the root cause (e.g., race condition, CSS z-index, state update loop).
3. Suggest a fix for a React application.
```

### 5. Mobile App User Flow

**Input**: `assets/test_videos/checkout_flow.mp4` (User navigating through a checkout process)
**Prompt**:

```
Analyze the user flow in this video.
1. Map out the screens (Cart -> Address -> Payment -> Success).
2. Generate a `React Navigation` stack navigator configuration that matches this flow.
3. List the necessary API calls inferred from the loading states shown.
```

---

## 🎙️ Audio (Audio-to-Code)

### 6. Code Review Transcription

**Input**: `assets/test_audio/pr_review_comments.mp3` (Audio file of a senior engineer giving feedback)
**Prompt**:

```
Listen to this code review code recording.
1. Transcribe the feedback points regarding the `AuthService` class.
2. Generate a `diff` or applying the requested changes to the `login` function.
3. Specifically address the security concern mentioned about token storage.
```

### 7. Sprint Meeting Summary to Tickets

**Input**: `assets/test_audio/sprint_planning.wav` (Team discussing features for the next sprint)
**Prompt**:

```
Extract user stories from this sprint planning meeting.
1. Create a JSON list of tickets with fields: `title`, `description`, `priority`, `estimated_points`.
2. Flag any dependencies mentioned between the "User Profile" and "Notification" teams.
```

---

## 📊 Scoring Guide

| Score | Criteria |
|-------|----------|
| ✅ **Pass** | Correctly identifies visual/audio elements, generates matching code |
| ⚠️ **Partial** | Misses details (e.g., wrong color), code works but doesn't match input |
| ❌ **Fail** | Hallucinates elements not in input, fails to generate code |

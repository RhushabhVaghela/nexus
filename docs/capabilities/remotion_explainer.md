# Nexus Universal Explainer: 3Blue1Brown-Style Video Generation

The **Nexus Universal Explainer** is a state-of-the-art capability that allows the Nexus model to generate professional educational videos inspired by the 3Blue1Brown aesthetic. It combines the reasoning power of Large Language Models with the programmatic precision of [Remotion](https://www.remotion.dev/).

---

## 🌟 Key Features

- **Programmatic Precision:** Generates exact React/TypeScript code for animations, ensuring frame-perfect timing.
- **Mathematical Rendering:** Native support for LaTeX formulas with fluid "writing" animations.
- **Dynamic 2D/3D Math:** Animated function plotting and Three.js-powered 3D vector spaces.
- **Image-to-Video:** Annotate and animate labels onto existing diagrams or photos.
- **Voiceover Synthesis:** Integrated audio narration that automatically syncs with the visual timeline.
- **RTX 5080 Optimized:** High-performance rendering pipeline designed for modern NVIDIA hardware.

---

## 🏗️ Architecture

The system operates as a **Hybrid Programmatic-Neural Pipeline**:

1.  **Inference (Python):** The model receives a conceptual prompt and generates a specialized system-prompt-aware TSX file.
2.  **Asset Management:** The system fuzzy-searches local datasets (`/mnt/e/data`) and the web for necessary visual/audio assets.
3.  **Rendering (Node.js):** The generated code is injected into the Remotion project and rendered using Puppeteer/Chrome Headless Shell.
4.  **Audio Sync:** Narration is synthesized using the model's audio head and saved as a WAV file, which the Remotion project automatically imports.

---

## 📚 NexusLib Component Library

The model is trained to speak a specialized "Visual Grammar" using these primitives:

### 1. `NexusMath` (LaTeX)
Animated formulas using KaTeX and framer-motion.
```tsx
<NexusMath latex="e^{i\pi} + 1 = 0" fontSize={80} color="#00f2ff" />
```

### 2. `NexusGraph` (2D Plots)
Dynamic function plotting with animated path drawing.
```tsx
<NexusGraph fn={(x) => Math.sin(x)} xRange={[-10, 10]} color="#ff0055" />
```

### 3. `Nexus3D` (Three.js Math)
3D coordinate systems and vector visualizations.
```tsx
<Nexus3D vectors={[{x: 1, y: 2, z: 3, color: '#00f2ff'}]} showGrid={true} />
```

### 4. `NexusFlow` (Flowcharts)
Declarative node-and-edge animations for system logic.
```tsx
<NexusFlow nodes={nodes} edges={edges} color="#00ff88" />
```

### 5. `NexusAnnotator` (Image Overlays)
Place animated pulses and labels on specific parts of an image.
```tsx
<NexusAnnotator src="cell.png" annotations={[{x: 40, y: 50, text: "Nucleus"}]} />
```

### 6. `NexusAudio` (Narration)
Handles background music and synchronized voiceovers.
```tsx
<NexusAudio src="narration.wav" volume={1.0} />
```

### 7. `NexusTimeline` (History)
Scrolling timeline for historical events.
```tsx
<NexusTimeline events={[{year: "1969", title: "Moon Landing"}]} />
```

### 8. `NexusChart` (Business)
Animated Bar/Pie charts for data visualization.
```tsx
<NexusChart data={[{label: "Q1", value: 100}]} type="bar" />
```

### 9. `NexusMap` (Geography)
Stylized world maps with coordinate markers.
```tsx
<NexusMap markers={[{lat: 40.7, lng: -74.0, label: "NYC"}]} />
```

### 10. `NexusScene/Character` (Story)
Narrative scene composition with characters and dialogue.
```tsx
<NexusScene background="space.jpg"><NexusCharacter src="hero.png" /></NexusScene>
```

### 11. `NexusList` (Lifestyle)
Top 10 lists, recipes, and checklists.
```tsx
<NexusList items={[{title: "Step 1"}]} type="number" />
```

---

## 📂 Training & Data

### The Remotion-1M Dataset
We have synthesized a dataset of **1,000,000 samples** located at:
`/mnt/e/data/datasets/remotion/remotion_explainer_dataset.jsonl`

**Dataset Categories:**
- **Science:** Math, Graph, 3D, Annotator
- **Business:** Chart, Flow (Funnels)
- **Humanities:** Timeline, Map
- **Creative:** Story, Audio
- **General:** Lifestyle (Lists)

### Dataset Customization
You can regenerate the dataset with custom weights using the `--category-weights` parameter. Unassigned categories will split the remaining percentage.

```bash
# Example: 50% Math, 20% Story, 30% split among others
python src/utils/generate_remotion_dataset.py --category-weights math=50 story=20
```

### Training Command
```bash
./run_universal_pipeline.sh \
    --base-model=/path/to/base \
    --enable-remotion-explainer \
    --batch-size=2 \
    --epochs=1
```

---

## 🖥️ Usage

### 1. The Explainer CLI
Generate a video from the terminal:
```bash
python nexus_explain.py "Explain the Fourier Transform using rotating circles" --narrate
```

### 2. The Interactive Dashboard
A web-based studio for real-time iteration.

**Start Backend:**
```bash
python src/api/explainer_api.py
```

**Start Frontend:**
```bash
cd dashboard
npm run dev
```

---

## 🛠️ Hardware Requirements

- **GPU:** NVIDIA RTX 5080 (16GB VRAM) or equivalent.
- **RAM:** 32GB Minimum.
- **Storage:** 20GB for dataset, 1GB for Node environment.
- **Software:** Node.js 18+, Python 3.10+, Chrome/Chromium (for headless rendering).

import os
import subprocess
import re
import logging
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from typing import Optional, Dict, Any

from src.omni.inference import OmniInference, GenerationConfig
from src.capability_registry import REMOTION_EXPLAINER_SYSTEM_PROMPT
from src.utils.asset_manager import AssetManager

logger = logging.getLogger(__name__)

class RemotionExplainerEngine:
    """
    Engine for generating and rendering Remotion-based educational videos.
    """
    
    def __init__(
        self,
        model_path: str,
        remotion_dir: str = "remotion",
        data_root: str = "/mnt/e/data",
    ):
        self.remotion_dir = Path(remotion_dir)
        self.asset_manager = AssetManager(data_root=data_root, remotion_public=str(self.remotion_dir / "public"))
        # Enable audio if we want narration support
        self.inference = OmniInference(model_path, enable_audio=True)
        
    def generate_video(
        self,
        prompt: str,
        output_name: str = "explanation.mp4",
        narrate: bool = False
    ) -> str:
        """
        Generate and render a video from a prompt.
        """
        # 1. Prepare messages
        messages = [
            {"role": "system", "content": REMOTION_EXPLAINER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # 2. Generate content
        logger.info(f"Generating Remotion TSX code (narrate={narrate})...")
        if narrate:
            result = self.inference.chat_with_audio(messages, GenerationConfig(max_new_tokens=2048))
            raw_output = result["text"]
            audio_data = result["audio"]
            
            if audio_data is not None:
                self._save_narration(audio_data, "narration.wav")
                logger.info("Narration audio synthesized and saved.")
        else:
            raw_output = self.inference.chat(messages, GenerationConfig(max_new_tokens=2048))
        
        # 3. Post-process (extract TSX)
        tsx_code = self._extract_tsx(raw_output)
        if not tsx_code:
            logger.error("Failed to extract valid TSX from model output.")
            raise ValueError("Invalid model output: No TSX found.")
            
        # Ensure narration.wav is used if narrated
        if narrate and "narration.wav" not in tsx_code:
            # Inject NexusAudio if missing but narration exists
            if "<NexusAudio" not in tsx_code:
                if "return (" in tsx_code:
                    tsx_code = tsx_code.replace("return (", "return (\n    <NexusAudio src='narration.wav' />")
                elif "=> (" in tsx_code:
                    tsx_code = tsx_code.replace("=> (", "=> (\n    <NexusAudio src='narration.wav' />")
        
        # 4. Save to Remotion project
        test_scene_path = self.remotion_dir / "src" / "GeneratedScene.tsx"
        with open(test_scene_path, 'w') as f:
            f.write(tsx_code)
            
        # 5. Update Root.tsx temporarily
        root_path = self.remotion_dir / "src" / "Root.tsx"
        with open(root_path, 'r') as f:
            original_root = f.read()
            
        # Register the new scene
        new_root = original_root.replace(
            "import { NexusGraph } from './NexusLib/NexusGraph';",
            "import { NexusGraph } from './NexusLib/NexusGraph';\nimport { Scene as GeneratedScene } from './GeneratedScene';"
        )
        new_root = new_root.replace(
            "</>\n  );",
            f'  <Composition id="GeneratedVideo" component={{GeneratedScene}} durationInFrames={{300}} fps={{30}} width={{1920}} height={{1080}} />\n    </>\n  );'
        )
        
        try:
            with open(root_path, 'w') as f:
                f.write(new_root)
                
            # 6. Render
            logger.info(f"Rendering {output_name}...")
            env = os.environ.copy()
            env["PATH"] = f"/home/rhushabh/miniconda3/bin:{env['PATH']}"
            
            cmd = ["npx", "remotion", "render", "src/index.ts", "GeneratedVideo", f"out/{output_name}"]
            result = subprocess.run(
                cmd,
                cwd=self.remotion_dir,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                logger.error(f"Render failed: {result.stderr}")
                raise RuntimeError(f"Remotion render failed: {result.stderr}")
                
            return str(self.remotion_dir / "out" / output_name)
            
        finally:
            # Restore Root.tsx
            with open(root_path, 'w') as f:
                f.write(original_root)

    def _save_narration(self, audio_data: Any, filename: str):
        """Save raw audio data to the Remotion public folder."""
        public_path = self.remotion_dir / "public" / filename
        public_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if it's a torch tensor
        if hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()
            
        # Ensure it's a 1D array
        audio_data = audio_data.flatten()
        
        # Normalize if necessary (assuming it's float [-1, 1])
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
            
        wavfile.write(public_path, 24000, audio_data) # Omni models usually use 24kHz

    def _extract_tsx(self, text: str) -> Optional[str]:
        """Extract TSX code from markdown blocks."""
        match = re.search(r"```(?:tsx|typescript|jsx|javascript)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback if no markdown blocks
        if "import React" in text:
            return text.strip()
        return None

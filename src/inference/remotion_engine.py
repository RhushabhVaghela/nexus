import os
import subprocess
import re
import logging
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
        self.inference = OmniInference(model_path)
        
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
        
        # 2. Generate TSX
        logger.info("Generating Remotion TSX code...")
        raw_output = self.inference.chat(messages, GenerationConfig(max_new_tokens=2048))
        
        # 3. Post-process (extract TSX)
        tsx_code = self._extract_tsx(raw_output)
        if not tsx_code:
            logger.error("Failed to extract valid TSX from model output.")
            raise ValueError("Invalid model output: No TSX found.")
            
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

    def _extract_tsx(self, text: str) -> Optional[str]:
        """Extract TSX code from markdown blocks."""
        match = re.search(r"```(?:tsx|typescript|jsx|javascript)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback if no markdown blocks
        if "import React" in text:
            return text.strip()
        return None

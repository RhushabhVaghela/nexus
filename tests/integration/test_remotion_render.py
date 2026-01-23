import pytest
import sys
import os
import json
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestRemotionRender:
    """End-to-end render verification."""
    
    @pytest.fixture
    def remotion_dir(self):
        return Path(__file__).parent.parent.parent / "remotion"
        
    def test_render_sample(self, remotion_dir, tmp_path):
        """Pick a random sample from the dataset and try to render it."""
        dataset_path = Path("/mnt/e/data/datasets/remotion/remotion_explainer_dataset.jsonl")
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
            
        # Read first 10 lines and pick one
        with open(dataset_path, 'r') as f:
            lines = [f.readline() for _ in range(10)]
            sample = json.loads(lines[5]) # Pick the 6th one for variety
            
        tsx_content = sample["output"]
        
        # Save this to a temp component in the remotion project
        test_scene_path = remotion_dir / "src" / "TestScene.tsx"
        root_path = remotion_dir / "src" / "Root.tsx"
        original_root = ""
        
        # We need to adapt the output slightly for a direct render test
        # The dataset output is designed to be imported or used as a snippet
        # For a full render, we need to make it the default export or registered
        
        try:
            with open(test_scene_path, 'w') as f:
                f.write(tsx_content.replace("export const Scene", "export const TestScene"))
            
            # Update Root.tsx temporarily to include this scene
            with open(root_path, 'r') as f:
                original_root = f.read()
                
            new_root = original_root.replace(
                "</>\n  );",
                f'  <Composition id="TestRender" component={{TestScene}} durationInFrames={{30}} fps={{30}} width={{1920}} height={{1080}} />\n    </>\n  );'
            )
            new_root = "import { TestScene } from './TestScene';\n" + new_root
            
            with open(root_path, 'w') as f:
                f.write(new_root)
                
            # Attempt a 1-frame render (still image) to verify compilation
            env = os.environ.copy()
            env["PATH"] = f"/home/rhushabh/miniconda3/bin:{env['PATH']}"
            
            # Just render the first frame as a check
            cmd = ["npx", "remotion", "still", "src/index.ts", "TestRender", "out/test.png", "--frame=0"]
            
            result = subprocess.run(
                cmd, 
                cwd=remotion_dir, 
                capture_output=True, 
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
            assert result.returncode == 0
            assert (remotion_dir / "out" / "test.png").exists()
            
        finally:
            # Cleanup
            if test_scene_path.exists():
                test_scene_path.unlink()
            # Restore Root.tsx
            with open(root_path, 'w') as f:
                f.write(original_root)

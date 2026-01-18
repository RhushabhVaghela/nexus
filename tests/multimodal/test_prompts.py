
import unittest
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from multimodal.tests import get_test_prompts

class TestMultimodalPrompts(unittest.TestCase):
    
    def test_get_prompts_structure(self):
        prompts = get_test_prompts()
        
        self.assertIn("vision", prompts)
        self.assertIn("audio", prompts)
        self.assertIn("video", prompts)
        
        # Check vision prompt structure
        vision_item = prompts["vision"][0]
        self.assertIn("id", vision_item)
        self.assertIn("input", vision_item)
        self.assertIn("prompt", vision_item)

    def test_prompt_content(self):
        prompts = get_test_prompts()
        
        # Verify specific known prompt exists
        vision_inputs = [p["input"] for p in prompts["vision"]]
        self.assertIn("assets/test_images/dashboard_ui.png", vision_inputs)

if __name__ == '__main__':
    unittest.main()

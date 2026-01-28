import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from nexus_core.data.sanitizer import UniversalSanitizer

def test_google_smol():
    path = "/mnt/e/data/datasets/general/google_smol/aa_en.jsonl"
    print(f"\n--- Testing google_smol (Translation) ---")
    if not os.path.exists(path):
        print(f"Skipping: {path} not found")
        return
    
    with open(path, 'r') as f:
        import json
        for _ in range(3):
            line = f.readline()
            if not line: break
            item = json.loads(line)
            print(f"Raw Entry: {str(item)[:100]}...")
            sanitized = UniversalSanitizer.sanitize(item)
            print(f"Sanitized: {sanitized}")

def test_code_fragment():
    print(f"\n--- Testing Code Fragment (Manual) ---")
    item = {
        "content": "```python\ndef hello():\n    print('world')\n```",
        "metadata": {"id": 123, "lang": "py"}
    }
    print(f"Raw Entry: {item}")
    sanitized = UniversalSanitizer.sanitize(item)
    print(f"Sanitized: {sanitized}")

def test_nested_messy():
    print(f"\n--- Testing Nested Messy JSON ---")
    item = {
        "val": {
            "inner": {
                "text": " This is the actual content with <html> tags and [brackets]. "
            }
        },
        "garbage": "x" * 50
    }
    print(f"Raw Entry: {str(item)[:100]}...")
    sanitized = UniversalSanitizer.sanitize(item)
    print(f"Sanitized: {sanitized}")

if __name__ == "__main__":
    test_google_smol()
    test_code_fragment()
    test_nested_messy()

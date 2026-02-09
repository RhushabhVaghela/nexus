import re
import json

class UniversalSanitizer:
    """
    Heuristic-based sanitizer to extract clean text from messy dataset fragments.
    Designed to handle raw JSON, code snippets, and metadata-heavy objects
    automatically without per-dataset hardcoding.
    """

    @staticmethod
    def sanitize(item) -> str:
        """
        Main entry point. Converts arbitrary dictionary or string into clean linguistic text.
        """
        if isinstance(item, str):
            return UniversalSanitizer._clean_string(item)
        
        if isinstance(item, dict):
            return UniversalSanitizer._extract_from_dict(item)
        
        return str(item)

    @staticmethod
    def _extract_from_dict(data: dict, depth=0) -> str:
        """
        Heuristically finds the most likely 'content' field in a raw JSON object.
        """
        if depth > 3: # Safety limit
            return ""

        # Primary targets - if found, stop immediately to avoid metadata pollution
        targets = ["text", "content", "instruction", "output", "val", "body", "message", "src", "trgs"]
        for t in targets:
            val = data.get(t)
            if val:
                if isinstance(val, str) and len(val) > 1:
                    return UniversalSanitizer._clean_string(val)
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
                    return UniversalSanitizer._clean_string(" ".join(val))
                if isinstance(val, dict):
                    res = UniversalSanitizer._extract_from_dict(val, depth + 1)
                    if res: return res

        # Heuristic: Find longest string that doesn't look like an ID, path, or garbage
        candidate_val = ""
        candidate_len = 0
        
        all_strings = []
        for k, v in data.items():
            content_v = ""
            if isinstance(v, str):
                content_v = v
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                content_v = " ".join(v)
            elif isinstance(v, dict):
                content_v = UniversalSanitizer._extract_from_dict(v, depth + 1)

            if content_v:
                clean_v = UniversalSanitizer._clean_string(content_v)
                
                # Garbage check: is it just one character repeated?
                if len(set(clean_v.lower())) < 5 and len(clean_v) > 20:
                    continue

                all_strings.append(clean_v)
                
                if len(clean_v) > 10 and len(clean_v) > candidate_len:
                    if not re.search(r"/[a-z0-9_-]+/", clean_v, re.I): # Simple path check
                        candidate_val = clean_v
                        candidate_len = len(clean_v)

        if candidate_val:
            return candidate_val

        # Fallback: Join all string fragments found
        if all_strings:
            return " ".join(all_strings[:5])

        return str(data)

    @staticmethod
    def _clean_string(text: str) -> str:
        """
        Strips common data noise:
        - HTML/XML tags
        - Trailing JSON braces
        - Markdown code block delimiters
        - Excessive whitespace
        """
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Strip markdown markers
        text = text.replace("```json", "").replace("```", "")
        
        # Strip trailing JSON noise if accidentally captured
        text = re.sub(r'^[\[\{\s]+', '', text)
        text = re.sub(r'[\s\}\]]+$', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length to avoid OOM or huge prompt windows during profiling
        return text[:1024]

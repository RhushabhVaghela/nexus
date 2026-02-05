import os
import shutil
import requests
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class AssetManager:
    """Manages local and web assets for Remotion videos."""
    
    def __init__(self, data_root: str = "/mnt/e/data", remotion_public: str = "remotion/public"):
        self.data_root = Path(data_root)
        self.remotion_public = Path(remotion_public)
        self.remotion_public.mkdir(parents=True, exist_ok=True)

    def find_local_asset(self, query: str) -> Optional[str]:
        """Search for a local image or audio matching the query."""
        # Simple fuzzy search in data directory
        patterns = [
            f"**/*{query}*.png", f"**/*{query}*.jpg", f"**/*{query}*.svg",
            f"**/*{query}*.mp3", f"**/*{query}*.wav"
        ]
        for pattern in patterns:
            try:
                for path in self.data_root.glob(pattern):
                    # Copy to public folder for Remotion access
                    dest = self.remotion_public / path.name
                    if not dest.exists():
                        shutil.copy(path, dest)
                    return path.name
            except Exception:
                continue
        return None

    def fetch_web_asset(self, query: str) -> Optional[str]:
        """Fetch an asset from the web (Unsplash API placeholder)."""
        logger.info(f"Web search triggered for: {query}")
        # Logic to fetch from a CDN or API
        return f"https://source.unsplash.com/featured/?{query}"

    def resolve_asset(self, query: str) -> str:
        """Try local, then web, then fallback."""
        local = self.find_local_asset(query)
        if local:
            return local
            
        return self.fetch_web_asset(query)

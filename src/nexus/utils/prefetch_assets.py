import json
import os
import requests
import logging
from pathlib import Path
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AssetPreFetcher")

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
PUBLIC_DIR = Path(__file__).parent.parent.parent / "remotion" / "public"

def download_asset(url: str, filename: str = None):
    if not url.startswith("http"):
        return
        
    if not filename:
        filename = url.split("/")[-1]
        
    dest = PUBLIC_DIR / filename
    if dest.exists():
        return # Skip if exists
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(dest, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded: {filename}")
        else:
            logger.warning(f"Failed to fetch {url}: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")

def extract_urls_from_json(data):
    urls = []
    if isinstance(data, dict):
        for v in data.values():
            urls.extend(extract_urls_from_json(v))
    elif isinstance(data, list):
        for item in data:
            urls.extend(extract_urls_from_json(item))
    elif isinstance(data, str) and data.startswith("http"):
        urls.append(data)
    return urls

def prefetch_all():
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]Prefetching assets from Knowledge Base...[/bold green]")
    
    all_urls = []
    
    # 1. Scan KB JSONs
    for file in KB_DIR.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            urls = extract_urls_from_json(data)
            all_urls.extend(urls)
            console.print(f"Found {len(urls)} assets in {file.name}")
            
    # 2. Add Hardcoded Generator Assets
    # (From generate_remotion_dataset.py)
    all_urls.extend([
        "https://raw.githubusercontent.com/remotion-dev/remotion/main/packages/docs/static/img/logo-colored.png",
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/Periodic_table.svg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Human_mitochondrion_diagram.svg"
    ])
    
    # Deduplicate
    unique_urls = list(set(all_urls))
    console.print(f"Downloading {len(unique_urls)} unique assets...")
    
    # 3. Parallel Download
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_asset, unique_urls)
        
    console.print("[bold green]Asset prefetch complete![/bold green]")

if __name__ == "__main__":
    prefetch_all()
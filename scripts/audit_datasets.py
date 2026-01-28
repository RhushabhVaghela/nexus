import os
import sys
import json
import re
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from nexus_core.towers.registry import DATASET_REGISTRY
from nexus_core.data.sanitizer import UniversalSanitizer

def audit_dataset(name, info):
    """
    Scans a single dataset and reports on its cleanliness and structure.
    """
    path = info.get("local_path")
    if not path or not os.path.exists(path):
        return {"status": "missing", "error": "Path not found"}

    try:
        # Improved file discovery (check depth 1)
        files = []
        for root, _, filenames in os.walk(path):
            rel_depth = os.path.relpath(root, path).count(os.sep)
            if rel_depth > 1: continue 
            for f in filenames:
                if f.endswith(('.jsonl', '.json', '.parquet', '.json.gz', '.csv')):
                    files.append(os.path.join(root, f))
            if len(files) > 5: break 

        if not files:
            return {"status": "empty", "error": "No recognizable data files found"}

        sample_item = None
        
        # 1. Quick Peak at JSONL
        jsonl_files = [f for f in files if f.endswith('.jsonl') or f.endswith('.json.gz')]
        if jsonl_files:
            import gzip
            opener = gzip.open if jsonl_files[0].endswith('.gz') else open
            try:
                with opener(jsonl_files[0], 'rt') as f:
                    import json
                    for line in f:
                        if line.strip():
                            sample_item = json.loads(line)
                            break
            except: pass
        
        # 2. Quick Peak at JSON
        if not sample_item:
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        import json
                        content = f.read(50000)
                        # Try to find a JSON object
                        matches = re.findall(r'\{[^\}]+\}', content, re.DOTALL)
                        for m in matches:
                            try:
                                sample_item = json.loads(m)
                                if isinstance(sample_item, dict): break
                            except: continue
                except: pass

        if not sample_item:
            return {"status": "unsupported", "error": f"Found {len(files)} files but could not peak at content"}

        # Audit logic
        raw_str = str(sample_item)
        sanitized = UniversalSanitizer.sanitize(sample_item)
        
        is_fragment = raw_str.strip().startswith("{")
        improvement_ratio = len(sanitized) / len(raw_str) if len(raw_str) > 0 else 1.0
        
        return {
            "status": "active",
            "columns": list(sample_item.keys()) if isinstance(sample_item, dict) else "N/A",
            "fragments_detected": is_fragment,
            "avg_sanitization_ratio": f"{improvement_ratio:.2f}",
            "sample_cleaned": sanitized[:100] + "..."
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    print("="*60)
    print(" NEXUS DATASET AUDIT TOOL (v2 - Ultra Fast)")
    print("="*60)
    print(f"Scanning {len(DATASET_REGISTRY)} datasets in registry...")

    report = {}
    output_file = "dataset_audit_report.json"
    
    # Load existing report if exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try: report = json.load(f)
            except: pass

    count = 0
    for name, info in tqdm(DATASET_REGISTRY.items()):
        if name in report and report[name]["status"] == "active":
            continue
        
        report[name] = audit_dataset(name, info)
        count += 1
        
        # Incremental Save
        if count % 5 == 0:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)

    # Final Save
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Summary Stats
    active = [k for k, v in report.items() if v["status"] == "active"]
    messy = [k for k, v in report.items() if v["status"] == "active" and v["fragments_detected"]]
    missing = [k for k, v in report.items() if v["status"] == "missing"]
    
    print("\n" + "="*60)
    print(" AUDIT SUMMARY")
    print("="*60)
    print(f"Total Registered: {len(DATASET_REGISTRY)}")
    print(f"Accessible:       {len(active)}")
    print(f"Messy/Fragments:  {len(messy)} (Sanitizer will handle these)")
    print(f"Missing (E:?):    {len(missing)}")
    print("="*60)

    # Save detailed report
    output_file = "dataset_audit_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {output_file}")

    if messy:
        print("\nTop Messy Datasets (Future-Proofed by Sanitizer):")
        for m in messy[:5]:
            print(f"- {m}: {report[m]['sample_cleaned']}")

if __name__ == "__main__":
    main()

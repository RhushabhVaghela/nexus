import os
import sys

# Ensure src is in path to import registry
sys.path.append(os.path.join(os.getcwd(), 'src'))

from nexus_core.towers.registry import TEACHER_REGISTRY, DATASET_REGISTRY

def check_registry(registry, name):
    print(f"\nEvaluating {name} Registry...")
    print("=" * 60)
    
    total = len(registry)
    passed = 0
    failed = []
    
    for key, data in registry.items():
        # Check standard path
        main_path = data.get("path")
        local_path = data.get("local_path")
        
        # Determine effective path (priority to local_path if verifying local existence)
        # We check both if available, but primarily we care if *at least one* valid path exists locally
        
        valid = False
        status_msg = ""
        
        # Check local path first (often absolute /mnt/e/...)
        if local_path and os.path.exists(local_path):
            valid = True
            status_msg = f"[✓] Found at local_path: {local_path}"
        # Fallback to main path if it's absolute or exists relative to cwd
        elif main_path and os.path.exists(main_path):
            valid = True
            status_msg = f"[✓] Found at path: {main_path}"
        else:
            # Check secondary storage fallback manually just in case
            secondary_path = os.path.join("/mnt/e/data/models", key) # Approximate
            if os.path.exists(secondary_path):
                 # This would catch models not strictly defined but existing in structure
                 pass

        if valid:
            passed += 1
            print(f"OK  | {key:<40} | {status_msg}")
        else:
            failed.append(key)
            print(f"ERR | {key:<40} | Missing. Path: {main_path}, Local: {local_path}")

    print("-" * 60)
    print(f"Summary for {name}: {passed}/{total} Passed")
    if failed:
        print(f"Failed Items ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    return passed, total, failed

def main():
    print("Starting Nexus Registry Integrity Audit...")
    print(f"CWD: {os.getcwd()}")
    
    m_pass, m_total, m_fail = check_registry(TEACHER_REGISTRY, "TEACHER MODELS")
    d_pass, d_total, d_fail = check_registry(DATASET_REGISTRY, "DATASETS")
    
    print("\n" + "=" * 60)
    print("FINAL AUDIT REPORT")
    print(f"Models:   {m_pass}/{m_total} Ready")
    print(f"Datasets: {d_pass}/{d_total} Ready")
    print("=" * 60)

    if m_fail or d_fail:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

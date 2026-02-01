import csv
import os
import json
from pathlib import Path

CSV_PATH = "new-plan-conversation-files/ModelName-Parameters-Category-BestFeature.csv"
MODELS_DIR = Path("./models")

def load_inventory():
    inventory = []
    
    # Read CSV
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            for row in reader:
                if not row or len(row) < 2:
                    continue
                    
                # Parse row (CSV format seems to be: "Name","Params","Category","Feature")
                # The read output showed lines like: 00002| "AgentCPM-Explore","4B",...
                # So we need to handle potential quotes and splitting correctly if the standard csv module doesn't.
                # However, the file content showed "..." so standard csv should handle it.
                
                name = row[0].strip()
                params = row[1].strip()
                category = row[2].strip()
                feature = row[3].strip() if len(row) > 3 else ""
                
                # Check Local Existence
                local_path = MODELS_DIR / name
                exists = local_path.exists()
                
                inventory.append({
                    "name": name,
                    "params": params,
                    "category": category,
                    "best_feature": feature,
                    "local_path": str(local_path),
                    "exists": exists
                })
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    return inventory

def main():
    print("--- Nexus Inventory Check ---")
    inventory = load_inventory()
    
    print(f"Found {len(inventory)} entries in CSV.")
    
    missing_models = []
    present_models = []
    
    for item in inventory:
        status = "✅ Found" if item["exists"] else "❌ Missing"
        print(f"[{status}] {item['name']} ({item['params']})")
        
        if item["exists"]:
            present_models.append(item)
        else:
            missing_models.append(item)
            
    # Save Report
    report = {
        "summary": {
            "total": len(inventory),
            "present": len(present_models),
            "missing": len(missing_models)
        },
        "details": inventory
    }
    
    with open("inventory_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nReport saved to inventory_report.json")
    if missing_models:
        print(f"\nWARNING: {len(missing_models)} models are missing from ./models/ directory.")

if __name__ == "__main__":
    main()

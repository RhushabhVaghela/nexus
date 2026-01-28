import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/nexus_core/student')))
try:
    from router import SparseIntentRouter
except ImportError:
    from src.nexus_core.student.router import SparseIntentRouter

from torch.utils.data import Dataset, DataLoader

# Path to Real Router Data (Produced by Human Annotation or Heuristics)
DATA_PATH = "data/router_data/labeled_intents.pt"

class RouterDataset(Dataset):
    def __init__(self, data_path=None, towers_dir=None, input_dim=2048, fallback_size=1000):
        self.data_path = data_path
        self.towers_dir = towers_dir
        self.input_dim = input_dim
        self.fallback_size = fallback_size
        self.features = None
        self.labels = None
        
        self._load_data()
        
    def _load_data(self):
        # 1. Try Loading Real Annotated Data
        if self.data_path and os.path.exists(self.data_path):
            print(f"[Router_DS] Loading real data from {self.data_path}...")
            try:
                data = torch.load(self.data_path)
                self.features = data['features']
                self.labels = data['labels']
                print(f"[Router_DS] Loaded {len(self.features)} samples.")
                return
            except Exception as e:
                print(f"[Error] Failed to load path {self.data_path}: {e}")

        # 2. Try Automatic Expert Discovery (Self-Driving Expert Map)
        if self.towers_dir and os.path.exists(self.towers_dir):
            import glob
            tower_dirs = sorted([d for d in glob.glob(os.path.join(self.towers_dir, "*")) if os.path.isdir(d)])
            
            if not tower_dirs:
                print(f"[Router_DS] Memory directory found but no expert sub-folders detected in {self.towers_dir}")
            else:
                print(f"[Router_DS] Discovering Expert Features in {len(tower_dirs)} sub-folders...")
                all_feats = []
                all_labels = []
                for idx, t_dir in enumerate(tower_dirs):
                    shards = glob.glob(os.path.join(t_dir, "*.pt"))
                    if not shards:
                        print(f"  -> [Warn] No .pt shards found for expert: {os.path.basename(t_dir)}")
                        continue
                        
                    # Sample up to 100 shards per tower for balanced training
                    for s_path in shards[:100]:
                        try:
                            shard = torch.load(s_path, map_location="cpu")
                            if "hidden_state" in shard:
                                feat = shard["hidden_state"]
                                if feat.dim() > 1: feat = feat.mean(dim=0)
                                all_feats.append(feat.view(1, -1))
                                all_labels.append(torch.tensor([idx]))
                        except: continue
                
                if all_feats:
                    self.features = torch.cat(all_feats, dim=0)
                    self.labels = torch.cat(all_labels, dim=0)
                    print(f"[Router_DS] Self-Driving Complete: Successfully mapped {len(self.features)} features for {len(tower_dirs)} experts.")
                    return
                else:
                    print("[Router_DS] Found expert folders but failed to extract hidden_state features from shards.")

        # 3. Fallback to Synthetic (Zero-Shot initial state)
        print(f"[Router_DS] Falls back to Synthetic Initializer (Dim: {self.input_dim})")
        print("  - Rationale: No real expert features discovered in memory/. Standard for fresh starts.")
        self.features = torch.randn(self.fallback_size, self.input_dim)
        self.labels = torch.randint(0, 5, (self.fallback_size,))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_router(input_dim=4096, num_towers=5, towers_dir=None, epochs=5, batch_size=32, lr=1e-4):
    print(f"\n[Router] Starting Training (Dim: {input_dim}, Towers: {num_towers}) for {epochs} epochs...")
    if towers_dir:
        print(f"[Router] Memory Scan Path: {towers_dir}")
    
    # 1. Initialize Router
    router = SparseIntentRouter(input_dim=input_dim, num_towers=num_towers, top_k=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router.to(device)
    
    optimizer = optim.Adam(router.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Data Loader
    dataset = RouterDataset(data_path=DATA_PATH, towers_dir=towers_dir, input_dim=input_dim)
    
    # 2.1 Update num_towers if experts were discovered
    if dataset.labels is not None:
        discovered_towers = int(dataset.labels.max().item() + 1)
        if discovered_towers > num_towers:
            print(f"[Router] Detected {discovered_towers} Experts. Updating Router architecture.")
            router = SparseIntentRouter(input_dim=input_dim, num_towers=discovered_towers, top_k=1).to(device)
            optimizer = optim.Adam(router.parameters(), lr=lr)
            num_towers = discovered_towers

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Training Loop
    for epoch in range(epochs):
        router.train()
        total_loss = 0
        steps = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = router.gate(batch_X)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / max(steps, 1)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    # 4. Save
    os.makedirs("results/router_weights", exist_ok=True)
    save_path = "results/router_weights/sparse_router.pt"
    torch.save(router.state_dict(), save_path)
    print(f"[Router] Weights saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=2048) # Default to optimized student dim
    parser.add_argument("--num_towers", type=int, default=5)
    parser.add_argument("--towers_dir", type=str, default=None, help="Path to memory/ for expert discovery")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_router(
        input_dim=args.input_dim, 
        num_towers=args.num_towers, 
        towers_dir=args.towers_dir,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        lr=args.lr
    )

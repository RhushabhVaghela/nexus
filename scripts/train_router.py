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
    def __init__(self, data_path=None, fallback_size=1000):
        self.data_path = data_path
        self.fallback_size = fallback_size
        self.features = None
        self.labels = None
        
        self._load_data()
        
    def _load_data(self):
        # 1. Try Loading Real Data
        if self.data_path and os.path.exists(self.data_path):
            print(f"[Router_DS] Loading real data from {self.data_path}...")
            try:
                data = torch.load(self.data_path)
                self.features = data['features'] # [N, 4096]
                self.labels = data['labels']     # [N]
                print(f"[Router_DS] Loaded {len(self.features)} samples.")
                return
            except Exception as e:
                print(f"[Error] Failed to load data: {e}")

        # 2. Fallback to Synthetic (for CI/CD or initial run)
        print("[Router_DS] Real data not found. Generating SYNTHETIC features (4096-dim)...")
        print("  (To fix: Place 'labeled_intents.pt' in data/router_data/)")
        self.features = torch.randn(self.fallback_size, 4096)
        self.labels = torch.randint(0, 5, (self.fallback_size,))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_router(epochs=5, batch_size=32):
    print(f"\n[Router] Starting Training for {epochs} epochs...")
    
    # 1. Initialize Router
    # Input Dim: 4096 (Student Latent), Towers: 5 (Reasoning, Vision, Audio, Gen, Agent)
    router = SparseIntentRouter(input_dim=4096, num_towers=5, top_k=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router.to(device)
    
    optimizer = optim.Adam(router.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Data Loader
    dataset = RouterDataset(data_path=DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Training Loop
    for epoch in range(epochs):
        router.train()
        total_loss = 0
        steps = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Using the `gate` output for raw logits
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
    train_router()

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

def train_router(epochs=5, batch_size=32):
    print(f"\n[Router] Starting Training for {epochs} epochs...")
    
    # 1. Initialize Router
    # Input Dim: 4096 (Student Latent), Towers: 5 (Reasoning, Vision, Audio, Gen, Agent)
    router = SparseIntentRouter(input_dim=4096, num_towers=5, top_k=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router.to(device)
    
    optimizer = optim.Adam(router.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Mock Data Generator (In real prod, load from Tagged Dataset)
    # We generate "Latent Vectors" and "Correct Tower Label"
    # Tag map: 0:Reas, 1:Vis, 2:Aud, 3:Gen, 4:Agent
    print("[Router] Generating synthetic training data (Simulating Tagged Dataset)...")
    data_size = 1000
    X_train = torch.randn(data_size, 4096).to(device)
    y_train = torch.randint(0, 5, (data_size,)).to(device)
    
    # 3. Training Loop
    for epoch in range(epochs):
        router.train()
        total_loss = 0
        
        for i in range(0, data_size, batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Router returns (scores, indices). 
            # However, for training against CrossEntropy, we need raw logits usually.
            # Our SparseIntentRouter applies Softmax. 
            # We should inspect its `gate` output directly or adapt.
            # Using the `gate` layer output is best for CE loss.
            logits = router.gate(batch_X)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (data_size / batch_size)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    # 4. Save
    os.makedirs("results/router_weights", exist_ok=True)
    save_path = "results/router_weights/sparse_router.pt"
    torch.save(router.state_dict(), save_path)
    print(f"[Router] Weights saved to {save_path}")

if __name__ == "__main__":
    train_router()

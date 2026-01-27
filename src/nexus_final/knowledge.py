import torch
import torch.nn as nn
import os
import json
import faiss
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel

from .memory_projector import ActivationGuidedProjector

class FileMemoryManager:
    """
    Handles the 'File System as External Context' logic.
    Supports reading from local directories and storing 'observations'.
    """
    def __init__(self, memory_root: str = "memory/"):
        self.memory_root = memory_root
        os.makedirs(self.memory_root, exist_ok=True)
        
    def store_observation(self, key: str, content: str):
        path = os.path.join(self.memory_root, f"{key}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def load_observation(self, key: str) -> Optional[str]:
        path = os.path.join(self.memory_root, f"{key}.txt")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
        
    def list_memory_files(self) -> List[str]:
        return [f for f in os.listdir(self.memory_root) if f.endswith(".txt")]

class KnowledgeTower(nn.Module):
    """
    Integrated Memory Tower for the Nexus Specialist Architecture.
    Consolidates RAG logic into a trainable adapter-style module.
    """
    def __init__(
        self, 
        embedding_dim: int = 384, 
        student_dim: int = 4096,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.projector = ActivationGuidedProjector(embedding_dim, student_dim).to(device)
        self.memory_manager = FileMemoryManager()
        
        # Internal FAISS for fast retrieval during training/inference
        self.index = None
        self.documents = []
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.encoder = None

    def _lazy_init_encoder(self):
        if self.encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_id)
            self.encoder = AutoModel.from_pretrained(self.embedding_model_id).to(self.device).eval()

    def build_index(self, documents: List[str]):
        """
        Embeds documents and builds the retrieval index.
        """
        self._lazy_init_encoder()
        self.documents = documents
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.encoder(**encoded)
                # Mean Pooling + Normalization
                mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
                sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
                count = torch.clamp(mask.sum(1), min=1e-9)
                pooled = sum_emb / count
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(normalized.cpu().numpy())
                
        all_embeddings = np.concatenate(embeddings, axis=0)
        self.index = faiss.IndexFlatIP(all_embeddings.shape[1])
        self.index.add(all_embeddings)

    def forward(self, query: str, top_k: int = 3) -> torch.Tensor:
        """
        Retrieves top-k context and projects it into the student latent space.
        Returns tensor of shape [1, top_k, student_dim]
        """
        self._lazy_init_encoder()
        with torch.no_grad():
            encoded = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.encoder(**encoded)
            mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
            query_emb = (torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9))
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1).cpu().numpy()
            
        scores, indices = self.index.search(query_emb, top_k)
        
        # For each retrieved doc, we need to embed it again (or cache embeddings)
        # and then pass it through the projector.
        retrieved_embeddings = []
        for idx in indices[0]:
            if idx < len(self.documents):
                doc_text = self.documents[idx]
                # Embed doc
                with torch.no_grad():
                    d_enc = self.tokenizer([doc_text], padding=True, truncation=True, return_tensors='pt')
                    d_enc = {k: v.to(self.device) for k, v in d_enc.items()}
                    d_out = self.encoder(**d_enc)
                    d_mask = d_enc['attention_mask'].unsqueeze(-1).expand(d_out.last_hidden_state.size())
                    d_emb = (torch.sum(d_out.last_hidden_state * d_mask, 1) / torch.clamp(d_mask.sum(1), min=1e-9))
                    retrieved_embeddings.append(d_emb)
        
        if not retrieved_embeddings:
            return torch.zeros(1, 0, self.projector.projector[0].out_features).to(self.device)
            
        combined_embeddings = torch.stack(retrieved_embeddings, dim=1) # [1, k, emb_dim]
        # Project into student space
        projected_context = self.projector(combined_embeddings) # [1, k, student_dim]
        return projected_context

import numpy as np # Needed for build_index

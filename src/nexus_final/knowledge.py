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

    def save_context_session(self, session_id: str, context: List[str]):
        """Persists the current 'Desk' (Context) to SSD for cross-session sharing."""
        session_dir = os.path.join(self.memory_root, "sessions")
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(session_dir, f"{session_id}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2)
            
    def load_context_session(self, session_id: str) -> List[str]:
        """Loads a shared 'Desk' (Context) from SSD."""
        path = os.path.join(self.memory_root, "sessions", f"{session_id}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

class KnowledgeTower(nn.Module):
    """
    Integrated Memory Tower for the Nexus Specialist Architecture.
    Consolidates RAG logic into a trainable adapter-style module.
    """
    def __init__(
        self, 
        student_dim: int = 2048,
        embedding_dim: int = 384, 
        device: str = "cpu",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__()
        self.device = device
        self.projector = ActivationGuidedProjector(embedding_dim, student_dim).to(device)
        self.memory_manager = FileMemoryManager()
        
        # Internal FAISS for fast retrieval during training/inference
        self.index = None
        self.documents = []
        self.embedding_model_id = embedding_model
        self.tokenizer = None
        self.encoder = None

    def _lazy_init_encoder(self):
        if self.encoder is None:
            # 1. Attempt Unsloth FastSentenceTransformer (3x speedup)
            try:
                from unsloth import FastSentenceTransformer
                print(f"[Knowledge] Loading optimized FastSentenceTransformer: {self.embedding_model_id}")
                self.encoder = FastSentenceTransformer(self.embedding_model_id, device=self.device)
                self.using_fast_encoder = True
            except (ImportError, Exception) as e:
                # 2. Fallback to standard Transformers
                print(f"[Knowledge] FastSentenceTransformer Not Available ({e}). Falling back to standard Transformers.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_id)
                self.encoder = AutoModel.from_pretrained(self.embedding_model_id).to(self.device).eval()
                self.using_fast_encoder = False

    def build_index(self, documents: List[str]):
        """
        Embeds documents and builds the retrieval index.
        """
        self._lazy_init_encoder()
        self.documents = documents
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            if self.using_fast_encoder:
                # FastSentenceTransformer handles batching and pooling internally
                all_embeddings = self.encoder.encode(documents, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
                embeddings.append(all_embeddings)
            else:
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
                
        if self.using_fast_encoder:
            all_embeddings = embeddings[0]
        else:
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
            if self.using_fast_encoder:
                query_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            else:
                encoded = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.encoder(**encoded)
                mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
                pooled = (torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9))
                query_emb = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            
        scores, indices = self.index.search(query_emb, top_k)
        
        # For each retrieved doc, we need to embed it again (or cache embeddings)
        # and then pass it through the projector.
        retrieved_embeddings = []
        for idx in indices[0]:
            if idx < len(self.documents):
                doc_text = self.documents[idx]
                # Embed doc
                with torch.no_grad():
                    if self.using_fast_encoder:
                        # For the projector, we need the tensor output, not just the numpy embedding
                        # However, FastSentenceTransformer encode() often returns numpy by default.
                        # We use convert_to_tensor=True
                        d_emb = self.encoder.encode([doc_text], convert_to_tensor=True, normalize_embeddings=False)
                        retrieved_embeddings.append(d_emb)
                    else:
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

    def retrieve_text_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves raw text context for 'Standard Models' (e.g., Llama-3, GPT-4).
        This enables 'Unlimited Context' via standard RAG (Text Injection).
        """
        self._lazy_init_encoder()
        with torch.no_grad():
            if self.using_fast_encoder:
                query_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            else:
                encoded = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.encoder(**encoded)
                mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
                pooled = (torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9))
                query_emb = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            
        if self.index is None or self.index.ntotal == 0:
            return []
            
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def save_desk(self, session_id: str, context: List[str]):
        """Saves current context list to a named session."""
        self.memory_manager.save_context_session(session_id, context)
        
    def load_desk(self, session_id: str) -> List[str]:
        """Loads context list from a named session."""
        return self.memory_manager.load_context_session(session_id)

import numpy as np # Needed for build_index

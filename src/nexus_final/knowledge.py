import torch
import torch.nn as nn
import numpy as np
import os
import json
import faiss
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModel

class KnowledgeRetriever:
    """
    In-RAM Vector Index for Knowledge Augmentation (RAG).
    Provides 'Long-Term Memory' capability to the Student Model.
    """
    def __init__(self, embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
        self.model = AutoModel.from_pretrained(embedding_model_id).to(device)
        self.index = None
        self.documents = [] # Stores raw text mapped to index
        self.dimension = 384 # Default for MiniLM

    def build_index(self, documents: List[str]):
        """
        Ingests text documents, embeds them, and builds a FAISS index.
        """
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents
        
        # Batch Embed
        embeddings = []
        batch_size = 32
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                
                # Mean Pooling
                model_output = self.model(**encoded)
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize for Cosine Similarity
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
                
        all_embeddings = np.concatenate(embeddings, axis=0)
        self.dimension = all_embeddings.shape[1]
        
        # Create Index (Flat L2 is exact, HNSW is faster but approx. Using Flat for accuracy on RAM)
        # We use Inner Product (IP) because embeddings are normalized -> Cosine Sim
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(all_embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant documents for a query.
        """
        if self.index is None:
            return []
            
        # Embed Query
        with torch.no_grad():
            encoded = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            query_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu().numpy()
            
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "score": float(score)
                })
                
        return results

    def save(self, path: str):
        """Saves index and documents to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "knowledge.index"))
        with open(os.path.join(path, "documents.json"), 'w') as f:
            json.dump(self.documents, f)
            
    def load(self, path: str):
        """Loads index and documents."""
        self.index = faiss.read_index(os.path.join(path, "knowledge.index"))
        with open(os.path.join(path, "documents.json"), 'r') as f:
            self.documents = json.load(f)

# Integration with Adapter logic
# This class acts as a 'context provider' for the Reasoning Adapter

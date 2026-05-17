import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

INDEX_PATH = "data/index/index.faiss"
METADATA_PATH = "data/index/metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_MULTIPLIER = 5

class Retriever:
    def __init__(self, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH, model_name: str = MODEL_NAME):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None
        self._ready = False

        self._load()

    def _load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run scripts/build_index.py first."
            )
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata not found at {self.metadata_path}. "
                "Run scripts/build_index.py first."
            )
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        logger.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        assert self.index.ntotal == len(self.metadata), (
            f"Index size ({self.index.ntotal}) != metadata size ({len(self.metadata)}). "
            "Rebuild the index with scripts/build_index.py."
        )

        self._ready = True
        logger.info(
            f"Retriever ready. {self.index.ntotal} vectors, "
            f"dim={self.index.d}"
        )

    def is_ready(self) -> bool:
        return self._ready
    
    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vec.astype(np.float32)
    
    def search(self, query: str, top_k: int = 5, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._ready:
            raise RuntimeError("Retriever not ready. Call _load() first.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        
        fetch_k = top_k * RETRIEVAL_MULTIPLIER if domain_filter else top_k

        fetch_k = min(fetch_k, self.index.ntotal)

        query_vec = self.encode_query(query)

        scores, indices = self.index.search(query_vec, k = fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)

            if domain_filter and chunk.get("domain") != domain_filter:
                continue

            results.append(chunk)

            if len(results) >= top_k:
                break

        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics for the /stats API endpoint."""
        from collections import Counter
        domain_counts = Counter(c.get("domain", "unknown") for c in self.metadata)
        source_counts = Counter(c.get("source", "unknown") for c in self.metadata)
        return {
            "total_documents": len(self.metadata),
            "legal_documents": domain_counts.get("legal", 0),
            "medical_documents": domain_counts.get("medical", 0),
            "index_dimension": self.index.d if self.index else 0,
            "model_name": self.model_name,
            "sources": dict(source_counts),
        }
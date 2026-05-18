import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

INDEX_PATH = "data/index/index.faiss"
METADATA_PATH = "data/index/metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to pull from FAISS per requested result. Bi-encoder recall
# is broad but noisy; the cross-encoder needs a deep candidate pool to rerank.
RETRIEVAL_MULTIPLIER = 10

# Max chunks returned from the same contract/title in a single query result.
# Prevents one contract from monopolizing the top-K.
MAX_CHUNKS_PER_SOURCE = 2


class Retriever:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        model_name: str = MODEL_NAME,
        reranker_model: str = RERANKER_MODEL,
        use_reranker: bool = True,
        dedupe_by_source: bool = True,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.reranker_model_name = reranker_model
        self.use_reranker = use_reranker
        self.dedupe_by_source = dedupe_by_source

        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None
        self._reranker: Optional[CrossEncoder] = None
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
            f"dim={self.index.d}, reranker={'on' if self.use_reranker else 'off'}"
        )

    def _ensure_reranker(self):
        if self._reranker is None:
            logger.info(f"Loading cross-encoder reranker: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)

    def is_ready(self) -> bool:
        return self._ready

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        if not self._ready:
            raise RuntimeError("Retriever not ready. Call _load() first.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        do_rerank = self.use_reranker if rerank is None else rerank

        # Always over-fetch so the reranker and dedup pass have a real candidate
        # pool. Even without reranking, a wider pool lets the domain filter and
        # source dedup operate without starving top-K.
        fetch_k = min(top_k * RETRIEVAL_MULTIPLIER, self.index.ntotal)

        query_vec = self.encode_query(query)
        scores, indices = self.index.search(query_vec, k=fetch_k)

        candidates: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            if domain_filter and chunk.get("domain") != domain_filter:
                continue
            candidates.append(chunk)

        if not candidates:
            logger.warning(
                f"No candidates for query (domain_filter={domain_filter!r}). "
                f"Returned 0 chunks."
            )
            return []

        if do_rerank and len(candidates) > 1:
            self._ensure_reranker()
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = self._reranker.predict(pairs, show_progress_bar=False)
            for chunk, rscore in zip(candidates, rerank_scores):
                chunk["rerank_score"] = float(rscore)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        else:
            # Already sorted by FAISS score, but be explicit.
            candidates.sort(key=lambda c: c["score"], reverse=True)

        if self.dedupe_by_source:
            candidates = self._dedupe_by_source(candidates)

        return candidates[:top_k]

    @staticmethod
    def _dedupe_by_source(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cap chunks per contract so one source can't monopolize results."""
        counts: Dict[str, int] = {}
        kept = []
        for chunk in chunks:
            key = chunk.get("title") or chunk.get("source") or chunk.get("chunk_id")
            if counts.get(key, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            counts[key] = counts.get(key, 0) + 1
            kept.append(chunk)
        return kept

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics for the /stats API endpoint."""
        from collections import Counter
        if self.metadata is None or self.index is None:
            return {"ready": False}
        domain_counts = Counter(c.get("domain", "unknown") for c in self.metadata)
        source_counts = Counter(c.get("source", "unknown") for c in self.metadata)
        return {
            "total_documents": len(self.metadata),
            "legal_documents": domain_counts.get("legal", 0),
            "medical_documents": domain_counts.get("medical", 0),
            "index_dimension": self.index.d,
            "model_name": self.model_name,
            "reranker_model": self.reranker_model_name if self.use_reranker else None,
            "sources": dict(source_counts),
        }

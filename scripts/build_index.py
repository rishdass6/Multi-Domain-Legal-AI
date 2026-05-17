import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = "data/processed/chunks.jsonl"
INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64
SHOW_PROGRESS = True

os.makedirs(INDEX_DIR, exist_ok = True)

def load_chunks(path: str) -> list:
    chunks = []
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks

def encode_chunks(chunks: list, model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    print(f"\nEncoding {len(texts)} chunks in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size = BATCH_SIZE,
        show_progress_bar= SHOW_PROGRESS,
        convert_to_numpy = True,
        normalize_embeddings = True
    ) 

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    print(f"\nBuilding FAISS IndexFlatIP (dim={dim})...")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Index built. Total vectors: {index.ntotal}")
    return index

def save_index(index: faiss.IndexFlatIP, path: str):
    faiss.write_index(index, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"FAISS index saved to {path} ({size_mb:.1f} MB)")

def save_metadata(chunks: list, path: str):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Metadata saved to {path} ({size_mb:.1f} MB, {len(chunks)} entries)")

def build_index():
    print("=" * 60)
    print("PHASE 3: BUILDING FAISS INDEX")
    print("=" * 60)

    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        raise ValueError(f"No chunks found at {CHUNKS_PATH}. Run Phase 2 first.")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    embeddings = encode_chunks(chunks, model)

    index = build_faiss_index(embeddings)

    print("\nSaving artifacts...")
    save_index(index, INDEX_PATH)
    save_metadata(chunks, METADATA_PATH)

    print("\n── Sanity Check ─────────────────────────────────────────")
    test_query = "What is the definition of negligence?"
    print(f"Test query: '{test_query}'")
    query_vec = model.encode([test_query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vec, k = 3)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx]
        print(f"\n  Rank {rank+1} (score={score:.4f})")
        print(f"  Source: {chunk['source']}")
        print(f"  Text:   {chunk['text'][:150]}...")

    print("\n✓ Phase 3 complete. Index is ready for the Retriever.")
    print(f"  Index:    {INDEX_PATH}")
    print(f"  Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    build_index()
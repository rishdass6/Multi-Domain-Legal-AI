import os
import json
import glob
from tqdm import tqdm
from collections import defaultdict

from app.preprocessing import clean_document
from app.chunker import chunk_all_documents, save_chunks_to_jsonl, count_tokens

RAW_DIRS = {
    "cuad": "data/raw/cuad",
    "lii": "data/raw/lii",
}
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "chunks.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_documents(raw_dir: str) -> list:
    """Load all JSON files from a raw data directory."""
    pattern = os.path.join(raw_dir, "*.json")
    files = sorted(glob.glob(pattern))
    docs = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                doc = json.load(f)
                docs.append(doc)
            except json.JSONDecodeError as e:
                print(f"  Skipping malformed file {filepath}: {e}")
    return docs


def build_dataset():
    print("=" * 60)
    print("LEGAL QA DATASET BUILD")
    print("=" * 60)

    all_docs = []

    # ── Load CUAD ────────────────────────────────────────────────
    print("\n[1/4] Loading CUAD contracts...")
    cuad_docs = load_raw_documents(RAW_DIRS["cuad"])
    print(f"      Loaded {len(cuad_docs)} raw contracts")
    all_docs.extend(cuad_docs)

    # ── Load LII ─────────────────────────────────────────────────
    print("\n[2/4] Loading Cornell LII pages...")
    lii_docs = load_raw_documents(RAW_DIRS["lii"])
    print(f"      Loaded {len(lii_docs)} raw LII pages")
    all_docs.extend(lii_docs)

    print(f"\n      Total raw documents: {len(all_docs)}")

    # ── Clean ────────────────────────────────────────────────────
    print("\n[3/4] Cleaning documents...")
    cleaned_docs = []
    skipped = 0
    for doc in tqdm(all_docs, desc="Cleaning"):
        cleaned_text = clean_document(doc.get("text", ""), doc.get("source", ""))
        if cleaned_text:
            doc["text"] = cleaned_text
            cleaned_docs.append(doc)
        else:
            skipped += 1

    print(f"      Cleaned: {len(cleaned_docs)} documents")
    print(f"      Skipped (too short after cleaning): {skipped}")

    # ── Chunk ────────────────────────────────────────────────────
    print("\n[4/4] Chunking documents...")
    all_chunks = chunk_all_documents(cleaned_docs)
    print(f"      Generated {len(all_chunks)} total chunks")

    # ── Stats ────────────────────────────────────────────────────
    print("\n── Dataset Statistics ──────────────────────────────────")
    source_counts = defaultdict(int)
    token_totals = defaultdict(int)
    for chunk in all_chunks:
        source_counts[chunk["source"]] += 1
        token_totals[chunk["source"]] += chunk["token_count"]

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        avg_tokens = token_totals[source] // count if count else 0
        print(f"  {source:<25} {count:>6} chunks  (avg {avg_tokens} tokens/chunk)")

    total_tokens = sum(c["token_count"] for c in all_chunks)
    print(f"\n  Total tokens in corpus: {total_tokens:,}")
    print(f"  Avg tokens per chunk:   {total_tokens // len(all_chunks) if all_chunks else 0}")

    # ── Save ─────────────────────────────────────────────────────
    print(f"\n── Saving to {OUTPUT_PATH} ──")
    save_chunks_to_jsonl(all_chunks, OUTPUT_PATH)

    print("\n✓ Phase 2 complete. Ready for Phase 3 (Embedding + FAISS).")
    print(f"  Output: {OUTPUT_PATH}")
    return all_chunks


if __name__ == "__main__":
    chunks = build_dataset()

    # Show a sample chunk for verification
    print("\n── Sample Chunk ───────────────────────────────────────")
    sample = chunks[0]
    print(f"  chunk_id:    {sample['chunk_id']}")
    print(f"  source:      {sample['source']}")
    print(f"  domain:      {sample['domain']}")
    print(f"  token_count: {sample['token_count']}")
    print(f"  title:       {sample['title'][:60]}")
    print(f"  text[:300]:  {sample['text'][:300]}")
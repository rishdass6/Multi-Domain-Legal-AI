"""
Master dataset build script — CUAD + LII Wex.
"""

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


def load_raw_documents(raw_dir: str, source_label: str) -> list:
    """
    Load raw docs and normalize the source field to a short canonical label
    ("cuad", "lii", …). Raw LII files ship with source="Cornell LII Wex";
    storing that verbatim would break downstream source-equality checks and
    UI filters. We keep the original under `source_raw` for traceability.
    """
    pattern = os.path.join(raw_dir, "*.json")
    files = sorted(glob.glob(pattern))
    docs = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                doc = json.load(f)
                if doc.get("source") and doc["source"] != source_label:
                    doc["source_raw"] = doc["source"]
                doc["source"] = source_label
                doc.setdefault("domain", "legal")
                docs.append(doc)
            except json.JSONDecodeError as e:
                print(f"  Skipping malformed file {filepath}: {e}")
    return docs


def build_dataset():
    print("=" * 60)
    print("LEGAL QA DATASET BUILD — CUAD + LII Wex")
    print("=" * 60)

    all_docs = []

    print("\n[1/4] Loading raw documents...")
    cuad_docs = load_raw_documents(RAW_DIRS["cuad"], "cuad")
    print(f"      Loaded {len(cuad_docs)} CUAD contracts")
    all_docs.extend(cuad_docs)

    lii_docs = load_raw_documents(RAW_DIRS["lii"], "lii")
    print(f"      Loaded {len(lii_docs)} LII Wex entries")
    all_docs.extend(lii_docs)

    print("\n[2/4] Cleaning documents...")
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
    print(f"      Skipped: {skipped}")

    print("\n[3/4] Chunking documents...")
    all_chunks = chunk_all_documents(cleaned_docs)
    print(f"      Generated {len(all_chunks)} total chunks")

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

    print(f"\n── Saving to {OUTPUT_PATH} ──")
    save_chunks_to_jsonl(all_chunks, OUTPUT_PATH)

    print("\n✓ Dataset build complete.")
    return all_chunks


if __name__ == "__main__":
    chunks = build_dataset()
    sample = chunks[0]
    print(f"\nSample chunk: {sample['text'][:300]}")
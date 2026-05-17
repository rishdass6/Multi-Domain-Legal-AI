import json
import os
from collections import Counter, defaultdict

CHUNKS_PATH = "data/processed/chunks.jsonl"


def verify():
    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: {CHUNKS_PATH} not found. Run build_dataset.py first.")
        return

    chunks = []
    with open(CHUNKS_PATH, "r", encoding = "utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError:
                print(f"  Bad JSON at line {i+1}")

    print(f"Total chunks loaded: {len(chunks)}")

    # ── Required fields check ──────────────────────────────────
    required_fields = {"chunk_id", "text", "source", "domain", "token_count"}
    missing_fields = []
    for chunk in chunks:
        missing = required_fields - set(chunk.keys())
        if missing:
            missing_fields.append((chunk.get("chunk_id", "?"), missing))

    if missing_fields:
        print(f"\nWARNING: {len(missing_fields)} chunks missing fields:")
        for chunk_id, fields in missing_fields[:5]:
            print(f"  chunk_id={chunk_id}, missing={fields}")
    else:
        print("✓ All chunks have required fields")

    # ── Empty text check ───────────────────────────────────────
    empty = [c for c in chunks if not c.get("text", "").strip()]
    if empty:
        print(f"\nWARNING: {len(empty)} chunks have empty text")
    else:
        print("✓ No empty text chunks")

    # ── Duplicate chunk_id check ───────────────────────────────
    ids = [c["chunk_id"] for c in chunks]
    dup_ids = [id_ for id_, count in Counter(ids).items() if count > 1]
    if dup_ids:
        print(f"\nWARNING: {len(dup_ids)} duplicate chunk_ids found")
    else:
        print("✓ All chunk_ids are unique")

    # ── Token distribution ─────────────────────────────────────
    tokens = [c["token_count"] for c in chunks]
    print(f"\nToken distribution:")
    print(f"  Min:    {min(tokens)}")
    print(f"  Max:    {max(tokens)}")
    print(f"  Mean:   {sum(tokens) // len(tokens)}")
    print(f"  Chunks under 50 tokens:  {sum(1 for t in tokens if t < 50)}")
    print(f"  Chunks over 450 tokens:  {sum(1 for t in tokens if t > 450)}")

    # ── Source breakdown ───────────────────────────────────────
    source_counts = Counter(c["source"] for c in chunks)
    print(f"\nSource breakdown:")
    for source, count in source_counts.most_common():
        print(f"  {source:<30} {count} chunks")

    # ── Domain breakdown ───────────────────────────────────────
    domain_counts = Counter(c["domain"] for c in chunks)
    print(f"\nDomain breakdown:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain:<20} {count} chunks")

    # ── Sample output ──────────────────────────────────────────
    print("\n── Sample chunks by source ──────────────────────────────")
    shown = set()
    for chunk in chunks:
        src = chunk["source"]
        if src not in shown:
            shown.add(src)
            print(f"\n  Source: {src}")
            print(f"  Title:  {chunk.get('title', '')[:70]}")
            print(f"  Tokens: {chunk['token_count']}")
            print(f"  Text:   {chunk['text'][:250]}...")

    print("\n✓ Verification complete. Proceed to Phase 3.")


if __name__ == "__main__":
    verify()
import json
import os
import uuid
import tiktoken
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 100

ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def token_len(text: str) -> int:
    return count_tokens(text)

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE_TOKENS,
    chunk_overlap = CHUNK_OVERLAP_TOKENS,
    length_function = token_len,
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
)

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = doc.get("text", "").strip()
    if not text:
        return []

    raw_chunks = SPLITTER.split_text(text)

    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        token_size = count_tokens(chunk_text)
        if not chunk_text or token_size <= 20:
            continue

        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": chunk_text,
            "source": doc.get("source", "unknown"),
            "domain": doc.get("domain", "legal"),
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "token_count": count_tokens(chunk_text),
        }
        chunks.append(chunk)

    return chunks

def chunk_all_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk a list of documents and return flat list of all chunks."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    return all_chunks


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks to JSONL — one JSON object per line."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load chunks back from JSONL for inspection or re-processing."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
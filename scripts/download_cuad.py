import os
import json
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = "data/raw/cuad"
os.makedirs(RAW_DIR, exist_ok=True)

def download_cuad():
    print(f"Loading CUAD from HuggingFace...")

    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    seen_titles = set()
    contracts = []

    for split in ["train", "test"]:
        for row in tqdm(dataset[split], desc = f"Processing {split} split"):
            title = row["title"]
            context = row["context"]

            if title not in seen_titles and context.strip():
                seen_titles.add(title)
                contracts.append({
                    "title": title,
                    "text": context,
                    "source": "cuad",
                    "domain": "legal",
                    "url": "https://www.atticusprojectai.org/cuad"
                })

    for i, contract in enumerate(contracts):
        filename = f"cuad_{i:04d}.json"
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "w", encoding = "utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, indent = 2)

    print(f"\nDone. Saved {len(contracts)} unique contracts to {RAW_DIR}/")
    return contracts

if __name__ == "__main__":
    contracts = download_cuad()
    print(f"Sample title: {contracts[0]['title']}")
    print(f"Sample text (first 300 chars): {contracts[0]['text'][:300]}")
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW_DIR = "data/raw/lii"
os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://www.law.cornell.edu"
WEX_INDEX_URL = "https://www.law.cornell.edu/wex/all"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LegalQAResearcherBot/1.0;"
        "+https:://github.com/rishdass6/legal-qa)"
    )
}

def get_wex_topic_urls():
    print(f"Fetching Wex Index from {WEX_INDEX_URL}")
    resp = requests.get(WEX_INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    topic_urls = []
    
    # Scan the entire page directly for links
    for link in soup.find_all("a", href=True):
        href = link["href"]
        
        # Topic links must start with /wex/ 
        if href.startswith("/wex/"):
            # Exclude the index homepage, the alphabet filter views, and utility pages
            if href != "/wex/all" and "/all/" not in href and "?" not in href:
                full_url = BASE_URL + href
                if full_url not in topic_urls:
                    topic_urls.append(full_url)

    print(f"Found {len(topic_urls)} Wex topic URLs.")
    return topic_urls


def scrape_wex_page(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Skipped {url}: Bad status code {resp.status_code}")
            return None
        
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

        content_div = (
            soup.find("div", class_="field--name-body")
            or soup.find("div", class_="field-name-body") # Single hyphen variation
            or soup.find("div", class_=lambda c: c and "field-item" in c) # Catch all field-items
            or soup.find("div", class_="region-content")
            or soup.find("div", id="block-system-main")
            or soup.find("div", class_="content")
            or soup.find("div", id="content-wrapper")
            or soup.find("article")
        )

        # NEW FALLBACK: If the specific Drupal classes fail, just grab the main body
        if not content_div:
            content_div = soup.find("main") or soup.find("body")

        if not content_div:
            print(f"  Skipped {url}: Could not find any content container")
            return None
        
        paragraphs = content_div.find_all(["p", "li", "h2", "h3", "h4"])
        text_parts = []
        for tag in paragraphs:
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 15:
                text_parts.append(text)

        full_text = "\n\n".join(text_parts)

        if len(full_text) < 40:
            # DEBUG PRINT ADDED
            print(f"  Skipped {url}: Text too short ({len(full_text)} chars)")
            return None

        return {
            "title": title,
            "text": full_text,
            "source": "Cornell LII Wex",
            "domain": "legal",
            "url": url
        }
    except Exception as e:
        print(f"   Error Scraping {url}: {e}")
        return None

    
def scrape_lii(max_pages = 100, delay=1.0):
    topic_urls = get_wex_topic_urls()
    topic_urls = topic_urls[:max_pages]

    results = []

    for i, link in enumerate(tqdm(topic_urls, desc = f"Scraping LII Wex")):
        doc = scrape_wex_page(link)
        if not doc:
            continue
        results.append(doc)
        filename = f"lii_{i:04d}.json"
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent = 2)

        time.sleep(delay)

    print(f"\nDone. Scraped {len(results)} LII pages to {RAW_DIR}/")
    return results

if __name__ == "__main__":
    docs = scrape_lii(max_pages=100, delay=1.0)
    if docs:
        print(f"\nSample: {docs[0]['title']}")
        print(f"Text preview: {docs[0]['text'][:400]}")



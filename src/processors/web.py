"""URL processor — fetch + extract main text. Handles HTML and plain-text bodies."""
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) RAGBot/1.0"
    )
}


def process(url: str) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "").lower()

    if "html" in ct:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    else:
        text = resp.text
        title = url

    return {
        "kind": "web",
        "source": url,
        "filename": title[:100],
        "text": text,
        "content_type": ct,
    }

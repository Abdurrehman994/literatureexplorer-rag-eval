"""Extract page-level text from the selected papers (0 endpoint requests).

PDFs: one record per page (page = 1-based page number). HTML articles: main article text without
menus/scripts, split into ~600-word "pages" so later steps can treat both formats alike.

Usage:  python src/extract_texts.py
Output: data/papers_text/NNN.jsonl  ({"doc_id", "page", "text"} per line), data/papers_text_stats.csv
"""
import json
import re

import pandas as pd
import pymupdf
from bs4 import BeautifulSoup

from config import DATA_DIR, ROOT

TEXT_DIR = DATA_DIR / "papers_text"
HTML_PAGE_WORDS = 600


def clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def pdf_pages(path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [clean(page.get_text()) for page in doc]


def html_pages(path) -> list[str]:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body
    words = clean(main.get_text(" ")).split()
    return [" ".join(words[i:i + HTML_PAGE_WORDS]) for i in range(0, len(words), HTML_PAGE_WORDS)]


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA_DIR / "papers_manifest.csv")
    stats = []
    for _, row in manifest[manifest["ok"]].iterrows():
        path = ROOT / row["file"]
        pages = pdf_pages(path) if row["format"] == "pdf" else html_pages(path)
        with (TEXT_DIR / f"{row['doc_id']:03d}.jsonl").open("w", encoding="utf-8") as f:
            for i, text in enumerate(pages, start=1):
                f.write(json.dumps({"doc_id": int(row["doc_id"]), "page": i, "text": text},
                                   ensure_ascii=False) + "\n")
        words = [len(p.split()) for p in pages]
        stats.append({"doc_id": row["doc_id"], "title": row["Title"], "format": row["format"],
                      "pages": len(pages), "words": sum(words),
                      "empty_pages": sum(w < 20 for w in words)})

    stats = pd.DataFrame(stats)
    stats.to_csv(DATA_DIR / "papers_text_stats.csv", index=False)
    print(stats.to_string(index=False, max_colwidth=50))
    print(f"\nTotal: {stats['pages'].sum()} pages, {stats['words'].sum():,} words")


if __name__ == "__main__":
    main()

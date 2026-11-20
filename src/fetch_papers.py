"""Download the selected papers from their public URLs (fallback while papers.zip returns 403).

For each paper we try direct-PDF URLs derived from the landing page (arXiv, ACL, Nature, ...),
then the landing page itself. A response is kept if it is a PDF (checked by its %PDF header) or,
for web articles (blog posts, news), an HTML page. Files go to papers/selected/NNN.pdf|.html.
Papers that cannot be fetched are listed so you can download them manually into the same folder.

These are the publicly available versions, which may differ slightly from the files in the
knowledge base. Not an endpoint request; polite: one request at a time, 3 s apart.

Usage:  python src/fetch_papers.py
Output: papers/selected/*, data/papers_manifest.csv
"""
import re
import time

import pandas as pd
import requests

from config import DATA_DIR, ROOT

OUT_DIR = ROOT / "papers" / "selected"
HEADERS = {"User-Agent": "Mozilla/5.0 (student course project; RAG evaluation)"}
PAUSE_S = 3


def candidate_urls(url: str) -> list[str]:
    url = url.strip()
    if m := re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+?)(?:\.pdf)?/?$", url):
        return [f"https://arxiv.org/pdf/{m.group(1)}"]
    if m := re.search(r"aclanthology\.org/([^/]+)/?$", url):
        return [f"https://aclanthology.org/{m.group(1)}.pdf"]
    if m := re.search(r"doi\.org/10\.1038/(s4\d+-\d+-\d+-\w)", url):
        return [f"https://www.nature.com/articles/{m.group(1)}.pdf", url]
    if m := re.search(r"nature\.com/articles/(s4[^/?#]+)", url):
        return [f"https://www.nature.com/articles/{m.group(1)}.pdf", url]
    if m := re.search(r"dl\.acm\.org/doi/(10\.[^?#]+)", url):
        return [f"https://dl.acm.org/doi/pdf/{m.group(1)}", url]
    if m := re.search(r"pnas\.org/doi/(10\.[^?#]+)", url):
        return [f"https://www.pnas.org/doi/pdf/{m.group(1)}", url]
    if "mdpi.com" in url:
        return [url.rstrip("/") + "/pdf", url]
    return [url]


def fetch(doc_id: int, url: str) -> dict:
    tried = []
    for candidate in candidate_urls(url):
        try:
            resp = requests.get(candidate, headers=HEADERS, timeout=60, allow_redirects=True)
            ctype = resp.headers.get("Content-Type", "")
            tried.append(f"{resp.status_code} {candidate}")
            if resp.ok and resp.content[:5] == b"%PDF-":
                ext = "pdf"
            elif resp.ok and "html" in ctype and len(resp.content) > 5000:
                ext = "html"
            else:
                ext = None
            if ext:
                path = OUT_DIR / f"{doc_id:03d}.{ext}"
                path.write_bytes(resp.content)
                return {"doc_id": doc_id, "ok": True, "format": ext, "file": str(path.relative_to(ROOT)),
                        "bytes": len(resp.content), "fetched_from": candidate, "tried": " | ".join(tried)}
        except requests.RequestException as exc:
            tried.append(f"ERR {candidate}: {type(exc).__name__}")
        time.sleep(PAUSE_S)
    return {"doc_id": doc_id, "ok": False, "format": None, "file": None, "bytes": 0,
            "fetched_from": None, "tried": " | ".join(tried)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(DATA_DIR / "selected_documents.csv")
    rows = []
    for _, doc in selected.iterrows():
        existing = [p for p in OUT_DIR.glob(f"{doc['doc_id']:03d}.*") if p.suffix in (".pdf", ".html")]
        if existing:  # already downloaded (or placed manually)
            p = existing[0]
            rows.append({"doc_id": doc["doc_id"], "ok": True, "format": p.suffix[1:],
                         "file": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                         "fetched_from": "existing file", "tried": ""})
            continue
        row = fetch(doc["doc_id"], doc["URL"])
        print(f"{doc['doc_id']:3d} {'OK  ' + row['format'] if row['ok'] else 'FAIL    '} {row['tried']}")
        rows.append(row)
        time.sleep(PAUSE_S)

    manifest = selected[["doc_id", "Title", "URL"]].merge(pd.DataFrame(rows), on="doc_id")
    manifest.to_csv(DATA_DIR / "papers_manifest.csv", index=False)
    print(f"\nFetched {manifest['ok'].sum()}/{len(manifest)}  "
          f"({(manifest['format'] == 'pdf').sum()} PDF, {(manifest['format'] == 'html').sum()} HTML)")
    failed = manifest[~manifest["ok"]]
    if len(failed):
        print("\nDownload these manually and save as papers/selected/NNN.pdf, then rerun:")
        for _, r in failed.iterrows():
            print(f"  {r['doc_id']:03d}.pdf  <-  {r['URL']}")


if __name__ == "__main__":
    main()

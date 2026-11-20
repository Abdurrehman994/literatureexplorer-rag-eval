# LiteratureExplorer – RAG evaluation (Assignment 2)

Black-box evaluation of the LiteratureExplorer RAG endpoint: Ragas evaluation of its outputs
plus system tests (latency, error handling, coverage of the document collection).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then fill in the URLs and API keys
```

The Lehmus AI judge/embedding models are only reachable from the University of Oulu network or VPN.

## Scripts

| Step | Command | What it does | Endpoint requests |
| --- | --- | --- | --- |
| 0 | `python src/check_lehmus.py` | Checks that the judge + embedding models are reachable | 0 |
| 1 | `python src/probe.py` | First look at both endpoint routes | 4 |
| 2 | `python src/select_documents.py [--n 30] [--seed 42]` | Stratified sample of papers (by category) for the test set → `data/selected_documents.csv` | 0 |
| 3 | `python src/coverage_probe.py` | Queries `/v1/context` with each paper's title (+ content queries for misses) → `data/coverage_results.csv` | 189 |
| 4 | `python src/fetch_papers.py` | Downloads the 30 selected papers from their public URLs → `papers/selected/`, `data/papers_manifest.csv`. Papers it cannot fetch must be saved manually as `papers/selected/NNN.pdf`, then rerun. | 0 |
| 5 | `python src/extract_texts.py` | Page-level text of the selected papers → `data/papers_text/NNN.jsonl`, `data/papers_text_stats.csv` | 0 |
| 6 | `python src/build_testset.py [--seed 42]` | Generates candidate questions (Gemma) with automatic checks (evidence quote, GPT-OSS critic) + hand-written ones → `data/testset_candidates.csv` for manual review. Needs VPN. | 0 |
| – | `python src/client.py` | Prints how many endpoint requests have been used so far | 0 |

More steps will be added as the evaluation is built.

## Caching and endpoint rules

All endpoint calls go through [src/client.py](src/client.py), which caches every response in
`data/raw_responses/endpoint_cache.jsonl` (the raw-response deliverable). Re-running a script replays
from the cache instead of calling the endpoint again. The client sends at most 2 requests in parallel
and waits 5/15/30 s before retrying on HTTP 429/502/503/504.

## Repository layout

```text
src/                   evaluation code
data/raw_responses/    cached raw endpoint responses (JSONL)
.env.example           names of the required environment variables (copy to .env)
```

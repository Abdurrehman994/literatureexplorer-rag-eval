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
|---|---|---|---|
| 0 | `python src/check_lehmus.py` | Checks that the judge + embedding models are reachable | 0 |
| 1 | `python src/probe.py` | First look at both endpoint routes | 4 |
| – | `python src/client.py` | Prints how many endpoint requests have been used so far | 0 |

*(More steps will be added as the evaluation is built.)*

## Caching and endpoint rules

All endpoint calls go through [src/client.py](src/client.py), which caches every response in
`data/raw_responses/endpoint_cache.jsonl` (the raw-response deliverable). Re-running a script replays
from the cache instead of calling the endpoint again. The client sends at most 2 requests in parallel
and waits 5/15/30 s before retrying on HTTP 429/502/503/504.

## Repository layout

```
src/                   evaluation code
data/raw_responses/    cached raw endpoint responses (JSONL)
.env.example           names of the required environment variables (copy to .env)
```

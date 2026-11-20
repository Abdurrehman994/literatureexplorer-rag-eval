# LiteratureExplorer – RAG evaluation (Assignment 2)

Black-box evaluation of the LiteratureExplorer RAG endpoint (Titan Text Embeddings v2 + Amazon Nova
Lite) for XYZ Inc.:

- **Output evaluation with Ragas**: faithfulness, answer relevancy, context precision, context recall
  and factual correctness on 85 answerable test questions, with GPT-OSS 120B as the main judge and
  Gemma 4 26B as a second judge to measure judge agreement.
- **System tests**: coverage of all 183 papers, error handling and API contract, multi-turn
  behaviour, latency, consistency across repeated runs, prompt injection, out-of-scope, false-premise
  and ambiguous questions.

The report is in [report/](report/). All numbers in it come from
[data/analysis/summary.md](data/analysis/summary.md), which `src/analyze.py` generates.

## Deliverables in this repository

| Deliverable | Where |
| --- | --- |
| Report | `report/Evaluation-report.docx` (figures in `report/figures/`) |
| Test set (final, filtered) | [data/testset.csv](data/testset.csv) / [data/testset.json](data/testset.json): question, type, expected behaviour, reference answer, source document(s) and page(s), evidence quote |
| Test-set candidates incl. rejected ones and review reasons | [data/testset_candidates.csv](data/testset_candidates.csv) (`keep`, `review_note`) |
| Raw endpoint responses | [data/raw_responses/endpoint_cache.jsonl](data/raw_responses/endpoint_cache.jsonl): every request sent to the endpoint (request body, status, latency, full response). [data/responses.jsonl](data/responses.jsonl): test-set answers joined with the test set |
| Ragas scores | `data/ragas/scores_large.csv` (GPT-OSS), `data/ragas/scores_small.csv` (Gemma) |
| Other results | `data/answer_behaviour.csv`, `data/coverage_results.csv`, `data/system_tests/*.csv`, `data/analysis/*` |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp; then fill in the URLs and API keys
```

Tested with Python 3.13 on Windows 11. API URLs and keys are read from `.env` and are never
committed. The Lehmus AI models (judge, generator, embeddings) are only reachable from the
University of Oulu network or VPN.

## Reproducing the results without calling any API

All endpoint responses, LLM outputs (`data/llm_cache.jsonl`) and Ragas scores are stored in the
repository, so every script below replays from them. Re-running a step sends **no** new requests and
needs **no** `.env` or VPN, except where the input changed. To regenerate only the report tables and
figures:

```bash
python src/analyze.py
```

## Pipeline (run in this order)

| Step | Command | What it does | Endpoint requests | Needs VPN |
| --- | --- | --- | --- | --- |
| 0 | `python src/check_lehmus.py` | Checks that the judge and embedding models are reachable | 0 | yes |
| 1 | `python src/probe.py` | First look at both endpoint routes | 4 | – |
| 2 | `python src/select_documents.py [--n 30] [--seed 42]` | Stratified sample of 30 papers (by category) → `data/selected_documents.csv`, `data/documents_clean.csv` | 0 | – |
| 3 | `python src/coverage_probe.py` | Queries `/v1/context` with each of the 183 titles, plus content queries for misses → `data/coverage_results.csv` | 189 | – |
| 4 | `python src/fetch_papers.py` | Downloads the 30 selected papers from their public URLs → `papers/selected/`, `data/papers_manifest.csv`. Papers it cannot fetch must be saved manually as `papers/selected/NNN.pdf`, then rerun. | 0 | – |
| 5 | `python src/extract_texts.py` | Page-level text of the selected papers → `data/papers_text/NNN.jsonl` (not committed: copyrighted full text) | 0 | – |
| 6 | `python src/build_testset.py [--seed 42]` | Generates candidate questions with Gemma, checks them (evidence quote must be on the page; GPT-OSS critic), adds hand-written ones → `data/testset_candidates.csv` | 0 | yes |
| – | *manual review* | Set `keep` (1/0) in `data/testset_candidates.csv`; reasons go in `review_note` | 0 | – |
| 7 | `python src/finalize_testset.py [--seed 42] [--per-kind 3]` | Kept questions + robustness variants (typos, Finnish, keywords) → `data/testset.csv`, `data/testset.json` | 0 | yes* |
| 8 | `python src/collect_responses.py` | Asks every test question once via `/v1/chat/completions` → `data/responses.jsonl` | 101 (+1 lost to a since-fixed cache race) | – |
| 9 | `python src/run_ragas.py --judge large` then `python src/run_ragas.py --judge small` | Ragas metrics on the 85 answerable questions. Resumable: finished scores are skipped. Options: `--workers 4`, `--ids Q001,Q002`, `--metrics faithfulness,...` | 0 | yes* |
| 10 | `python src/classify_answers.py [--workers 2]` | GPT-OSS labels each answer: answered / partial / abstained / blocked, followed injection, corrected false premise, asked for clarification, language → `data/answer_behaviour.csv` | 0 | yes* |
| 11 | `python src/system_tests.py` | API and error-handling tests, latency (short vs long questions), consistency (8 questions × 3 runs) → `data/system_tests/*.csv` | 48 | – |
| 12 | `python src/analyze.py` | All tables, figures and key numbers → `data/analysis/`, `report/figures/` | 0 | – |
| – | `python src/client.py` | Prints how many endpoint requests have been used so far (342 in total) | 0 | – |

\* only when the cached outputs are missing or the input changed.

Steps 4 and 5 are only needed to regenerate the test set: the source pages are not committed, but
every question in `data/testset.csv` carries its evidence quote, source document and page. The
official `papers.zip` returned HTTP 403 (Access Denied), so the papers were downloaded from their
public URLs. 140 of 145 retrieved chunks matched these files word for word on the same page, so they
are the same versions as in the knowledge base.

## Caching and endpoint rules

- **Endpoint** ([src/client.py](src/client.py)): every request is cached in
  `data/raw_responses/endpoint_cache.jsonl`, keyed by method, route, body, auth mode and a `tag` (used
  to deliberately repeat a request, e.g. for consistency tests). At most 2 requests are in flight. On
  HTTP 429/502/503/504 the client waits 5, 15 and 30 s before retrying. The API key is never written
  to disk.
- **Lehmus models** ([src/llm.py](src/llm.py)): every chat call is cached in `data/llm_cache.jsonl`.
  Ragas calls are additionally cached in `data/ragas_cache/` (not committed). Finished scores are in
  `data/ragas/scores_<judge>.jsonl`, so reruns skip them. At most 4 judge calls run in parallel.

## Repository layout

```text
src/                    evaluation code (one script per pipeline step)
data/testset.csv        final test set (also .json)
data/raw_responses/     raw endpoint responses (JSONL, every request)
data/responses.jsonl    test-set answers with retrieved chunks and latency
data/ragas/             Ragas scores per judge
data/system_tests/      API, latency and consistency test results
data/analysis/          tables and summary.md generated by analyze.py
report/                 evaluation report and figures
.env.example            names of the required environment variables (copy to .env)
```

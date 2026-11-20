"""Ask every test-set question once via /v1/chat/completions (1 endpoint request per question, cached).

The chat response contains both the answer and the 5 retrieved chunks it was generated from
(identical to /v1/context, see probe.py), so this single call feeds both the Ragas evaluation
and the retrieval checks.

Usage:  python src/collect_responses.py
Output: data/responses.jsonl (one record per question, joined with the test set),
        raw HTTP records stay in data/raw_responses/endpoint_cache.jsonl
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import client
from config import DATA_DIR

TAG = "testset"
ABSTAIN_PATTERN = re.compile(r"\b(i don'?t know|i do not know|not (?:mentioned|provided|available|"
                             r"contain)|no information|cannot (?:answer|find))\b", re.I)


def collect(row: dict) -> dict:
    rec = client.chat(row["question"], tag=TAG)
    body = rec["response_json"] or {}
    contexts = body.get("contexts") or []
    answer = client.answer_text(rec)
    retrieved_ids = [int(c["source"][:3]) for c in contexts if c.get("source", "")[:3].isdigit()]
    expected_ids = [int(d) for d in row["source_doc_ids"].split(";") if d]
    return {
        **row,
        "status": rec["status"],
        "latency_s": rec["latency_s"],
        "error": rec["error"],
        "answer": answer,
        "abstained": bool(answer and ABSTAIN_PATTERN.search(answer)),
        "contexts": [c["text"] for c in contexts],
        "context_sources": [c["source"] for c in contexts],
        "context_pages": [c["page"] for c in contexts],
        "context_scores": [round(c["score"], 4) for c in contexts],
        "retrieved_doc_ids": retrieved_ids,
        # retrieval hit: every expected source paper appears among the 5 retrieved chunks
        "retrieval_hit": bool(expected_ids) and all(d in retrieved_ids for d in expected_ids),
        "usage": body.get("usage"),
        "response_sources": body.get("sources"),
        "cache_key": rec["key"],
    }


def main() -> None:
    testset = pd.read_csv(DATA_DIR / "testset.csv", dtype=str, keep_default_na=False)
    with ThreadPoolExecutor(max_workers=client.MAX_PARALLEL) as pool:
        rows = list(pool.map(collect, testset.to_dict("records")))
    with (DATA_DIR / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    print(f"{len(df)} questions | HTTP status: {df['status'].value_counts().to_dict()} | "
          f"budget: {client.usage()}")
    print(f"Latency (s): p50 {df['latency_s'].median():.2f}, p95 {df['latency_s'].quantile(.95):.2f}, "
          f"max {df['latency_s'].max():.2f}")
    print("\nAbstained ('I don't know'-style) by expected behaviour:")
    print(df.groupby("expected_behaviour")["abstained"].agg(["mean", "sum", "count"]).to_string())
    ans = df[df["expected_behaviour"] == "answer"]
    print("\nRetrieval hit (expected paper(s) in top 5), answerable questions, by type:")
    print(ans.groupby("type")["retrieval_hit"].agg(["mean", "count"]).to_string())
    print(f"overall: {ans['retrieval_hit'].mean():.1%}")


if __name__ == "__main__":
    main()

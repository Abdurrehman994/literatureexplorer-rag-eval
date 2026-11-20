"""Ragas evaluation of the collected endpoint outputs (0 endpoint requests; Lehmus models, VPN).

Metrics (Ragas 0.4 `metrics.collections`), all single-turn, applied to the answerable questions:
  faithfulness         answer claims supported by the 5 retrieved chunks?        (generation)
  answer_relevancy     does the answer address the question? (LLM + embeddings)  (generation)
  context_precision    are chunks relevant to the reference ranked high?         (retrieval)
  context_recall       do the chunks contain the reference answer's claims?      (retrieval)
  factual_correctness  answer vs reference answer, claim-level F1                (end-to-end)
The retrieved chunks come from the chat response itself, i.e. exactly what the generator saw.

Each (question, metric) score is appended to data/ragas/scores_<judge>.jsonl as soon as it is
done; a rerun skips completed pairs (failed ones are retried). Judge calls are also cached on disk
(data/ragas_cache). The `--judge small` run on the same questions measures judge agreement.

Usage:  python src/run_ragas.py --judge large [--workers 4] [--ids Q001,Q002] [--metrics faithfulness,...]
Output: data/ragas/scores_<judge>.jsonl (long) and data/ragas/scores_<judge>.csv (one row per question)
"""
import argparse
import asyncio
import json
import math
import time

import pandas as pd
from openai import AsyncOpenAI
from ragas import DiskCacheBackend
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithReference, ContextRecall,
                                       FactualCorrectness, Faithfulness)

from config import (DATA_DIR, LEHMUS_API_KEY, LEHMUS_BASE_URL, LEHMUS_EMBED_MODEL,
                    LEHMUS_LARGE_MODEL, LEHMUS_SMALL_MODEL, require)

OUT_DIR = DATA_DIR / "ragas"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]
JUDGES = {"large": LEHMUS_LARGE_MODEL, "small": LEHMUS_SMALL_MODEL}
TIMEOUT_S = 600


def build_metrics(llm, embeddings) -> dict:
    """metric name -> (metric object, function mapping a response record to ascore kwargs)."""
    return {
        "faithfulness": (Faithfulness(llm=llm), lambda r: dict(
            user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"])),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=embeddings), lambda r: dict(
            user_input=r["question"], response=r["answer"])),
        "context_precision": (ContextPrecisionWithReference(llm=llm), lambda r: dict(
            user_input=r["question"], reference=r["reference_answer"], retrieved_contexts=r["contexts"])),
        "context_recall": (ContextRecall(llm=llm), lambda r: dict(
            user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["reference_answer"])),
        "factual_correctness": (FactualCorrectness(llm=llm, mode="f1"), lambda r: dict(
            response=r["answer"], reference=r["reference_answer"])),
    }


def load_done(path) -> set:
    done = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["error"] is None:
                done.add((rec["id"], rec["metric"]))
    return done


async def score_one(sem, metric_name, metric, kwargs, row, judge_model, out_file):
    async with sem:
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(metric.ascore(**kwargs), TIMEOUT_S)
            value, error = result.value, None
            value = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
        except Exception as exc:  # record and continue; rerun retries failures
            value, error = None, f"{type(exc).__name__}: {str(exc)[:300]}"
        rec = {"id": row["id"], "metric": metric_name, "value": value, "error": error,
               "judge_model": judge_model, "seconds": round(time.perf_counter() - start, 1)}
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {row['id']} {metric_name:20s} {value if error is None else 'ERROR ' + error[:80]}")


async def run(args) -> None:
    judge_model = JUDGES[args.judge]
    metric_names = [m for m in METRIC_NAMES if not args.metrics or m in args.metrics.split(",")]

    rows = [json.loads(line) for line in (DATA_DIR / "responses.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["expected_behaviour"] == "answer" and r["status"] == 200]
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids.split(",")]
    for r in rows:
        r["answer"] = r["answer"] or ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"scores_{args.judge}.jsonl"
    done = load_done(out_file)
    todo = [(m, r) for r in rows for m in metric_names if (r["id"], m) not in done]
    print(f"Judge {args.judge} ({judge_model}): {len(rows)} questions x {len(metric_names)} metrics, "
          f"{len(todo)} to score ({len(rows) * len(metric_names) - len(todo)} already done)")
    if not todo:  # everything cached: no client needed (works without .env / VPN)
        write_summary(args.judge)
        return

    client = AsyncOpenAI(base_url=require("LEHMUS_BASE_URL", LEHMUS_BASE_URL),
                         api_key=require("LEHMUS_API_KEY", LEHMUS_API_KEY), timeout=TIMEOUT_S)
    # one cache per judge: Ragas' cache key does not include the model name, so a shared cache
    # would silently replay one judge's outputs for the other
    cache = DiskCacheBackend(cache_dir=str(DATA_DIR / "ragas_cache" / args.judge))
    llm = llm_factory(judge_model, client=client, cache=cache, max_tokens=8192)
    embeddings = embedding_factory("openai", model=LEHMUS_EMBED_MODEL, client=client, cache=cache)
    metrics = build_metrics(llm, embeddings)
    sem = asyncio.Semaphore(args.workers)
    await asyncio.gather(*(score_one(sem, m, metrics[m][0], metrics[m][1](r), r, judge_model, out_file)
                           for m, r in todo))
    write_summary(args.judge)


def write_summary(judge: str) -> None:
    long = pd.read_json(OUT_DIR / f"scores_{judge}.jsonl", lines=True)
    long = long[long["error"].isna()].drop_duplicates(["id", "metric"], keep="last")
    wide = long.pivot(index="id", columns="metric", values="value").reset_index()
    resp = pd.read_json(DATA_DIR / "responses.jsonl", lines=True)[["id", "type", "abstained", "retrieval_hit"]]
    wide = resp.merge(wide, on="id", how="inner")
    wide.to_csv(OUT_DIR / f"scores_{judge}.csv", index=False)
    metric_cols = [c for c in wide.columns if c not in ("id", "type", "abstained", "retrieval_hit")]
    print(f"\nMean scores ({judge} judge, n={len(wide)}):")
    print(wide[metric_cols].mean().round(3).to_string())
    errors = pd.read_json(OUT_DIR / f"scores_{judge}.jsonl", lines=True)
    n_err = errors[errors["error"].notna() & ~errors.set_index(["id", "metric"]).index.isin(
        long.set_index(["id", "metric"]).index)].shape[0]
    if n_err:
        print(f"{n_err} scores still failing; rerun the same command to retry them.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ragas evaluation of collected endpoint responses.")
    parser.add_argument("--judge", choices=JUDGES, default="large")
    parser.add_argument("--workers", type=int, default=4, help="parallel judge calls")
    parser.add_argument("--ids", help="comma-separated question ids (default: all answerable)")
    parser.add_argument("--metrics", help="comma-separated subset of metrics")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

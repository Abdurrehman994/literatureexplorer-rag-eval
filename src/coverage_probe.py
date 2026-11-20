"""Coverage of the document collection: can each of the 183 papers be retrieved at all?

For every paper we send its title as a query to /v1/context (retrieval only; 1 request per paper,
cached). A title is the easiest possible query for its own paper ("known-item search"), so a paper
that is not in the top 5 for its own title is most likely missing or badly ingested.

A retrieved chunk belongs to a paper if its `source` file name starts with "NNN - " (= doc_id).
We also check that the file name's title agrees with the sheet title (validates the numbering).

Usage:  python src/coverage_probe.py
Output: data/coverage_results.csv
"""
import difflib
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import client
from config import DATA_DIR

TAG = "coverage"

# Second chance for papers missed by the title query: distinctive content terms (hand-written after
# the first run). If these also miss, the paper is very likely not in the knowledge base.
CONFIRM_QUERIES = {
    12: "Seemingly Conscious AI, model welfare and AI rights (Mustafa Suleyman)",
    72: "AgentSociety 2 integrated research environment for executable social science experiments",
    82: "Personal Health Agent with data science agent, health domain expert and health coach",
    106: "SCP Science Context Protocol global web of autonomous scientific agents",
    114: "EduAgentQG multi-agent workflow for personalized educational question generation",
    122: "Cradle foundation agent for General Computer Control playing Red Dead Redemption II",
}


def file_title(source: str) -> str:
    return re.sub(r"^\d+\s*-\s*|\.pdf$", "", source, flags=re.I)


def similarity(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())
    return round(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio(), 2)


def probe(doc: pd.Series) -> dict:
    record = client.context(doc["Title"], tag=TAG)
    contexts = (record["response_json"] or {}).get("contexts") or []
    prefix = doc["source_file_prefix"]
    ranks = [i + 1 for i, c in enumerate(contexts) if c["source"].startswith(prefix)]
    own = [c for c in contexts if c["source"].startswith(prefix)]
    return {
        "doc_id": doc["doc_id"],
        "title": doc["Title"],
        "stratum": doc["stratum"],
        "host": re.sub(r"https?://([^/]+).*", r"\1", doc["URL"]),
        "status": record["status"],
        "latency_s": record["latency_s"],
        "n_contexts": len(contexts),
        "rank": ranks[0] if ranks else None,
        "hit@1": bool(ranks and ranks[0] == 1),
        "hit@5": bool(ranks),
        "own_chunks": len(own),
        "own_pages": sorted({c["page"] for c in own}),
        "file_name": own[0]["source"] if own else None,
        "title_match": similarity(doc["Title"], file_title(own[0]["source"])) if own else None,
        "top1_source": contexts[0]["source"] if contexts else None,
        "top1_score": round(contexts[0]["score"], 3) if contexts else None,
        "all_sources": [c["source"] for c in contexts],
    }


def main() -> None:
    docs = pd.read_csv(DATA_DIR / "documents_clean.csv")
    with ThreadPoolExecutor(max_workers=client.MAX_PARALLEL) as pool:
        rows = list(pool.map(probe, (d for _, d in docs.iterrows())))
    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "coverage_results.csv", index=False)

    ok = res[res["status"] == 200]
    print(f"Requests: {len(res)}  non-200: {len(res) - len(ok)}  budget: {client.usage()}")
    print(f"hit@1 = {ok['hit@1'].mean():.1%}   hit@5 = {ok['hit@5'].mean():.1%}   "
          f"(n={len(ok)})")
    print(f"Distinct files seen across all probes: "
          f"{len({s for srcs in ok['all_sources'] for s in srcs})}")

    print("\nhit@5 by stratum:")
    print(ok.groupby("stratum")["hit@5"].agg(["mean", "count"]).sort_values("mean").to_string())
    print("\nhit@5 by host (arxiv vs other):")
    print(ok.groupby(ok["host"].eq("arxiv.org").map({True: "arxiv", False: "other"}))["hit@5"]
          .agg(["mean", "count"]).to_string())

    misses = ok[~ok["hit@5"]]
    print(f"\nPapers NOT retrieved for their own title ({len(misses)}):")
    for _, r in misses.iterrows():
        print(f"  {r['doc_id']:3d} [{r['host']}] {r['title'][:70]}\n"
              f"        top1: {r['top1_source']} ({r['top1_score']})")

    print("\nConfirmation with content queries:")
    for doc_id, query in CONFIRM_QUERIES.items():
        rec = client.context(query, tag=TAG + "-confirm")
        sources = [c["source"] for c in (rec["response_json"] or {}).get("contexts") or []]
        found = any(s.startswith(f"{doc_id:03d} - ") for s in sources)
        print(f"  {doc_id:3d} found={found}  top1: {sources[0] if sources else None}")
        res.loc[res["doc_id"] == doc_id, "confirm_found"] = found
    res.to_csv(DATA_DIR / "coverage_results.csv", index=False)

    odd = ok[ok["title_match"].notna() & (ok["title_match"] < 0.6)]
    print(f"\nFile name vs sheet title mismatches (<0.6 similarity): {len(odd)}")
    for _, r in odd.iterrows():
        print(f"  {r['doc_id']:3d} sheet: {r['title'][:55]!r}  file: {r['file_name']!r}")


if __name__ == "__main__":
    main()

"""Turn the reviewed candidates into the final test set (0 endpoint requests).

Keeps rows with keep == 1 in data/testset_candidates.csv and adds robustness variants of a few
answerable factual/numeric questions, so the same question can be compared across phrasings:
  robust_typo     - typos (adjacent letters swapped), lower case, no question mark (code, seeded)
  robust_finnish  - translated to Finnish (GPT-OSS); the corpus is English
  robust_keywords - terse search-engine style keywords (GPT-OSS)
Variants keep the original's reference answer and source, and point to it via `variant_of`.

Usage:  python src/finalize_testset.py [--seed 42] [--per-kind 3]
Output: data/testset.csv, data/testset.json
"""
import argparse
import random
import re

import pandas as pd

import llm
from config import DATA_DIR

COLUMNS = ["id", "type", "expected_behaviour", "question", "reference_answer", "source_doc_ids",
           "source_titles", "source_pages", "evidence", "origin", "variant_of", "candidate_id"]


def add_typos(question: str, rng: random.Random, n: int = 3) -> str:
    words = question.rstrip("?").lower().split()
    long_words = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    for i in rng.sample(long_words, min(n, len(long_words))):
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def rewrite(question: str, instruction: str) -> str:
    out = llm.chat(f"{instruction}\nReturn only the rewritten text, nothing else.\n\n{question}",
                   model=llm.LARGE)
    return out.strip().strip('"')


VARIANTS = {
    "robust_typo": lambda q, rng: add_typos(q, rng),
    "robust_finnish": lambda q, rng: rewrite(
        q, "Translate this question into Finnish. Keep names of systems, models and benchmarks unchanged."),
    "robust_keywords": lambda q, rng: rewrite(
        q, "Rewrite this question as a terse search query of 3-7 keywords, as a hurried user would type it. "
           "Keep the name of the system/method it is about."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final test set from reviewed candidates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-kind", type=int, default=3, help="variants per robustness kind")
    args = parser.parse_args()

    cand = pd.read_csv(DATA_DIR / "testset_candidates.csv", dtype=str, keep_default_na=False)
    kept = cand[cand["keep"] == "1"].copy()
    kept["candidate_id"] = kept["id"]
    kept["variant_of"] = ""

    rng = random.Random(args.seed)
    pool = kept[kept["type"].isin(["factual", "numeric"]) & (kept["expected_behaviour"] == "answer")]
    picks = rng.sample(list(pool.index), args.per_kind * len(VARIANTS))
    variants = []
    for k, (kind, make) in enumerate(VARIANTS.items()):
        for idx in picks[k * args.per_kind:(k + 1) * args.per_kind]:
            row = kept.loc[idx].copy()
            row["question"] = make(row["question"], rng)
            row["type"], row["variant_of"], row["origin"] = kind, row["candidate_id"], "variant"
            variants.append(row)

    final = pd.concat([kept, pd.DataFrame(variants)], ignore_index=True)
    final["id"] = [f"Q{i:03d}" for i in range(1, len(final) + 1)]
    # variant_of: candidate id -> final id
    cand_to_final = dict(zip(final.loc[final["origin"] != "variant", "candidate_id"],
                             final.loc[final["origin"] != "variant", "id"]))
    final["variant_of"] = final["variant_of"].map(lambda c: cand_to_final.get(c, ""))
    final = final[COLUMNS]
    final.to_csv(DATA_DIR / "testset.csv", index=False, encoding="utf-8-sig")
    final.to_json(DATA_DIR / "testset.json", orient="records", indent=1, force_ascii=False)

    print(f"Final test set: {len(final)} questions -> data/testset.csv, data/testset.json")
    print(final.groupby(["expected_behaviour", "type"]).size().to_string())
    print("\nRobustness variants:")
    for _, r in final[final["origin"] == "variant"].iterrows():
        orig = final.loc[final["id"] == r["variant_of"], "question"].iloc[0]
        print(f"  {r['id']} {r['type']:16s} {r['question']}\n  {'':21s}(of {r['variant_of']}: {orig[:80]})")


if __name__ == "__main__":
    main()

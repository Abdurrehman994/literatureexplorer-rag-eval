"""Select a stratified sample of papers to build the Ragas test set from (0 endpoint requests).

Strata are the sheet's categories; categories with fewer than MIN_STRATUM papers are merged
("Applications: other", "Other"). Papers are allocated proportionally to stratum size
(largest remainder, at least 1 per stratum) and drawn at random with a fixed seed.

Note: the sheet uses "Num" 92 twice, so from that row on the knowledge-base file names
("NNN - Title.pdf") are one higher than "Num". `doc_id` (row position, 1..183) matches the files.

Usage:  python src/select_documents.py [--n 30] [--seed 42]
"""
import argparse

import pandas as pd

from config import DATA_DIR

MIN_STRATUM = 4


def stratum(category: str, counts: pd.Series) -> str:
    if counts[category] >= MIN_STRATUM:
        return category
    return "Applications: other" if category.startswith("Applications") else "Other"


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation with largest remainder and a minimum of 1 per stratum."""
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int).clip(lower=1)
    while alloc.sum() < n:
        remainder = (exact - alloc).sort_values(ascending=False)
        alloc[remainder.index[0]] += 1
    while alloc.sum() > n:  # only if the minimum of 1 pushed us over
        alloc[(alloc - exact).idxmax()] -= 1
    return alloc.clip(upper=sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=30, help="number of papers to select")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = pd.read_csv(DATA_DIR / "documents.csv", skiprows=1)  # row 1 is a sheet title
    docs.insert(0, "doc_id", range(1, len(docs) + 1))
    docs["source_file_prefix"] = docs["doc_id"].map(lambda i: f"{i:03d} - ")
    counts = docs["Category"].value_counts()
    docs["stratum"] = docs["Category"].map(lambda c: stratum(c, counts))
    docs.to_csv(DATA_DIR / "documents_clean.csv", index=False)

    alloc = allocate(docs["stratum"].value_counts(), args.n)
    selected = pd.concat(
        docs[docs["stratum"] == s].sample(k, random_state=args.seed) for s, k in alloc.items()
    ).sort_values("doc_id")
    selected.to_csv(DATA_DIR / "selected_documents.csv", index=False)

    summary = pd.DataFrame({"papers": docs["stratum"].value_counts(), "selected": alloc})
    print(summary.sort_values("papers", ascending=False).to_string())
    print(f"\nSelected {len(selected)} of {len(docs)} papers -> data/selected_documents.csv")
    print("Hosts:", selected["URL"].str.extract(r"https?://([^/]+)")[0].value_counts().to_dict())


if __name__ == "__main__":
    main()

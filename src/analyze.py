"""Combine all results into report tables and figures (0 endpoint requests, 0 judge calls).

Inputs : data/responses.jsonl, data/answer_behaviour.csv, data/ragas/scores_{large,small}.csv,
         data/coverage_results.csv, data/system_tests/*.csv, data/raw_responses/endpoint_cache.jsonl
Outputs: data/analysis/*.csv, data/analysis/summary.md (all key numbers), report/figures/*.png

Usage:  python src/analyze.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_testset import norm, quote_found  # noqa: E402
from config import DATA_DIR, ROOT, RAW_DIR  # noqa: E402

OUT = DATA_DIR / "analysis"
FIG = ROOT / "report" / "figures"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]
METRIC_LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy",
                 "context_precision": "Context precision", "context_recall": "Context recall",
                 "factual_correctness": "Factual correctness"}
TYPE_GROUPS = {"factual": "Single-hop", "numeric": "Single-hop", "conceptual": "Conceptual",
               "multi_hop": "Multi-hop", "cross_paper": "Cross-paper", "false_premise": "False premise",
               "robust_typo": "Robustness", "robust_finnish": "Robustness", "robust_keywords": "Robustness"}

# reference palette (light mode, fixed slot order) and chart ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
})

lines: list[str] = []  # summary.md


def section(title: str) -> None:
    lines.append(f"\n## {title}\n")


def table(df: pd.DataFrame, name: str, floatfmt: int = 3) -> None:
    df.to_csv(OUT / f"{name}.csv")
    lines.append(df.round(floatfmt).to_markdown())
    lines.append("")


def bootstrap_ci(x: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x.dropna().to_numpy()
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def load() -> pd.DataFrame:
    resp = pd.read_json(DATA_DIR / "responses.jsonl", lines=True, dtype={"source_doc_ids": str, "source_pages": str})
    beh = pd.read_csv(DATA_DIR / "answer_behaviour.csv")[
        ["id", "behaviour", "corrected_premise", "followed_injection", "leaked_instructions",
         "asked_clarification", "answer_language", "expectation_met"]]
    df = resp.drop(columns=["abstained"]).merge(beh, on="id", how="left")
    df["group"] = df["type"].map(TYPE_GROUPS).fillna(df["type"])
    df["responded"] = df["behaviour"].isin(["answered", "partial"])
    # chunk-level retrieval: is the evidence quote of the reference answer inside a retrieved chunk?
    def evidence_retrieved(r):
        quotes = [q for q in str(r["evidence"]).split(" | ") if q.strip()]
        if not quotes or not r["contexts"]:
            return None
        ctx = norm(" ".join(r["contexts"]))
        return all(quote_found(q, ctx) for q in quotes)
    df["evidence_retrieved"] = df.apply(evidence_retrieved, axis=1)
    return df


def ragas_scores(judge: str) -> pd.DataFrame | None:
    path = DATA_DIR / "ragas" / f"scores_{judge}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)[["id"] + [m for m in METRICS if m in pd.read_csv(path).columns]]


def analyse_ragas(df: pd.DataFrame) -> pd.DataFrame | None:
    large = ragas_scores("large")
    if large is None:
        return None
    ans = df[df["expected_behaviour"] == "answer"].merge(large, on="id", how="inner")
    section(f"Ragas scores - GPT-OSS judge (n={len(ans)} answerable questions scored)")
    rows = {}
    for label, sub in [("all answerable", ans), ("responded (answered/partial)", ans[ans["responded"]]),
                       ("abstained", ans[ans["behaviour"] == "abstained"])]:
        for m in METRICS:
            if m in sub:
                lo, hi = bootstrap_ci(sub[m])
                rows[(label, m)] = {"mean": sub[m].mean(), "ci_low": lo, "ci_high": hi, "n": sub[m].notna().sum()}
    overall = pd.DataFrame(rows).T
    table(overall, "ragas_overall")
    abst = ans[ans["behaviour"] == "abstained"]
    lines.append("Note: Ragas faithfulness scores abstentions arbitrarily, depending on their wording: a bare "
                 "\"I don't know.\" contains no claims and scores 0, while \"The context does not provide X\" is "
                 "treated as a supported claim and scores 1 "
                 f"(abstentions: {(abst['faithfulness'] == 0).sum()}x 0, {(abst['faithfulness'] == 1).sum()}x 1, "
                 f"n={len(abst)}). Faithfulness is therefore only meaningful on the 'responded' subset.\n")

    section("Ragas means by question group (responded subset | all answerable)")
    by_group_resp = ans[ans["responded"]].groupby("group")[METRICS].mean()
    by_group_resp["n"] = ans[ans["responded"]].groupby("group").size()
    table(by_group_resp, "ragas_by_group_responded")
    by_group_all = ans.groupby("group")[METRICS].mean()
    by_group_all["n"] = ans.groupby("group").size()
    table(by_group_all, "ragas_by_group_all")
    return ans


def analyse_judges(ans: pd.DataFrame | None) -> None:
    large, small = ragas_scores("large"), ragas_scores("small")
    if large is None or small is None:
        return
    both = large.merge(small, on="id", suffixes=("_large", "_small"))
    section(f"Judge agreement GPT-OSS vs Gemma (n={len(both)} questions scored by both)")
    rows = {}
    for m in METRICS:
        a, b = both.get(f"{m}_large"), both.get(f"{m}_small")
        if a is None or b is None:
            continue
        ok = a.notna() & b.notna()
        rows[m] = {"mean_gpt_oss": a[ok].mean(), "mean_gemma": b[ok].mean(),
                   "spearman": a[ok].corr(b[ok], method="spearman"),
                   "mean_abs_diff": (a[ok] - b[ok]).abs().mean(), "n": ok.sum()}
    table(pd.DataFrame(rows).T, "judge_agreement")


def analyse_behaviour(df: pd.DataFrame) -> None:
    section("Answer behaviour (GPT-OSS classification)")
    table(pd.crosstab(df["expected_behaviour"], df["behaviour"], margins=True), "behaviour_by_expected")
    met = df.groupby("expected_behaviour")["expectation_met"].agg(["mean", "sum", "count"])
    table(met, "expectation_met")
    by_type = pd.crosstab(df["type"], df["behaviour"])
    table(by_type, "behaviour_by_type")
    special = {
        "false_premise corrected": (df["type"] == "false_premise", "corrected_premise"),
        "injection followed": (df["type"] == "prompt_injection", "followed_injection"),
        "instructions leaked": (df["type"] == "prompt_injection", "leaked_instructions"),
        "ambiguous: asked clarification": (df["type"] == "ambiguous", "asked_clarification"),
    }
    for label, (mask, col) in special.items():
        lines.append(f"- {label}: {(df.loc[mask, col] == True).sum()}/{mask.sum()}")
    lines.append(f"- answer languages: {df['answer_language'].value_counts().to_dict()}")
    oos = df[(df["expected_behaviour"] == "abstain") & ~df["behaviour"].isin(["abstained", "blocked"])]
    for _, r in oos.iterrows():
        lines.append(f"- should have abstained: {r['id']} \"{r['question']}\" -> \"{str(r['answer'])[:200]}\"")
    lines.append("")


def analyse_retrieval(df: pd.DataFrame, ans: pd.DataFrame | None) -> None:
    section("Retrieval (answerable questions)")
    a = df[df["expected_behaviour"] == "answer"]
    gen = a[a["evidence_retrieved"].notna()]
    t = pd.DataFrame({
        "paper_in_top5": a.groupby("group")["retrieval_hit"].mean(),
        "evidence_chunk_in_top5": gen.groupby("group")["evidence_retrieved"].mean(),
        "n": a.groupby("group").size()})
    t.loc["ALL"] = [a["retrieval_hit"].mean(), gen["evidence_retrieved"].mean(), len(a)]
    table(t, "retrieval_by_group")
    section("Why does the system abstain on answerable questions?")
    abst = a[a["behaviour"] == "abstained"]
    diag = pd.DataFrame({
        "abstained": [len(abst)],
        "paper_not_retrieved": [(~abst["retrieval_hit"]).sum()],
        "paper_retrieved_but_evidence_chunk_missing": [(abst["retrieval_hit"] & (abst["evidence_retrieved"] == False)).sum()],
        "evidence_chunk_retrieved_yet_abstained": [(abst["evidence_retrieved"] == True).sum()],
    })
    table(diag, "abstention_diagnosis")
    resp = a[a["responded"]]
    lines.append(f"Responded questions with the evidence chunk retrieved: "
                 f"{(resp['evidence_retrieved'] == True).sum()}/{resp['evidence_retrieved'].notna().sum()}\n")


def analyse_coverage() -> None:
    cov = pd.read_csv(DATA_DIR / "coverage_results.csv")
    section("Collection coverage (title query per paper, /v1/context)")
    lines.append(f"- papers: {len(cov)}; hit@1 {cov['hit@1'].mean():.1%}; hit@5 {cov['hit@5'].mean():.1%}")
    missing = cov[cov["confirm_found"] == False] if "confirm_found" in cov else cov[~cov["hit@5"]]
    lines.append(f"- not retrievable even with content queries ({len(missing)}): "
                 + "; ".join(f"{r.doc_id} {r.title[:60]}" for r in missing.itertuples()))
    lines.append("")


def analyse_latency() -> pd.DataFrame:
    recs = [json.loads(line) for line in (RAW_DIR / "endpoint_cache.jsonl").open(encoding="utf-8")]
    lat = pd.DataFrame([{"tag": r["tag"], "path": r["path"], "status": r["status"], "latency_s": r["latency_s"],
                         "completion_tokens": ((r["response_json"] or {}).get("usage") or {}).get("completion_tokens")
                         if isinstance(r["response_json"], dict) else None}
                        for r in recs])
    ok = lat[(lat["status"] == 200) & lat["path"].isin(["/v1/context", "/v1/chat/completions"])]
    section("Latency (successful requests, all runs)")
    stats = ok.groupby("path")["latency_s"].describe(percentiles=[.5, .95])[["count", "50%", "95%", "max"]]
    table(stats, "latency_by_route", floatfmt=2)
    chat = ok[(ok["path"] == "/v1/chat/completions") & ok["completion_tokens"].notna()]
    lines.append(f"- chat latency vs completion tokens: Spearman "
                 f"{chat['latency_s'].corr(chat['completion_tokens'], method='spearman'):.2f} (n={len(chat)})")
    sys_lat = pd.read_csv(DATA_DIR / "system_tests" / "latency.csv")
    lines.append(f"- sequential chat test: short questions median {sys_lat[sys_lat.kind == 'short'].latency_s.median():.2f} s, "
                 f"long questions median {sys_lat[sys_lat.kind == 'long'].latency_s.median():.2f} s; "
                 f"first request of run {sys_lat[sys_lat.possibly_cold].latency_s.iloc[0]:.2f} s")
    errors = lat[lat["status"] != 200]
    lines.append(f"- non-200 responses (all deliberately provoked in api tests): {errors['status'].value_counts().to_dict()}")
    lines.append(f"- endpoint requests used in total: {len(recs)} records, "
                 f"{sum(len(r['attempts']) for r in recs)} HTTP calls\n")
    return ok


def analyse_system_tests() -> None:
    section("API / error-handling tests")
    api = pd.read_csv(DATA_DIR / "system_tests" / "api_tests.csv")
    api["response"] = api["response"].str.replace(r"\s+", " ", regex=True).str[:160]
    table(api[["test", "status", "expected", "response"]].set_index("test"), "api_tests_summary")
    section("Consistency (8 questions x 3 runs)")
    con = pd.read_csv(DATA_DIR / "system_tests" / "consistency.csv")
    table(con[["id", "first_behaviour", "same_chunks_all_runs", "identical_answers_all_runs",
               "distinct_answers"]].set_index("id"), "consistency_summary")


def worst_examples(ans: pd.DataFrame | None) -> None:
    if ans is None:
        return
    section("Examples")
    resp = ans[ans["responded"]]
    for title, sub in [("Low faithfulness (possible hallucination), responded", resp.nsmallest(5, "faithfulness")),
                       ("Low factual correctness although faithful", resp[resp["faithfulness"] >= 0.8].nsmallest(5, "factual_correctness"))]:
        lines.append(f"**{title}**\n")
        for _, r in sub.iterrows():
            lines.append(f"- {r['id']} ({r['type']}) F={r['faithfulness']:.2f} FC={r['factual_correctness']:.2f} "
                         f"CR={r['context_recall']:.2f}: Q: {r['question'][:140]}\n  A: {str(r['answer'])[:220]}\n"
                         f"  REF: {r['reference_answer'][:220]}")
        lines.append("")


# ------------------------------------------------------------------ figures
def fig_ragas(ans: pd.DataFrame) -> None:
    subsets = [("All answerable questions", ans), ("Only questions it answered", ans[ans["responded"]])]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    y = np.arange(len(METRICS))
    h = 0.36
    for i, (label, sub) in enumerate(subsets):
        means = [sub[m].mean() for m in METRICS]
        cis = [bootstrap_ci(sub[m]) for m in METRICS]
        err = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T
        pos = y - h / 2 + i * h
        ax.barh(pos, means, height=h - 0.04, color=SERIES[i], label=f"{label} (n={len(sub)})",
                edgecolor=SURFACE, linewidth=1)
        ax.errorbar(means, pos, xerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for p, v, (lo, hi) in zip(pos, means, cis):
            ax.text(hi + 0.015, p, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [METRIC_LABELS[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean score (0-1), 95% bootstrap CI")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8)
    fig.savefig(FIG / "fig1_ragas_metrics.png")
    plt.close(fig)


def fig_behaviour(df: pd.DataFrame) -> None:
    order = ["Single-hop", "Conceptual", "Multi-hop", "Cross-paper", "False premise", "Robustness",
             "out_of_scope", "prompt_injection", "ambiguous"]
    names = {"out_of_scope": "Out of scope*", "prompt_injection": "Prompt injection*", "ambiguous": "Ambiguous*"}
    behaviours = ["answered", "partial", "abstained", "blocked"]
    ct = pd.crosstab(df["group"], df["behaviour"]).reindex(index=order, columns=behaviours, fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    left = np.zeros(len(order))
    y = np.arange(len(order))
    for i, b in enumerate(behaviours):
        ax.barh(y, share[b], left=left, height=0.62, color=SERIES[i], label=b.capitalize(),
                edgecolor=SURFACE, linewidth=1.5)
        for yi, (s, c, l) in enumerate(zip(share[b], ct[b], left)):
            if s >= 0.09:
                ax.text(l + s / 2, yi, str(c), ha="center", va="center", fontsize=7.5, color=SURFACE if i == 0 else INK)
        left += share[b].to_numpy()
    ax.set_yticks(y, [f"{names.get(g, g)} (n={ct.loc[g].sum()})" for g in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=4, fontsize=8)
    ax.text(0, -0.2, "* expected behaviour: abstain / resist / ask for clarification; all others should be answered",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.savefig(FIG / "fig2_behaviour.png")
    plt.close(fig)


def fig_latency(ok: pd.DataFrame) -> None:
    routes = [("/v1/context", "Retrieval only\n/v1/context"), ("/v1/chat/completions", "Retrieval + generation\n/v1/chat/completions")]
    data = [ok.loc[ok["path"] == p, "latency_s"].to_numpy() for p, _ in routes]
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": SERIES[0],
                                "markeredgecolor": SURFACE, "alpha": 0.7},
                    whiskerprops={"color": INK2}, capprops={"color": INK2})
    for patch in bp["boxes"]:
        patch.set(facecolor="#9ec5f4", edgecolor=SERIES[0])
    for i, d in enumerate(data, start=1):
        ax.text(d.max() + 0.08, i, f"p50 {np.median(d):.2f} s · p95 {np.percentile(d, 95):.2f} s · n={len(d)}",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([1, 2], [label for _, label in routes])
    ax.invert_yaxis()
    ax.set_xlim(0, max(d.max() for d in data) + 3.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Latency per request (s)")
    fig.savefig(FIG / "fig3_latency.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    lines.append("# Evaluation summary (generated by src/analyze.py)\n")
    df = load()
    df.drop(columns=["contexts"]).to_csv(OUT / "per_question.csv", index=False, encoding="utf-8-sig")
    ans = analyse_ragas(df)
    analyse_judges(ans)
    analyse_behaviour(df)
    analyse_retrieval(df, ans)
    analyse_coverage()
    ok = analyse_latency()
    analyse_system_tests()
    worst_examples(ans)
    if ans is not None:
        fig_ragas(ans)
        ans.drop(columns=["contexts"]).to_csv(OUT / "per_question_ragas.csv", index=False, encoding="utf-8-sig")
    fig_behaviour(df)
    fig_latency(ok)
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFigures -> {FIG}")


if __name__ == "__main__":
    main()

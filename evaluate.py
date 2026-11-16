"""Evaluate ReturnsIQ on a labeled question set and write eval/report.md.

Metrics
  Retrieval  hit@1, hit@3, MRR      Is the right policy section found, and how high?
  Answers    keyword accuracy       Does the answer contain the expected fact?
             groundedness           Share of answer sentences supported by retrieved text.
             citation rate          Share of answers that cite a source.
  Agent      routing accuracy       Did data questions go to the right SQL tool?
             refusal accuracy       Were out-of-scope questions declined?
             false refusals         Were answerable questions wrongly declined?

Usage:
    python evaluate.py                              # extractive mode, TF-IDF
    python evaluate.py --llm ollama                 # with a local model
    python evaluate.py --compare-retrievers         # TF-IDF vs dense vs hybrid
"""
import argparse
import json
from pathlib import Path

from returnsiq.agent import ReturnsAgent
from returnsiq.ingest import load_chunks
from returnsiq.llm import get_llm
from returnsiq.retriever import Retriever, dense_available

ROOT = Path(__file__).parent


def retrieval_metrics(retriever: Retriever, items: list[dict]) -> dict:
    hit1 = hit3 = rr = 0.0
    for it in items:
        ids = [c.id for c, _ in retriever.search(it["q"], k=5)]
        ranks = [ids.index(s) + 1 for s in it["sources"] if s in ids]
        best = min(ranks) if ranks else None
        hit1 += best == 1
        hit3 += best is not None and best <= 3
        rr += 1 / best if best else 0
    n = len(items)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "MRR": rr / n}


def groundedness(agent: ReturnsAgent, answer_text: str, chunk_ids: list[str]) -> float:
    return agent.support(answer_text, chunk_ids)[0]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="none", choices=["none", "ollama"])
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense", "hybrid"])
    p.add_argument("--db", default=None)
    p.add_argument("--compare-retrievers", action="store_true")
    a = p.parse_args()

    agent = ReturnsAgent(get_llm(a.llm, a.model), a.db, a.retriever)
    items = json.loads((ROOT / "eval" / "questions.json").read_text())
    if agent.db:
        for it in items:
            it["q"] = it["q"].replace("{sample_customer}", str(agent.db.sample_customer()))
    else:
        items = [it for it in items if it["type"] != "tool"]
        print("[warning] No database: skipping data-tool questions.")

    policy = [it for it in items if it["type"] == "policy"]
    tools = [it for it in items if it["type"] == "tool"]
    oos = [it for it in items if it["type"] == "out_of_scope"]

    rows, kw_ok, grounded, cited, false_refusals, fallbacks = [], 0, [], 0, 0, 0
    for it in policy:
        ans = agent.ask(it["q"])
        ok = ans.route != "refused" and all(k.lower() in ans.text.lower() for k in it["keywords"])
        kw_ok += ok
        false_refusals += ans.route == "refused"
        cited += bool(ans.sources)
        fallbacks += ans.fallback
        g = groundedness(agent, ans.text, [cid for cid, _ in ans.retrieved]) if ans.route != "refused" else 0.0
        grounded.append(g)
        rows.append(("policy", it["q"], ans.route, "pass" if ok else "FAIL"))

    route_ok = 0
    for it in tools:
        ans = agent.ask(it["q"])
        ok = ans.route == it["tool"]
        route_ok += ok
        rows.append(("tool", it["q"], ans.route, "pass" if ok else "FAIL"))

    refuse_ok = 0
    for it in oos:
        ans = agent.ask(it["q"])
        ok = ans.route == "refused"
        refuse_ok += ok
        rows.append(("out of scope", it["q"], ans.route, "pass" if ok else "FAIL"))

    summary = {
        "Policy answer accuracy (keywords)": kw_ok / len(policy),
        "Groundedness (avg)": sum(grounded) / len(grounded),
        "Citation rate": cited / len(policy),
        "False refusals": false_refusals / len(policy),
        **({"LLM answers replaced by verified fallback": fallbacks / len(policy)} if a.llm != "none" else {}),
        "Tool routing accuracy": route_ok / len(tools) if tools else float("nan"),
        "Out-of-scope refusal accuracy": refuse_ok / len(oos),
    }

    methods = ["tfidf"] + (["dense", "hybrid"] if a.compare_retrievers and dense_available() else [])
    chunks = load_chunks()
    retr = {m: (agent.retriever if m == a.retriever else Retriever(chunks, m)) for m in methods}
    retr_results = {m: retrieval_metrics(r, policy) for m, r in retr.items()}

    lines = [
        "# ReturnsIQ evaluation report", "",
        f"Mode: LLM = `{a.llm}`{' (' + a.model + ')' if a.llm != 'none' else ''}, retriever = `{a.retriever}`. "
        f"{len(policy)} policy, {len(tools)} data-tool, and {len(oos)} out-of-scope questions.", "",
        "## Summary", "", "| Metric | Score |", "| --- | --- |",
        *[f"| {k} | {pct(v)} |" for k, v in summary.items()], "",
        "## Retrieval quality (policy questions)", "", "| Retriever | Hit@1 | Hit@3 | MRR |", "| --- | --- | --- | --- |",
        *[f"| {m} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {r['MRR']:.2f} |" for m, r in retr_results.items()], "",
    ]
    if a.compare_retrievers and not dense_available():
        lines += ["_Dense and hybrid skipped: run `pip install sentence-transformers` to compare._", ""]
    lines += ["## Per-question results", "", "| Type | Question | Route | Result |", "| --- | --- | --- | --- |",
              *[f"| {t} | {q} | {r} | {res} |" for t, q, r, res in rows]]
    (ROOT / "eval" / "report.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines[:len(lines) - len(rows) - 4]))
    fails = [r for r in rows if r[3] == "FAIL"]
    print(f"\n{len(fails)} failed question(s)" + (":" if fails else "."))
    for t, q, r, _ in fails:
        print(f"  - [{t}] {q} -> {r}")
    print(f"\nFull report: {ROOT / 'eval' / 'report.md'}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")

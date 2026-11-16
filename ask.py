"""Ask ReturnsIQ a question.

Examples:
    python ask.py "How quickly must I report a damaged item?"
    python ask.py "Would customer 12346 pay the restocking fee?"
    python ask.py --llm ollama "Can I return a gift card?"
    python ask.py                      # interactive mode
"""
import argparse

from returnsiq.agent import ReturnsAgent
from returnsiq.llm import get_llm


def show(ans) -> None:
    print(f"\n[{ans.route}]{' ' + str(ans.tool_args) if ans.tool_args else ''}")
    print(ans.text)
    if ans.retrieved:
        print("retrieved:", ", ".join(f"{cid} ({s})" for cid, s in ans.retrieved))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--llm", default="none", choices=["none", "ollama"])
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense", "hybrid"])
    p.add_argument("--db", default=None, help="path to returns.db (defaults to ../returns-analytics/returns.db)")
    a = p.parse_args()

    agent = ReturnsAgent(get_llm(a.llm, a.model), a.db, a.retriever)
    try:
        if a.question:
            show(agent.ask(" ".join(a.question)))
            return
        print("ReturnsIQ. Ask about returns policy or returns data. Type 'quit' to exit.")
        while (q := input("\n> ").strip()).lower() not in {"quit", "exit", ""}:
            show(agent.ask(q))
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()

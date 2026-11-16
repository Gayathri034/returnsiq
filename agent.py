"""The agent: route each question to policy search (RAG) or to a SQL tool."""
import json
import re
from dataclasses import dataclass, field

from .ingest import load_chunks, split_sentences
from .retriever import Retriever
from .tools import TOOLS, ReturnsDB

REFUSAL = "I couldn't find this in the returns policy documents or the available data tools."
MIN_RETRIEVAL_SCORE = {"tfidf": 0.10, "dense": 0.35, "hybrid": 0.30}  # below this, refuse
MIN_QUERY_COVERAGE = 0.5  # refuse if most of the question's key words never appear in the policies
SUPPORT_THRESHOLD = 0.35  # TF-IDF cosine for a sentence to count as supported by a source
MIN_GROUNDEDNESS = 0.5    # LLM answers below this are replaced by the quoted section


@dataclass
class Answer:
    question: str
    route: str                      # "policy_search", a tool name, or "refused"
    text: str
    sources: list[str] = field(default_factory=list)            # chunk ids cited
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    tool_args: dict = field(default_factory=dict)
    fallback: bool = False          # True when the LLM answer failed verification


class ReturnsAgent:
    def __init__(self, llm=None, db_path=None, retriever_method: str = "tfidf"):
        self.llm = llm
        self.method = retriever_method
        self.retriever = Retriever(load_chunks(), retriever_method)
        self.db = None
        try:
            self.db = ReturnsDB(db_path) if db_path else ReturnsDB()
        except FileNotFoundError as e:
            print(f"[warning] Data tools disabled: {e}")
        self.countries = self.db.countries() if self.db else []

    # ---------------- routing ----------------
    def _rule_route(self, q: str) -> tuple[str, dict]:
        ql = q.lower()
        if self.db:
            m = re.search(r"customer\s*(?:id|#)?\s*(\d{4,6})", ql)
            if m:
                return "check_restocking_fee", {"customer_id": m.group(1)}
            if "top" in ql and "product" in ql:
                m = re.search(r"top\s+(\d+)", ql)
                return "top_returned_products", {"n": int(m.group(1)) if m else 5}
            if "return rate" in ql:
                for c in self.countries:
                    if re.search(rf"\b{re.escape(c.lower())}\b", ql):
                        return "country_return_rate", {"country": c}
                if any(w in ql for w in ("month", "trend", "over time")):
                    return "monthly_return_rate", {}
                if any(w in ql for w in ("overall", "total", "our return rate", "current return rate")):
                    return "overall_return_rate", {}
        return "policy_search", {}

    def _llm_route(self, q: str) -> tuple[str, dict] | None:
        tool_list = "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())
        prompt = (
            "You route questions for a returns assistant. Choose exactly one option.\n"
            "- policy_search: questions about return rules, refunds, fees, damage, abuse review, grading, international returns.\n"
            f"{tool_list}\n\n"
            'Reply with JSON only, like {"tool": "policy_search", "args": {}}.\n'
            f"Question: {q}\nJSON:"
        )
        try:
            raw = self.llm.generate(prompt)
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            tool, args = data.get("tool"), data.get("args") or {}
            if tool == "policy_search":
                return tool, {}
            if tool in TOOLS and self.db and self._args_grounded(tool, args, q):
                return tool, args
        except (ValueError, KeyError, RuntimeError):
            pass
        return None  # fall back to rules

    def _args_grounded(self, tool: str, args: dict, q: str) -> bool:
        """Reject tool calls whose arguments the model invented: every required
        argument must actually appear in the user's question."""
        ql = q.lower()
        if tool == "check_restocking_fee":
            return str(args.get("customer_id", "")).strip() != "" and str(args["customer_id"]) in q
        if tool == "country_return_rate":
            c = str(args.get("country", ""))
            return c in self.countries and c.lower() in ql
        if tool == "top_returned_products":
            return "product" in ql
        # No-argument data tools: only for questions about the data, not the fee rule.
        return "return rate" in ql and "fee" not in ql

    def route(self, q: str) -> tuple[str, dict]:
        rule = self._rule_route(q)
        if rule[0] != "policy_search" or not self.llm:
            return rule  # explicit data patterns are matched deterministically
        return self._llm_route(q) or rule

    # ---------------- answering ----------------
    def _answer_policy(self, q: str) -> Answer:
        hits = self.retriever.search(q, k=3)
        retrieved = [(c.id, round(s, 3)) for c, s in hits]
        if (hits[0][1] < MIN_RETRIEVAL_SCORE[self.method]
                or self.retriever.coverage(q) < MIN_QUERY_COVERAGE):
            return Answer(q, "refused", REFUSAL, retrieved=retrieved)

        if self.llm is None:
            return self._extractive(q, hits, retrieved)

        context = "\n\n".join(f"[{c.id}] {c.heading}: {c.text}" for c, _ in hits)
        prompt = (
            "You are a returns assistant. Answer the question in 1-3 sentences using ONLY the policy excerpts below, "
            "reusing their wording where possible. Put the excerpt id in square brackets after the answer, "
            "e.g. [01_return_policy#return-window]. "
            f'Only if none of the excerpts are relevant, reply: "{REFUSAL}"\n\n'
            f"Excerpts:\n{context}\n\nQuestion: {q}\nAnswer:"
        )
        text = self.llm.generate(prompt)

        # Verify before trusting the model. Retrieval already passed both relevance
        # gates, so a refusal here is a false refusal, and an answer that the
        # retrieved text does not support is a possible hallucination. Either way,
        # fall back to quoting the best-matching section.
        grounded, best_id = self.support(text, [c.id for c, _ in hits])
        refused = REFUSAL.lower()[:30] in text.lower()
        if refused or grounded < MIN_GROUNDEDNESS:
            ans = self._extractive(q, hits, retrieved)
            ans.fallback = True
            return ans
        cited = [c.id for c, _ in hits if c.id in text]
        if not cited and best_id:  # model skipped citations: attach the supporting source
            text, cited = f"{text} [{best_id}]", [best_id]
        return Answer(q, "policy_search", text, sources=cited, retrieved=retrieved)

    def support(self, text: str, chunk_ids: list[str]) -> tuple[float, str | None]:
        """Share of the answer's sentences supported by the given chunks (TF-IDF
        cosine >= SUPPORT_THRESHOLD), and the chunk that supports it most."""
        chunks = {c.id: c for c in self.retriever.chunks}
        support, owner = [], []
        for cid in chunk_ids:
            if cid in chunks:
                for s in split_sentences(chunks[cid].text):
                    support.append(s)
                    owner.append(cid)
        claims = [re.sub(r"\[[^\]]+\]", "", s).strip() for s in split_sentences(text)]
        claims = [c for c in claims if len(c.split()) >= 4]
        if not claims or not support:
            return 0.0, None
        sims = self.retriever.sentence_similarity(claims, support)
        best = owner[int(sims.max(axis=0).argmax())]
        return float((sims.max(axis=1) >= SUPPORT_THRESHOLD).mean()), best

    def _extractive(self, q: str, hits, retrieved) -> Answer:
        """No-LLM mode: quote the best-matching policy section verbatim, with its citation.
        Sections are 2-4 sentences, so the full rule (and any exceptions) stays intact."""
        top = hits[0][0]
        text = f"{top.heading}: {top.text} [{top.id}]"
        return Answer(q, "policy_search", text, sources=[top.id], retrieved=retrieved)

    def ask(self, q: str) -> Answer:
        route, args = self.route(q)
        if route == "policy_search":
            return self._answer_policy(q)
        fn = TOOLS[route][0]
        try:
            result = fn(self.db, **args)
        except TypeError:
            return Answer(q, "refused", REFUSAL)
        sources = re.findall(r"\[([\w#-]+)\]", result["summary"])
        return Answer(q, route, result["summary"], sources=sources, tool_args=args)

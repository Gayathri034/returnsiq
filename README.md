# ReturnsIQ: An Evaluated Agentic RAG Assistant for E-commerce Returns

ReturnsIQ answers two kinds of questions a returns team gets every day:

- **Policy questions** ("Can I return a gift card?") are answered by **retrieval-augmented generation (RAG)** over a set of returns policy documents, with a citation for every answer.
- **Data questions** ("Would customer 12346 pay the restocking fee?") are routed by an **agent** to **predefined, read-only SQL tools** that query 1M+ real transactions from my [E-commerce Sales and Returns Analytics](https://github.com/Gayathri034/ecommerce-returns-analytics) project.

It ships with an **evaluation suite** that measures retrieval quality, groundedness, routing accuracy, and refusals, and it runs **free and fully locally**, with no API keys.

> The policy documents in `docs/` describe **Northwind Home Goods, a fictional retailer** created for this project. The restocking-fee rule (10% fee at a 20% return-rate threshold) comes from the policy simulation in the analytics project.

---

## How it works

```
                        question
                           │
                     ┌─────▼─────┐
                     │  Router   │  rules for clear data patterns,
                     │  (agent)  │  local LLM for everything else
                     └─────┬─────┘
            ┌──────────────┴───────────────┐
     policy question                  data question
            │                               │
  ┌─────────▼─────────┐          ┌──────────▼──────────┐
  │ Retriever         │          │ SQL tools (read-only)│
  │ TF-IDF / dense /  │          │ overall_return_rate  │
  │ hybrid, top 3     │          │ monthly_return_rate  │
  └─────────┬─────────┘          │ top_returned_products│
            │                    │ country_return_rate  │
  ┌─────────▼─────────┐          │ check_restocking_fee │
  │ Guardrails        │          └──────────┬──────────┘
  │ score + coverage  │                     │
  │ → refuse if weak  │          numbers come straight
  └─────────┬─────────┘          from SQL, never the LLM
            │                               │
  answer with citations ──────────── answer with citations
```

## Design decisions (responsible AI)

| Decision | Why |
| --- | --- |
| The agent can only call **5 predefined, parameterized, read-only SQL tools**; it never writes SQL | Prevents SQL injection and runaway queries |
| **Every number in a data answer comes from SQL**, not from the language model | The model cannot invent figures |
| **Two refusal gates**: low retrieval score, or low *query coverage* (most of the question's key words never appear in the policies) | Out-of-scope questions get "I couldn't find this" instead of a made-up answer |
| **Every answer cites its source** section, e.g. `[03_damaged_defective#reporting-damage]` | Answers can be checked |
| `check_restocking_fee` **combines live data with the written policy** | One answer applies the rule (20% rate, at least 3 orders) to a real customer's history |
| **LLM output is verified, not trusted**: a refusal after retrieval passed, or an answer the sources don't support (groundedness < 50%), is replaced by the cited policy section | Small local models over-refuse and hallucinate; users always get a sourced answer |
| **LLM tool calls are checked**: a customer ID or country must actually appear in the question | Stops the model from inventing tool arguments |
| Runs on **Ollama** (local model) or with **no model** (extractive mode) | Free, private, reproducible |

---

## Run it

```bash
pip install -r requirements.txt

# Uses ../returns-analytics/returns.db from the analytics project by default (or pass --db)
python ask.py "How quickly must I report a damaged item?"
python ask.py "Would customer 12346 pay the restocking fee?"
python ask.py "What is the return rate in Germany?"
python ask.py                       # interactive mode

python evaluate.py                  # writes eval/report.md
```

### Optional: use a free local LLM
1. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2`.
2. Add `--llm ollama` to any command, e.g. `python evaluate.py --llm ollama`.

### Optional: compare retrievers
```bash
pip install sentence-transformers
python evaluate.py --compare-retrievers     # TF-IDF vs dense vs hybrid
```

---

## Evaluation

`eval/questions.json` holds **30 labeled questions**: 21 policy questions (each with the correct source section and the expected fact), 5 data questions (each with the correct tool), and 4 out-of-scope questions that should be refused.

**Results (extractive mode, TF-IDF retriever):**

| Metric | Score |
| --- | --- |
| Retrieval hit@3 (correct section in top 3) | 100% |
| Retrieval hit@1 | 76% |
| Retrieval MRR | 0.87 |
| Tool routing accuracy | 100% |
| Out-of-scope refusal accuracy | 100% |
| False refusals | 0% |
| Citation rate | 100% |
| Answer accuracy (expected fact present) | 76% |

### What I changed after the first evaluation

| Metric | Baseline | Final | Change made |
| --- | --- | --- | --- |
| Out-of-scope refusal | 50% | **100%** | Added the query-coverage gate. "Warranty period for furniture" shared one word with the policies, so a score cutoff alone couldn't catch it |
| Retrieval hit@3 | 95% | **100%** | Added light stemming so "reporting damage" matches "report... damaged" |
| Retrieval MRR | 0.84 | **0.87** | Ablation over stemming, n-grams, and heading boosts; kept the simplest best setting |

I deliberately did **not** keep tuning against these 30 questions: differences of one question on a set this small are noise, and tuning further would overfit the test set.

### Testing a small local LLM (Llama 3.2 1B, CPU only)

The first run with `--llm ollama --model llama3.2:1b` made answers **worse than using no model at all**, even though the correct section reached the model every time (100% hit@3):

| Metric | No LLM | Llama 3.2 1B, unverified |
| --- | --- | --- |
| Answer accuracy | 76% | 14% |
| False refusals | 0% | 62% |
| Citation rate | 100% | 24% |
| Groundedness | 100% | 19% |

The model refused answerable questions, skipped citations, and invented a customer ID to call a data tool. I added **output verification** (fallback on refusal or low groundedness, automatic citation of the supporting source) and **tool-argument checks**. The lesson: with small models, retrieval quality is not enough; the system has to check the model's output.

### Known limitations
- **Keyword retrieval misses meaning.** "How long do refunds take?" ranks the shipping-refund section above the refund-timeline section, and "blender" doesn't match "small appliances." Dense and hybrid retrieval (`--compare-retrievers`) target exactly this.
- **Extractive mode quotes a whole section**, so it can include extra detail. LLM mode writes a focused answer from the top 3 sections, where the right one appears 100% of the time.
- **Groundedness is 100% in extractive mode by construction** (answers are quoted). It becomes a meaningful check in LLM mode.

## Project structure
```
docs/                 fictional returns policy documents (6 files, 24 sections)
returnsiq/
  ingest.py           loads docs, one chunk per section
  retriever.py        TF-IDF (with stemming), dense, and hybrid retrieval; coverage check
  tools.py            read-only, parameterized SQL tools
  llm.py              Ollama provider (or none)
  agent.py            routing, guardrails, answering with citations
ask.py                command-line interface
evaluate.py           evaluation suite → eval/report.md
eval/questions.json   30 labeled questions
```

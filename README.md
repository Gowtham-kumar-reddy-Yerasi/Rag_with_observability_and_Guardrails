# 🤖 Smart AI Agent: RAG-Only Q&A with Guardrails, RAGAS & Observability

![Smart AI Agent Thumbnail](assets/agentBot.png)

🔗 **[Download Project PDF](assets/agentBot.pdf)**

---
This repository contains a **RAG-only** AI agent. It answers strictly from the documents you upload to the knowledge base (Pinecone). If the knowledge base does not contain the answer, the agent says so instead of searching the web or guessing. Every message is screened by input safety guardrails, every run is traced in Phoenix, and answer quality is measured with RAGAS against a golden set.

---

## ✨ Key Features

- **Strictly grounded answers**: no web search, no general-knowledge fallback. Unsupported questions get *"This information does not exist in the knowledge base."*
- **Two-stage grounding check**: an LLM judge decides whether the retrieved chunks are sufficient; the answer model must also reply `NOT_FOUND` rather than use outside knowledge. Empty retrieval short-circuits without any LLM call.
- **Input guardrails (2 layers)**: regex rules for prompt injection, jailbreaks, harmful requests, abuse, oversize input and PII redaction (email, phone, card with Luhn check, Aadhaar, PAN, SSN), plus an optional LLM safety classifier for paraphrased attacks. Blocked input never reaches retrieval or the answer model. PII is replaced by placeholders before it is stored in chat memory or sent to an LLM.
- **Observability with Arize Phoenix (open source)**: every request is a trace with spans for each graph node, retriever call and LLM call; sessions are grouped by chat session id. `GET /stats` exposes live counters.
- **RAGAS evaluation**: faithfulness, context recall, context precision, answer relevancy, plus not-found accuracy and guardrail block/false-positive rates, over a 24-question golden set.
- **Transparent trace in the UI**: guardrail verdict, retrieval sufficiency and final outcome per message.
- **Dynamic knowledge ingestion**: upload PDFs from the UI; they are chunked, embedded and indexed (the Pinecone index is created automatically if missing).

---

## 🚀 Architecture

```
guardrail ──blocked──────────────▶ END   (refusal message)
    │
  router ──greeting/thanks───────▶ END   (canned reply, no LLM)
    │
rag_lookup ──insufficient────────▶ not_found ─▶ END
    │
  answer ──"NOT_FOUND"───────────▶ not_found ─▶ END
    │
   END   (grounded answer)
```

## 📦 Project structure

```
frontend/                 Streamlit UI (app.py, ui_components.py, backend_api.py, ...)
backend/
├── main.py               FastAPI: /chat/, /upload-document/, /stats, /health
├── agent.py              LangGraph agent (guardrail -> router -> rag -> answer / not_found)
├── guardrails.py         Input guardrails (rules + optional LLM classifier)
├── observability.py      Phoenix / OpenTelemetry setup + in-process counters
├── vectorstore.py        Pinecone index + chunking
├── evaluate_ragas.py     RAGAS + behaviour + guardrail evaluation
├── golden_set.json       24 questions (20 answerable, 4 out-of-scope) with references
├── safety_set.json       52 labelled prompts (attacks, PII, benign look-alikes)
└── tests/                79 offline tests (no API keys needed)
data/automotiveguide.pdf  Sample document the golden set was written for
```

---

## 🛠️ Setup

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), a Groq API key and a Pinecone API key.

```bash
uv sync                     # installs everything from pyproject.toml / uv.lock
cp .env.example .env        # then fill in GROQ_API_KEY and PINECONE_API_KEY
```

## 🏃 Running

Open three terminals from the project root.

```bash
# 1. Observability UI + trace collector  ->  http://localhost:6006
uv run phoenix serve

# 2. Backend
cd backend && uv run uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Frontend
cd frontend && uv run streamlit run app.py
```

Then upload `data/automotiveguide.pdf` in the UI and ask questions. Open Phoenix, project **rag-agent**, to inspect traces. Phoenix is optional: if it is not running, the app works normally and simply drops traces (set `PHOENIX_ENABLED=false` to silence the warnings).

## 📊 Evaluation

```bash
cd backend
uv run evaluate_ragas.py --limit 6    # quick smoke run
uv run evaluate_ragas.py              # full run (24 questions; uses Groq as the RAGAS judge)
uv run evaluate_ragas.py --skip-ragas # only not-found + guardrail metrics, no judge calls
uv run evaluate_ragas.py --strict     # exit code 1 if a metric misses its target (CI)
```

Outputs `ragas_results.csv` (per question), `eval_summary.json` and `eval_report.md`.
Index the sample PDF first. Targets live in `TARGETS` at the top of `evaluate_ragas.py`.

| Metric | What it measures | Scored on |
|---|---|---|
| `context_recall` | Did retrieval surface what the reference answer needs? | answerable questions |
| `context_precision` | Are the useful chunks ranked first? | answerable questions |
| `faithfulness` | Is every claim in the answer supported by the retrieved chunks? | questions the agent answered |
| `answer_relevancy` | Does the answer address the question? | questions the agent answered |
| `answer_rate` | Answerable questions that get an answer (1 - false-refusal rate) | answerable questions |
| `correct_not_found_rate` | Out-of-scope questions that get "does not exist" | out-of-scope questions |
| `attack_block_rate` / `false_positive_rate` | Guardrail rules on `safety_set.json` | safety set |

## ✅ Tests

```bash
cd backend && uv run --group dev pytest -q      # 79 tests, fully offline
```

## 🧪 API

`POST /chat/`
```json
{ "session_id": "test-session-001", "query": "How often should brake fluid be replaced?" }
```
```json
{ "response": "...", "outcome": "answered", "trace_events": [ { "step": 1, "node_name": "guardrail", "...": "..." } ] }
```
`outcome` is one of `answered`, `not_found`, `blocked`, `smalltalk`. (`enable_web_search` is still accepted from older clients and ignored.)

`POST /upload-document/` (multipart, field `file`, PDF only) returns the number of **chunks** indexed.
`GET /stats` returns request / outcome / guardrail counters. `GET /health` returns `{"status": "ok"}`.

---

## ⚠️ Notes

- Raw user text is visible in the *root* span of a trace (the guardrail node redacts PII before anything is stored or sent to an LLM, but tracing starts before it). Phoenix is self-hosted, so this data stays on your machine; keep that in mind before exposing it.
- Chat memory is in-process (`MemorySaver`): it resets when the backend restarts. Retrieval uses only the latest message, so follow-ups like "and what about its interval?" are not resolved from history.

> Built with LangGraph, LangChain, Groq, Pinecone, RAGAS, Arize Phoenix and Streamlit

# backend/evaluate_ragas.py  -- run from the backend folder:  uv run evaluate_ragas.py
"""Evaluate the RAG agent end to end.

Four groups of metrics:

 1. RAGAS retrieval  (answerable questions)      : context_recall, context_precision
 2. RAGAS generation (questions the agent answered): faithfulness, answer_relevancy
 3. Not-found behaviour                           : answer_rate on answerable questions,
                                                    correct "doesn't exist" rate on out-of-scope ones
 4. Input guardrails (rules layer, deterministic) : attack block rate, false-positive rate, PII redaction

The agent under test is the real LangGraph agent, so what is scored is what users get: the
`retrieved_contexts` are the chunks the agent actually used, not a second retrieval.

Usage:
    uv run evaluate_ragas.py                 # full run
    uv run evaluate_ragas.py --limit 6       # quick smoke run
    uv run evaluate_ragas.py --skip-ragas    # behaviour + safety only (no judge LLM calls)
    uv run evaluate_ragas.py --strict        # exit 1 if any metric is below its target
"""
import argparse
import json
import math
import time
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

HERE = Path(__file__).parent

# Pass/fail targets used in the report (tune to your needs).
TARGETS = {
    "faithfulness": 0.85,
    "context_recall": 0.80,
    "context_precision": 0.60,
    "answer_relevancy": 0.75,
    "answer_rate": 0.85,            # answerable questions that get an answer
    "correct_not_found_rate": 1.00,  # out-of-scope questions that get "does not exist"
    "attack_block_rate": 0.95,
    "false_positive_rate": 0.00,     # lower is better; target is a ceiling
}
LOWER_IS_BETTER = {"false_positive_rate"}


# --------------------------------------------------------------------------
# 1. Run the agent over the golden set
# --------------------------------------------------------------------------
def run_pipeline(agent, golden, sleep_s: float = 1.0, run_tag: str | None = None):
    from langchain_core.messages import AIMessage, HumanMessage

    run_tag = run_tag or time.strftime("eval-%Y%m%d-%H%M%S")
    records = []
    for i, item in enumerate(golden, 1):
        print(f"[{i}/{len(golden)}] ({item['id']}) {item['question']}")
        cfg = {
            "configurable": {"thread_id": str(uuid.uuid4())},     # fresh memory per question
            "metadata": {"session_id": run_tag, "golden_id": item["id"]},   # shows up in Phoenix
            "run_name": f"eval_{item['id']}",
        }
        agent.invoke({"messages": [HumanMessage(content=item["question"])]}, config=cfg)
        st = agent.get_state(cfg).values
        answer = next((m.content for m in reversed(st["messages"]) if isinstance(m, AIMessage)), "")
        records.append({
            **item,
            "outcome": st.get("outcome"),
            "response": answer,
            "retrieved_contexts": st.get("contexts", []) or [],
            "guardrail_category": (st.get("guardrail") or {}).get("category"),
        })
        time.sleep(sleep_s)   # be gentle with Groq rate limits
    return records


# --------------------------------------------------------------------------
# 2. Behaviour metrics (the "info does not exist" path)
# --------------------------------------------------------------------------
def behaviour_metrics(records):
    ans = [r for r in records if r["answerable"]]
    oos = [r for r in records if not r["answerable"]]
    answered = [r for r in ans if r["outcome"] == "answered"]
    out = {
        "n_answerable": len(ans),
        "n_out_of_scope": len(oos),
        "answer_rate": len(answered) / len(ans) if ans else None,
        "false_refusal_rate": 1 - len(answered) / len(ans) if ans else None,
        "correct_not_found_rate": (sum(r["outcome"] == "not_found" for r in oos) / len(oos)) if oos else None,
        "hallucination_on_out_of_scope": (sum(r["outcome"] == "answered" for r in oos) / len(oos)) if oos else None,
        "blocked_by_guardrail_unexpectedly": sum(r["outcome"] == "blocked" for r in records),
    }
    return out


# --------------------------------------------------------------------------
# 3. RAGAS
# --------------------------------------------------------------------------
def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return (sum(vals) / len(vals)) if vals else None, len(vals)


def run_ragas(records, judge_llm, judge_embeddings):
    """Returns (per_row_dicts, summary_dict). judge_llm / judge_embeddings are LangChain objects."""
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(judge_llm)
    emb = LangchainEmbeddingsWrapper(judge_embeddings)
    cfg = RunConfig(max_workers=2, timeout=180)   # low parallelism avoids Groq 429s

    def rows_for(recs):
        return [{
            "user_input": r["question"],
            "retrieved_contexts": r["retrieved_contexts"],
            "response": r["response"],
            "reference": r["reference"],
        } for r in recs]

    answerable = [r for r in records if r["answerable"]]
    answered = [r for r in answerable if r["outcome"] == "answered"]
    per_row: dict[str, dict] = {r["id"]: {} for r in records}
    summary: dict = {}

    def score(recs, metrics, label):
        if not recs:
            print(f"(no rows for {label}; skipped)")
            return
        print(f"\nRAGAS {label}: {len(recs)} rows")
        res = evaluate(EvaluationDataset.from_list(rows_for(recs)), metrics=metrics, llm=llm,
                       embeddings=emb, run_config=cfg, show_progress=True)
        df = res.to_pandas()
        for col in [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference",
                                                        "reference_contexts", "retrieved_context_ids")]:
            for rec, v in zip(recs, df[col].tolist()):
                per_row[rec["id"]][col] = None if (isinstance(v, float) and math.isnan(v)) else v
            m, n = _mean(df[col].tolist())
            summary[col] = {"mean": m, "n": n, "n_rows": len(recs)}

    # Retrieval quality: judged on every answerable question, whether or not the agent answered.
    score(answerable, [LLMContextRecall(), LLMContextPrecisionWithReference()], "retrieval (answerable questions)")
    # Generation quality: only where the agent produced an answer (a refusal has nothing to be faithful to).
    score(answered, [Faithfulness(), ResponseRelevancy(strictness=1)],       # strictness=1: Groq rejects n>1
          "generation (answered questions)")

    # Normalise ragas' metric column names to stable keys
    alias = {"llm_context_precision_with_reference": "context_precision"}
    summary = {alias.get(k, k): v for k, v in summary.items()}
    for d in per_row.values():
        for old, new in alias.items():
            if old in d:
                d[new] = d.pop(old)
    return per_row, summary


# --------------------------------------------------------------------------
# 4. Guardrail evaluation (deterministic rules layer)
# --------------------------------------------------------------------------
def safety_metrics(safety_cases=None):
    from guardrails import check_rules

    cases = safety_cases or json.loads((HERE / "safety_set.json").read_text(encoding="utf-8"))
    attacks = [c for c in cases if c["expect"] == "block"]
    benign = [c for c in cases if c["expect"] == "allow"]
    pii = [c for c in cases if c["expect"] == "redact"]

    blocked = [c for c in attacks if not check_rules(c["text"]).allowed]
    wrongly_blocked = [c for c in benign if not check_rules(c["text"]).allowed]
    redacted = [c for c in pii if check_rules(c["text"]).category == "pii_redacted"
                and c["category"] in check_rules(c["text"]).pii_found]

    by_cat: dict[str, list[int]] = {}
    for c in attacks:
        ok = check_rules(c["text"]).category == c["category"]
        by_cat.setdefault(c["category"], []).append(int(ok))
    return {
        "n_attacks": len(attacks), "n_benign": len(benign), "n_pii": len(pii),
        "attack_block_rate": len(blocked) / len(attacks) if attacks else None,
        "false_positive_rate": len(wrongly_blocked) / len(benign) if benign else None,
        "pii_redaction_rate": len(redacted) / len(pii) if pii else None,
        "per_category_accuracy": {k: sum(v) / len(v) for k, v in by_cat.items()},
        "false_positives": [c["text"] for c in wrongly_blocked],
        "missed_attacks": [c["text"] for c in attacks if c not in blocked],
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def build_report(summary: dict) -> tuple[str, bool]:
    ragas = summary.get("ragas", {})
    beh, safe = summary["behaviour"], summary["safety"]
    flat = {
        "faithfulness": ragas.get("faithfulness", {}).get("mean"),
        "context_recall": ragas.get("context_recall", {}).get("mean"),
        "context_precision": ragas.get("context_precision", {}).get("mean"),
        "answer_relevancy": ragas.get("answer_relevancy", {}).get("mean"),
        "answer_rate": beh["answer_rate"],
        "correct_not_found_rate": beh["correct_not_found_rate"],
        "attack_block_rate": safe["attack_block_rate"],
        "false_positive_rate": safe["false_positive_rate"],
    }
    lines = ["| Metric | Score | Target | Status |", "|---|---|---|---|"]
    all_ok = True
    for k, v in flat.items():
        t = TARGETS[k]
        if v is None:
            status = "skipped"
        else:
            ok = v <= t if k in LOWER_IS_BETTER else v >= t
            status = "PASS" if ok else "FAIL"
            all_ok &= ok
        lines.append(f"| {k} | {_fmt(v)} | {'<=' if k in LOWER_IS_BETTER else '>='} {t:.2f} | {status} |")
    lines += [
        "",
        f"Answerable questions: {beh['n_answerable']}  |  out-of-scope: {beh['n_out_of_scope']}  |  "
        f"hallucinated on out-of-scope: {_fmt(beh['hallucination_on_out_of_scope'])}  |  "
        f"PII redaction rate: {_fmt(safe['pii_redaction_rate'])}",
    ]
    return "\n".join(lines), all_ok


def main(argv=None, agent=None, judge_llm=None, judge_embeddings=None, out_dir: Path | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run the first N golden questions")
    ap.add_argument("--skip-ragas", action="store_true", help="skip judge-LLM metrics")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any metric misses its target")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args(argv)
    out_dir = out_dir or HERE

    golden = json.loads((HERE / "golden_set.json").read_text(encoding="utf-8"))
    if args.limit:
        # keep the mix: take from answerable and out-of-scope proportionally
        ans = [g for g in golden if g["answerable"]][: max(1, args.limit - max(1, args.limit // 4))]
        oos = [g for g in golden if not g["answerable"]][: max(1, args.limit // 4)]
        golden = ans + oos

    if agent is None:
        from agent import rag_agent as agent   # real agent: Pinecone + Groq
    records = run_pipeline(agent, golden, sleep_s=args.sleep)

    summary = {"behaviour": behaviour_metrics(records), "safety": safety_metrics(), "ragas": {}}
    per_row = {r["id"]: {} for r in records}

    if not args.skip_ragas:
        if judge_llm is None:
            from langchain_groq import ChatGroq
            from config import GROQ_API_KEY, GROQ_MODEL
            from vectorstore import embeddings
            judge_llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
            judge_embeddings = embeddings
        per_row, summary["ragas"] = run_ragas(records, judge_llm, judge_embeddings)

    # ---- save artefacts ----
    import pandas as pd
    rows = []
    for r in records:
        rows.append({
            "id": r["id"], "answerable": r["answerable"], "question_type": r["question_type"],
            "question": r["question"], "outcome": r["outcome"], "response": r["response"],
            "n_contexts": len(r["retrieved_contexts"]), **per_row.get(r["id"], {}),
        })
    pd.DataFrame(rows).to_csv(out_dir / "ragas_results.csv", index=False)
    report, all_ok = build_report(summary)
    summary["report_markdown"] = report
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "eval_report.md").write_text("# Evaluation report\n\n" + report + "\n", encoding="utf-8")

    print("\n" + "=" * 70 + "\n" + report + "\n" + "=" * 70)
    print(f"Saved: {out_dir / 'ragas_results.csv'}, eval_summary.json, eval_report.md")
    try:
        from observability import flush
        flush()
    except Exception:
        pass
    if args.strict and not all_ok:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()

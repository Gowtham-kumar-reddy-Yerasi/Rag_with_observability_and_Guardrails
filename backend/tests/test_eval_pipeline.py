"""End-to-end test of evaluate_ragas.py with a fake agent + fake judge (fully offline)."""
import json
import math
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import evaluate_ragas as ev
from agent import RagJudge, build_agent
from tests.conftest import FakeAnswerLLM
from tests.fake_judge import FakeRagasJudge

GOLD = json.loads((Path(ev.HERE) / "golden_set.json").read_text(encoding="utf-8"))
BY_Q = {g["question"]: g for g in GOLD}

HALLUCINATE = "q09"      # agent answers with a fact that is NOT in its context
FALSE_REFUSE = "q16"     # agent wrongly says "does not exist"
LEAK = "oos02"           # agent answers an out-of-scope question (hallucination)


class GoldRetriever:
    """Returns the golden reference_contexts for answerable questions, an unrelated chunk otherwise."""
    def invoke(self, query):
        g = BY_Q[query]
        texts = g["reference_contexts"] or ["Tyres should be rotated regularly to ensure even wear."]
        return [Document(page_content=t) for t in texts]


class GoldJudge:
    def invoke(self, messages):
        q = messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1].content
        q = q.split("Question: ", 1)[1].split("\n\nPassages:", 1)[0]
        g = BY_Q[q]
        return RagJudge(sufficient=(g["answerable"] and g["id"] != FALSE_REFUSE) or g["id"] == LEAK)


class GoldAnswer:
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        q = messages[-1].content.split("QUESTION: ", 1)[1]
        g = BY_Q[q]
        if g["id"] == HALLUCINATE:
            return AIMessage(content="Car batteries last about 10 to 12 years in all conditions.")
        if g["id"] == LEAK:
            return AIMessage(content="Use 5W-30 viscosity oil for every engine.")
        return AIMessage(content=g["reference"])


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    agent = build_agent(retriever=GoldRetriever(), judge_llm=GoldJudge(), answer_llm=GoldAnswer(), guard_llm=None)
    s = ev.main(["--sleep", "0"], agent=agent, judge_llm=FakeRagasJudge(),
                judge_embeddings=DeterministicFakeEmbedding(size=16), out_dir=out)
    s["_out"] = out
    return s


def test_behaviour_metrics_match_the_planted_failures(summary):
    b = summary["behaviour"]
    assert b["n_answerable"] == 20 and b["n_out_of_scope"] == 4
    assert b["answer_rate"] == pytest.approx(19 / 20)                    # one false refusal (q16)
    assert b["correct_not_found_rate"] == pytest.approx(3 / 4)           # one leak (oos02)
    assert b["hallucination_on_out_of_scope"] == pytest.approx(1 / 4)
    assert b["blocked_by_guardrail_unexpectedly"] == 0


def test_ragas_metrics_are_present_and_in_range(summary):
    r = summary["ragas"]
    assert set(r) >= {"context_recall", "context_precision", "faithfulness", "answer_relevancy"}
    for k, v in r.items():
        assert v["mean"] is not None and 0 <= v["mean"] <= 1, (k, v)
    assert r["context_recall"]["n_rows"] == 20          # retrieval scored on all answerable questions
    assert r["faithfulness"]["n_rows"] == 19            # generation scored only where the agent answered (q16 refused)


def test_hallucinated_answer_gets_lower_faithfulness_than_grounded_ones(summary):
    import pandas as pd
    df = pd.read_csv(summary["_out"] / "ragas_results.csv").set_index("id")
    assert df.loc[HALLUCINATE, "faithfulness"] < 0.5
    grounded = df.loc[["q02", "q03", "q07", "q15"], "faithfulness"]
    assert (grounded >= 0.99).all()


def test_false_refusal_row_has_no_generation_score_but_has_retrieval_score(summary):
    import pandas as pd
    df = pd.read_csv(summary["_out"] / "ragas_results.csv").set_index("id")
    assert df.loc[FALSE_REFUSE, "outcome"] == "not_found"
    assert math.isnan(df.loc[FALSE_REFUSE, "faithfulness"])
    assert not math.isnan(df.loc[FALSE_REFUSE, "context_recall"])


def test_safety_metrics(summary):
    s = summary["safety"]
    assert s["attack_block_rate"] == 1.0 and s["false_positive_rate"] == 0.0 and s["pii_redaction_rate"] == 1.0
    assert s["missed_attacks"] == [] and s["false_positives"] == []


def test_report_files_written_and_report_flags_failures(summary):
    out = summary["_out"]
    for f in ("ragas_results.csv", "eval_summary.json", "eval_report.md"):
        assert (out / f).exists()
    rep = summary["report_markdown"]
    assert "correct_not_found_rate" in rep and "FAIL" in rep        # the leaked OOS question must be flagged

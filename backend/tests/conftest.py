"""Offline test fixtures: no Pinecone, no Groq, no HuggingFace download, no Phoenix."""
import os
import sys
import types

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PHOENIX_ENABLED", "false")
os.environ.setdefault("GUARDRAIL_LLM_ENABLED", "false")

# Stub the heavy `vectorstore` module (it loads an embedding model at import time).
_vs = types.ModuleType("vectorstore")
_vs.get_retriever = lambda k=5: (_ for _ in ()).throw(RuntimeError("real retriever must not be used in tests"))
_vs.add_document_to_vectorstore = lambda text: max(1, len(text) // 1000)
_vs.embeddings = None
_vs.INDEX_NAME = "test"
sys.modules.setdefault("vectorstore", _vs)

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agent import GuardVerdict, RagJudge, build_agent


class FakeRetriever:
    """Returns docs whose text shares a keyword with the query (case-insensitive)."""
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def invoke(self, query):
        self.queries.append(query)
        words = {w.strip("?.,!").lower() for w in query.split() if len(w) > 3}
        return [Document(page_content=d) for d in self.docs if words & {w.strip(".,").lower() for w in d.split()}]


class FakeJudge:
    """sufficient == retrieved passages mention a word from QUESTION that is a 'topic' word."""
    def __init__(self, answer=True, raises=False):
        self.answer, self.raises, self.calls = answer, raises, 0

    def invoke(self, messages):
        self.calls += 1
        if self.raises:
            raise RuntimeError("judge down")
        return RagJudge(sufficient=self.answer)


class FakeAnswerLLM:
    def __init__(self, text="Brake fluid is replaced every 2 years."):
        self.text, self.calls, self.prompts = text, 0, []

    def invoke(self, messages):
        self.calls += 1
        self.prompts.append("\n".join(m.content for m in messages))
        return AIMessage(content=self.text)


class FakeGuard:
    def __init__(self, verdict=None, raises=False):
        self.verdict, self.raises, self.calls = verdict or GuardVerdict(safe=True), raises, 0

    def invoke(self, messages):
        self.calls += 1
        if self.raises:
            raise RuntimeError("guard down")
        return self.verdict


KB = [
    "Brake fluid Every 2 years. Replace it regardless of mileage.",
    "Timing belt 100,000 to 160,000 km, depending on the manufacturer.",
]


@pytest.fixture
def make_agent():
    def _make(judge=None, answer=None, guard=None, docs=None):
        r = FakeRetriever(KB if docs is None else docs)
        j, a = judge or FakeJudge(True), answer or FakeAnswerLLM()
        agent = build_agent(retriever=r, judge_llm=j, answer_llm=a, guard_llm=guard)
        return agent, r, j, a
    return _make


def run(agent, text, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}}
    agent.invoke({"messages": [("user", text)]}, config=cfg)
    st = agent.get_state(cfg).values
    last_ai = next(m.content for m in reversed(st["messages"]) if isinstance(m, AIMessage))
    return st, last_ai

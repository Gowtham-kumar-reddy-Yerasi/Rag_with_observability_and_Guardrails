"""Offline stand-in for the RAGAS judge LLM.

Parses the real RAGAS prompts and answers with simple lexical-overlap heuristics, so the
evaluation plumbing (dataset building, per-row mapping, aggregation, report) can be tested
without network access. It validates the PIPELINE, not the quality of real judge scores.
"""
import json
import re
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_STOP = {"the", "and", "for", "are", "that", "with", "from", "this", "does", "not", "has", "have", "can", "its", "into", "than"}


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9\.]+", s.lower()) if len(w) > 2 and w not in _STOP}


def _overlap(statement: str, context: str) -> float:
    st = _tokens(statement)
    return len(st & _tokens(context)) / len(st) if st else 0.0


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _parse_input(prompt: str) -> dict:
    tail = prompt.rsplit("Now perform the same with the following input\ninput:", 1)[1].strip()
    obj, _ = json.JSONDecoder().raw_decode(tail)
    return obj


class FakeRagasJudge(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-ragas-judge"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        self.calls += 1
        inp = _parse_input(prompt)
        if "Break down each sentence into one or more fully understandable statements" in prompt:
            out = {"statements": _sentences(inp["answer"])}
        elif "judge the faithfulness of a series of statements" in prompt:
            out = {"statements": [
                {"statement": s, "reason": "lexical overlap", "verdict": int(_overlap(s, inp["context"]) >= 0.7)}
                for s in inp["statements"]]}
        elif "classify if the sentence can be attributed to the given context" in prompt:
            out = {"classifications": [
                {"statement": s, "reason": "lexical overlap", "attributed": int(_overlap(s, inp["context"]) >= 0.5)}
                for s in _sentences(inp["answer"])]}
        elif "verify if the context was useful in arriving at the given answer" in prompt:
            out = {"reason": "lexical overlap", "verdict": int(_overlap(inp["answer"], inp["context"]) >= 0.3)}
        elif "Generate a question for the given answer" in prompt:
            resp = inp["response"].lower()
            out = {"question": "What is stated?", "noncommittal": int("does not exist" in resp)}
        else:
            raise AssertionError("Unrecognised RAGAS prompt:\n" + prompt[:300])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(out)))])

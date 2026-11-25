"""Observability with Arize Phoenix (open source, self-hosted) + OpenInference.

* `setup_observability()` registers an OpenTelemetry tracer that ships to Phoenix and
  auto-instruments every LangChain / LangGraph call (each graph node, each LLM call,
  each retriever call becomes a span with inputs, outputs, latency and token counts).
* It never raises: if Phoenix isn't running, the app keeps working and traces are
  simply dropped (the OTel exporter logs a warning).
* `stats` is a tiny in-process counter exposed at GET /stats for a quick health view
  (answered / not_found / blocked ...), independent of Phoenix.

Run Phoenix:   phoenix serve        (UI + collector on http://localhost:6006)
"""
from __future__ import annotations

import threading
from collections import Counter

from config import PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_ENABLED, PHOENIX_PROJECT_NAME

_initialised = False
_tracer_provider = None


def setup_observability():
    """Idempotent. Returns the tracer provider, or None when disabled/unavailable."""
    global _initialised, _tracer_provider
    if _initialised:
        return _tracer_provider
    _initialised = True
    if not PHOENIX_ENABLED:
        print("[observability] disabled (PHOENIX_ENABLED=false)")
        return None
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        _tracer_provider = register(
            project_name=PHOENIX_PROJECT_NAME,
            endpoint=f"{PHOENIX_COLLECTOR_ENDPOINT.rstrip('/')}/v1/traces",
            batch=True,
            auto_instrument=False,
            set_global_tracer_provider=True,
            verbose=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
        print(f"[observability] Phoenix tracing -> {PHOENIX_COLLECTOR_ENDPOINT} (project '{PHOENIX_PROJECT_NAME}')")
    except Exception as exc:  # never break the app because of tracing
        print(f"[observability] tracing NOT enabled: {exc}")
        _tracer_provider = None
    return _tracer_provider


def flush(timeout_ms: int = 5000):
    """Force-export pending spans (used by scripts that exit right after running)."""
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_ms)
        except Exception:
            pass


class _Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self._c: Counter = Counter()

    def inc(self, key: str, n: int = 1):
        with self._lock:
            self._c[key] += n

    def snapshot(self) -> dict:
        with self._lock:
            c = dict(self._c)
        total = c.get("requests", 0)
        return {
            "counters": c,
            "not_found_rate": round(c.get("outcome.not_found", 0) / total, 3) if total else None,
            "blocked_rate": round(c.get("outcome.blocked", 0) / total, 3) if total else None,
        }


stats = _Stats()

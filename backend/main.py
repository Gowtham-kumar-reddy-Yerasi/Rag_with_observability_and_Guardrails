# rag_agent_app/backend/main.py

import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from agent import rag_agent
from observability import flush, stats
from vectorstore import add_document_to_vectorstore

app = FastAPI(
    title="LangGraph RAG Agent API",
    description="RAG-only LangGraph agent with input guardrails, Pinecone, Groq and Phoenix tracing.",
    version="2.0.0",
)


# --- Pydantic Models for API ---
class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type: str


class QueryRequest(BaseModel):
    session_id: str
    query: str
    # Accepted for backward compatibility with older clients; web search no longer exists.
    enable_web_search: Optional[bool] = None


class AgentResponse(BaseModel):
    response: str
    outcome: str = "answered"        # answered | not_found | blocked | smalltalk
    trace_events: List[TraceEvent] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int


# --- Document Upload Endpoint ---
@app.post("/upload-document/", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, and adds it to the RAG knowledge base."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        temp_file_path = tmp_file.name

    try:
        documents = PyPDFLoader(temp_file_path).load()
        text = "\n\n".join(d.page_content for d in documents).strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable text found in this PDF (is it a scanned image?).",
            )
        chunks = add_document_to_vectorstore(text)
        return DocumentUploadResponse(
            message=f"PDF '{file.filename}' successfully uploaded and indexed.",
            filename=file.filename,
            processed_chunks=chunks,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF document: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process PDF: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Trace helpers ---------------------------------------------------------
def _describe(node: str, upd: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    """Turn one LangGraph node update into (description, details, event_type) for the UI."""
    if node == "guardrail":
        g = upd.get("guardrail", {})
        if not g.get("allowed", True):
            return (f"Input BLOCKED by guardrail ({g.get('category')}).",
                    {"category": g.get("category"), "layer": g.get("layer"), "reason": g.get("reason")},
                    "guardrail_block")
        if g.get("category") == "pii_redacted":
            return (f"Input allowed; PII redacted ({', '.join(g.get('pii_found', []))}).",
                    {"pii_found": g.get("pii_found")}, "guardrail_pass")
        return "Input passed safety checks.", {"category": g.get("category")}, "guardrail_pass"
    if node == "router":
        return (f"Router decided: '{upd.get('route')}'", {"decision": upd.get("route")}, "router_decision")
    if node == "rag_lookup":
        ctx = upd.get("contexts", [])
        ok = upd.get("sufficient", False)
        return (
            f"Retrieved {len(ctx)} chunk(s); judged {'SUFFICIENT' if ok else 'NOT sufficient'}.",
            {"chunks": len(ctx), "sufficiency_verdict": "Sufficient" if ok else "Not Sufficient",
             "retrieved_content_summary": (ctx[0][:200] + "...") if ctx else ""},
            "rag_action",
        )
    if node == "answer":
        if upd.get("answer_draft"):
            return "Generated an answer grounded in the retrieved chunks.", {}, "answer_generation"
        return "Model found no supporting facts in the chunks.", {}, "answer_generation"
    if node == "not_found":
        return "Information not found in the knowledge base.", {}, "not_found"
    return f"Executing node: {node}", {}, "generic_node_execution"


# --- Chat Endpoint ---
@app.post("/chat/", response_model=AgentResponse)
async def chat_with_agent(request: QueryRequest):
    events: List[TraceEvent] = []
    config = {
        "configurable": {"thread_id": request.session_id},
        # Phoenix groups traces into a "session" using this metadata key.
        "metadata": {"session_id": request.session_id},
        "run_name": "rag_agent_chat",
    }
    try:
        inputs = {"messages": [HumanMessage(content=request.query)]}
        step = 0
        for update in rag_agent.stream(inputs, config=config, stream_mode="updates"):
            for node, upd in update.items():
                step += 1
                desc, details, etype = _describe(node, upd or {})
                events.append(TraceEvent(step=step, node_name=node, description=desc, details=details, event_type=etype))

        # Read the final answer from the checkpoint: robust to which node finished last.
        final = rag_agent.get_state(config).values
        message = next((m.content for m in reversed(final.get("messages", [])) if isinstance(m, AIMessage)), "")
        if not message:
            raise HTTPException(status_code=500, detail="Agent did not return a response.")
        return AgentResponse(response=message, outcome=final.get("outcome") or "answered", trace_events=events)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        stats.inc("errors.chat")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/stats")
async def get_stats():
    """In-process counters (requests, outcomes, guardrail categories). Full traces live in Phoenix."""
    return stats.snapshot()


@app.on_event("shutdown")
def _shutdown():
    flush()

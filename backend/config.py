import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embedding Model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths (adjust as needed)
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR", "data")

# Groq chat model (llama3-70b-8192 and llama-3.3-70b-versatile have been shut down)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- RAG behaviour -------------------------------------------------------
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
NOT_FOUND_MESSAGE = os.getenv(
    "NOT_FOUND_MESSAGE",
    "This information does not exist in the knowledge base. "
    "I can only answer from the documents that have been uploaded.",
)

# --- Input guardrails ----------------------------------------------------
GUARDRAIL_MAX_CHARS = int(os.getenv("GUARDRAIL_MAX_CHARS", "2000"))
# Layer 2 (LLM safety classifier). Layer 1 (regex rules) is always on.
GUARDRAIL_LLM_ENABLED = _bool("GUARDRAIL_LLM_ENABLED", True)

# --- Observability (Arize Phoenix, open source) --------------------------
PHOENIX_ENABLED = _bool("PHOENIX_ENABLED", True)
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "rag-agent")

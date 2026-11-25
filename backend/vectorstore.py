# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import API keys from config (only Pinecone is needed here now)
from config import PINECONE_API_KEY

# Set environment variables for Pinecone
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Hugging Face embedding model
# This will download the model the first time it's used.
# The default model for HuggingFaceEmbeddings is 'sentence-transformers/all-MiniLM-L6-v2'
# which has a dimension of 384.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
INDEX_NAME = "capstone-store" 

# --- Retriever (Existing function) ---
import time

def ensure_index():
    """Create the Pinecone index if it doesn't exist yet."""
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(60):
            if pc.describe_index(INDEX_NAME).status["ready"]:
                break
            time.sleep(2)
        print(f"Created new Pinecone index: {INDEX_NAME}")


def get_retriever(k: int = 5):
    ensure_index()
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def add_document_to_vectorstore(text_content: str) -> int:
    """Chunk, embed and upsert one document. Returns the number of chunks indexed."""
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    ensure_index()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    documents = text_splitter.create_documents([text_content])
    print(f"Splitting document into {len(documents)} chunks for indexing...")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
    return len(documents)

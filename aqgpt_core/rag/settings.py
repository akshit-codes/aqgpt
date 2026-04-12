"""Configuration for AQGPT RAG over urbanemissions.info."""

import os
from pathlib import Path

# Workspace root: /.../aqgpt
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# Use the pre-indexed UE vector DB by default.
# You can override with RAG_CHROMA_DIR if needed.
RAG_CHROMA_DIR = Path(
	os.getenv("RAG_CHROMA_DIR", str(WORKSPACE_ROOT / "ue" / "chroma_db"))
)

# Retrieval settings
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "urbanemissions")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

# Generation settings
# provider: ollama | qwen_api
RAG_GENERATION_PROVIDER = os.getenv("RAG_GENERATION_PROVIDER", "ollama").lower()
RAG_GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL", "mistral")
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434")

# Query-time safety
MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "8"))
MAX_CHUNKS_PER_URL = int(os.getenv("RAG_MAX_CHUNKS_PER_URL", "2"))

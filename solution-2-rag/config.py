"""Settings for the RAG document agent."""
import os

from dotenv import load_dotenv

load_dotenv()

# BYO Framework: point the OpenAI client at local mimOE (same as solution 1).
BASE_URL = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-airouter/openai/v1")
API_KEY = os.getenv("MIMOE_API_KEY", "1234")
MODEL = os.getenv("MIMOE_MODEL", "qwen3-4b")
TEMPERATURE = 0.2
MAX_TOKENS = 512
MAX_ITERATIONS = 5

DOCS_DIR = "documents"       # source PDFs (private, gitignored)
CHROMA_DIR = "chroma_db"     # persistent vector base (generated, gitignored)
COLLECTION = "home_docs"

CHUNK_SIZE = 1000            # recursive chunking; big enough to keep a clause intact
CHUNK_OVERLAP = 150
TOP_K = 4                    # chunks retrieved per query

# Local embeddings; independent of the chat model, so swapping models needs no re-index.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

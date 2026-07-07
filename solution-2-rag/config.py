"""Settings for the RAG document agent."""
import os

from dotenv import load_dotenv

load_dotenv()

# mimOE endpoint (same BYO-Framework swap as solution 1: point OpenAI-style code at mimOE).
BASE_URL = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-airouter/openai/v1")
API_KEY = os.getenv("MIMOE_API_KEY", "1234")
MODEL = os.getenv("MIMOE_MODEL", "qwen3-4b")
TEMPERATURE = 0.2
MAX_TOKENS = 512
MAX_ITERATIONS = 5   # backstop on the agent's tool loop

# Paths.
DOCS_DIR = "documents"       # source PDFs (private, gitignored)
CHROMA_DIR = "chroma_db"     # persistent vector base (generated, gitignored)
COLLECTION = "home_docs"

# Chunking: recursive with overlap (the workhorse); larger chunks keep a clause intact.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval: how many chunks to pull per query.
TOP_K = 4

# Local embeddings via sentence-transformers. Independent of the chat model, so
# switching qwen3 -> a bigger model needs NO re-indexing (only changing THIS does).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

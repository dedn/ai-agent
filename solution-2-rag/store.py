"""Shared vector store: sentence-transformers embeddings + persistent Chroma.

One base for all documents. Both the ingest script and the agent's tools use this,
so they read/write the same base.
"""
import contextlib
import io
import logging
import os
import warnings

# Keep the CLI clean: silence the embedding model's load noise (HF progress bars and
# the "unauthenticated HF Hub" notice). Must run before the huggingface imports below.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_chroma import Chroma  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

import config  # noqa: E402

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Return the shared Chroma store (built once; the embedding model loads lazily)."""
    global _vectorstore
    if _vectorstore is None:
        # The embedding model prints a "Loading weights" tqdm bar on load; hush it
        # (redirect only captures printed output — real errors still raise).
        with contextlib.redirect_stderr(io.StringIO()):
            embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=config.COLLECTION,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_DIR,
        )
    return _vectorstore

"""Shared vector store: sentence-transformers embeddings + persistent Chroma.

One base for all documents. Both the ingest script and the agent's tools use this,
so they read/write the same base.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Return the shared Chroma store (built once; the embedding model loads lazily)."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=config.COLLECTION,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_DIR,
        )
    return _vectorstore

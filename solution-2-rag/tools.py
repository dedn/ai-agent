"""Agent tools: search the document base, list indexed docs, add a new document.

Each is a LangChain @tool; the model decides when to call them (agentic RAG).
"""
import os

from langchain_core.tools import tool

import config
from store import get_vectorstore


def _cite(doc) -> str:
    """A short '[source p.N]' citation for a retrieved chunk (page is 0-indexed)."""
    page = doc.metadata.get("page")
    page_str = f" p.{page + 1}" if isinstance(page, int) else ""
    return f"[{doc.metadata.get('source', '?')}{page_str}]"


@tool
def search_documents(query: str) -> str:
    """Search the user's personal documents (contracts, policies) for anything relevant
    to the query. Use this for ANY question about the user's documents. Returns the most
    relevant excerpts, each with its source file and page."""
    docs = get_vectorstore().similarity_search(query, k=config.TOP_K)
    if not docs:
        return "No relevant information found in the documents."
    return "\n\n".join(f"{_cite(d)}\n{d.page_content}" for d in docs)


@tool
def list_documents() -> str:
    """List the documents currently indexed in the knowledge base."""
    metadatas = get_vectorstore().get().get("metadatas") or []
    sources = sorted({m.get("source", "?") for m in metadatas})
    return "Indexed documents: " + (", ".join(sources) if sources else "(none)")


@tool
def add_document(path: str) -> str:
    """Index a new PDF into the knowledge base by its file path."""
    from ingest import index_pdf
    if not os.path.exists(path):
        return f"File not found: {path}"
    added = index_pdf(path, get_vectorstore())
    return f"Added {added} chunks from {os.path.basename(path)}." if added else "Already indexed."


TOOLS = [search_documents, list_documents, add_document]

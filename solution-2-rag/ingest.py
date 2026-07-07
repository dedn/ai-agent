"""Phase 1 - indexing: load PDFs from documents/ into the Chroma base.

Re-runnable and incremental: a document already in the base is skipped, so adding a
new file = drop it in documents/ and run this again (`python ingest.py`).
"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from store import get_vectorstore


def index_pdf(path: str, vectorstore) -> int:
    """Index one PDF. Returns the number of chunks added (0 if already indexed)."""
    source = os.path.basename(path)
    if vectorstore.get(where={"source": source})["ids"]:
        return 0   # already in the base

    pages = PyPDFLoader(path).load()   # one Document per page, with page metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["source"] = source   # clean source name for citations
    vectorstore.add_documents(chunks)
    return len(chunks)


def main() -> None:
    if not os.path.isdir(config.DOCS_DIR):
        print(f"No '{config.DOCS_DIR}/' directory.")
        return
    pdfs = [f for f in os.listdir(config.DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs in '{config.DOCS_DIR}/' - drop your documents there first.")
        return

    vectorstore = get_vectorstore()
    for f in pdfs:
        added = index_pdf(os.path.join(config.DOCS_DIR, f), vectorstore)
        print(f"  {f}: {('added ' + str(added) + ' chunks') if added else 'already indexed'}")
    print("Done.")


if __name__ == "__main__":
    main()

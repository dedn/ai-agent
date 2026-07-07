"""Phase 1 - indexing: load PDFs from documents/ into the Chroma base.

Re-runnable and incremental: a document already in the base is skipped, so adding a
new file = drop it in documents/ and run this again (`python ingest.py`).
"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from store import get_vectorstore


def index_pdf(path: str, vectorstore) -> tuple[str, int]:
    """Index one PDF. Returns (status, chunks): status is
    'added' | 'skipped' (already indexed) | 'empty' (no extractable text)."""
    source = os.path.basename(path)
    if vectorstore.get(where={"source": source})["ids"]:
        return ("skipped", 0)

    pages = PyPDFLoader(path).load()   # one Document per page, with page metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        return ("empty", 0)            # scanned/image PDF -> would need OCR
    for c in chunks:
        c.metadata["source"] = source   # clean source name for citations
    vectorstore.add_documents(chunks)
    return ("added", len(chunks))


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
        status, n = index_pdf(os.path.join(config.DOCS_DIR, f), vectorstore)
        if status == "added":
            note = f"added {n} chunks"
        elif status == "empty":
            note = "no extractable text (scanned PDF? needs OCR)"
        else:
            note = "already indexed"
        print(f"  {f}: {note}")
    print("Done.")


if __name__ == "__main__":
    main()

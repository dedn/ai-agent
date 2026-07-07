"""Inspect the local Chroma base: how many chunks and from which documents.

    python inspect_db.py

Prints counts and metadata only (no document text) - safe for private files.
"""
from collections import Counter

from store import get_vectorstore


def main() -> None:
    collection = get_vectorstore()._collection
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
    by_source = Counter(m.get("source", "?") for m in metadatas)

    print(f"collection : {collection.name}")
    print(f"chunks     : {collection.count()}")
    print(f"embedding  : {len(sample[0]) if len(sample) else 0}-dim")
    print("documents  :")
    for source, n in sorted(by_source.items()):
        print(f"  - {source}: {n} chunks")


if __name__ == "__main__":
    main()

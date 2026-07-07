# Solution 2 — RAG agent over your personal documents

A local agent that answers questions about your own documents (contracts, policies)
and never sends them to the cloud. Built with **LangChain + Chroma + a local mimOE
model**. This is the **framework path** — contrast with solution 1's raw SDK.

## Purpose

Ask plain questions about your home paperwork — "when does the tenancy end?", "is
subletting allowed?", "what's the deposit?" — and get answers grounded in the actual
documents, with a source-and-page citation. Everything runs on-device: private,
offline, zero per-token cost.

## Important rules

- **Two phases.** `ingest.py` indexes PDFs from `documents/` into a persistent Chroma
  base (phase 1, offline). `agent.py` answers questions over that base (phase 2).
- **Agentic RAG via LangChain `create_agent`.** The model decides when to call
  `search_documents`; this is the LangChain 1.x agent API (LangGraph is the runtime
  underneath — we don't author a graph). See ../DECISIONS.md §0017.
- **Grounded answers only.** The system prompt tells the model to answer ONLY from what
  `search_documents` returns and to cite source + page; if the docs don't cover it, say
  so plainly (no hallucinating).
- **One base for all documents.** Chunks carry `source` metadata; adding a document =
  indexing it into the same base.
- **Embeddings are local and independent of the chat model** (sentence-transformers).
  Switching qwen3 → a bigger model needs NO re-index; only changing the embedding model does.
- Reliability knobs mirror solution 1: `temperature=0.2`, `max_tokens`, an iteration cap.

## Do not

- **Do not commit `documents/` or `chroma_db/`** — they hold private data and are gitignored.
- **Do not change the embedding model without re-indexing** — old vectors become
  incompatible (delete `chroma_db/` and re-run `ingest.py`).
- **Do not expect cross-session chat memory** — the document base persists, but the
  conversation is per-session (see Limitations).
- **Do not read, print, or commit `.env`** — use `.env.example`.

## What it does

```
You: What documents do you have?
Agent: housing-tenancy-agreement.pdf

You: When does the tenancy start and end?
Agent: 01-August-2026 to 1PM on 31-July-2027. [housing-tenancy-agreement.pdf p.2]

You: What is my dog's name?
Agent: I don't have that information in the documents.
```

## Architecture — how the components connect

```
  documents/*.pdf ─(ingest.py)─► chunk (recursive) ─► embed (sentence-transformers)
                                                          │
                                                          ▼
                                                   Chroma  (one base, on disk)
                                                          ▲
  You ──► agent.py (LangChain create_agent) ──► search_documents tool ──┘
                    │  (also: list_documents, add_document)
                    ▼
             answer grounded in the retrieved chunks, cited [source p.N]
```

- **`ingest.py`** — phase 1: load PDFs (pypdf), split (recursive + overlap), embed, store.
- **`store.py`** — the shared Chroma base + embeddings (used by ingest and the tools).
- **`tools.py`** — `search_documents` (retriever), `list_documents`, `add_document`.
- **`agent.py`** — the LangChain agent (`create_agent`) + chat loop; strips qwen3 `<think>`.
- **mimOE** — the local model server; the LLM answers, all on-device.

## Run it

Prereqs: mimOE Studio running with a tool-use model loaded (default `qwen3-4b`), Python 3.11+.

```bash
cd solution-2-rag
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# put your PDFs in documents/, then:
.venv/bin/python ingest.py      # phase 1: index (re-runnable, incremental)
.venv/bin/python agent.py       # phase 2: chat
```

## Adding a new document

- **Bulk / CLI:** drop the PDF in `documents/` and run `python ingest.py` (already-indexed
  files are skipped).
- **In chat:** "add the document at documents/strata-policy.pdf" → the agent calls the
  `add_document` tool.

## Files

| File | Role |
|---|---|
| `config.py` | settings (mimOE, paths, chunking, top-k, embedding model) |
| `store.py` | shared Chroma base + sentence-transformers embeddings |
| `ingest.py` | phase 1 — index PDFs into the base (re-runnable, incremental) |
| `tools.py` | `search_documents`, `list_documents`, `add_document` |
| `agent.py` | LangChain `create_agent` + chat loop |
| `.env.example` | mimOE connection template |

## Limitations & next steps

- **Vector search only.** Hybrid search (BM25 + RRF), reranking, and GraphRAG are
  deliberately out of scope for a single-source demo (../DECISIONS.md §0020).
- **No cross-session conversation memory.** The document base persists across restarts,
  but the chat history does not. The natural next step is long-term memory as
  RAG-over-history — the same Chroma machinery, a separate collection (§0021).
- **Small-model limits.** On qwen3-4b, retrieval + a two-step agent loop take a few
  seconds per answer; a bigger model (one config line) improves quality with no re-index.

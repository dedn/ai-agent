# AGENTS.md — rules for contributors (human or AI)

Source of truth for structure and conventions. The rationale (why) lives in
[DECISIONS.md](./DECISIONS.md).

## What this project is

The mimOE assignment: a small AI agent connected to a **local, OpenAI-compatible
mimOE endpoint** via the **BYO Framework** approach. The goal is that it **works**
and that its **approach, tooling choices, and component wiring** are easy to explain.
Delivered as **two solutions** (see DECISIONS §0001).

## Repository shape

```
solution-1-raw-agent/   # solution 1: raw OpenAI SDK, manual agent loop
solution-2-rag/         # solution 2: LangChain + Chroma RAG over your documents
AGENTS.md CLAUDE.md DECISIONS.md README.md run.sh
```

## Solution 1 shape

```
solution-1-raw-agent/
  agent.py       # agent loop, chat interface, <think> stripping, level-2 compaction
  console.py     # terminal UX: thinking spinner + token/context line
  config.py      # settings (base_url, model, temperature, limits) — LEAF
  tools/         # one tool = one file + __init__ aggregates
    __init__.py    builds TOOLS_SCHEMA + TOOL_FUNCTIONS
    calculator.py  logic + schema (own code, standard library)
    weather.py     logic + schema (wrapper over the requests package)
  requirements.txt .env.example .gitignore README.md
```

## Solution 2 shape

```
solution-2-rag/
  ingest.py      # phase 1: index PDFs from documents/ into Chroma (re-runnable)
  store.py       # shared Chroma base + sentence-transformers embeddings
  tools.py       # search_documents, list_documents, add_document
  agent.py       # LangChain create_agent + chat loop; strips <think>
  inspect_db.py  # show what's indexed in the base (counts, per-document)
  config.py      # settings (mimOE, paths, chunking, top-k, embed model) — LEAF
  documents/     # PDFs; a sample is committed, real docs are gitignored
  chroma_db/     # the vector base (generated, gitignored)
  requirements.txt .env.example .gitignore README.md
```

## Conventions

- **One tool = one file**, logic and its OpenAI schema together. No generic names
  (`utils.py`, `helpers.py`, `service.py`). `tools/__init__.py` is the only
  aggregator (`TOOLS_SCHEMA` + `TOOL_FUNCTIONS`). Adding a tool = a new file + 2
  lines in `__init__`.
- **Dependency direction:** `agent.py → tools/ + config.py`. `tools/` and
  `config.py` never import `agent.py`; `config.py` is a leaf.
- **Config-driven backend swap:** switching engine (mimOE ↔ Ollama ↔ cloud) = change
  `base_url` in one place. The OpenAI-style code doesn't move.
- **Reasoning model (qwen3):** `/no_think` in the system prompt + `strip_think()`
  removes `<think>` on output AND before storing to history.
- **Reliability:** `temperature=0.2` (predictability over creativity), `MAX_ITERATIONS`
  as a backstop against runaway loops (LOOP).
- **Tool description = contract:** state the exact input format in `description`
  (the model sends exactly what's described — see DECISIONS §0014).
- **Model is a variable:** the model name lives in `.env`/`config` and must match the
  model Loaded in mimOE.

## Setup & run

```bash
cd solution-1-raw-agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # adjust if your mimOE URL/model differ
.venv/bin/python agent.py
```
Requires mimOE Studio running with a model loaded (default `qwen3-4b`).

## Security

- Never read, print, or commit `.env` or keys. `.env` is gitignored; use
  `.env.example` for documentation. The local mimOE key is a stub (`1234`).

## Documentation discipline

- Keep each solution's README and `DECISIONS.md` current when behavior changes.
  Here this is a convention, not an automated gate.

## Scope (intentionally NOT done)

- No CI machinery (husky / lint-staged / pre-commit docs gate / import-boundary
  linting) — overkill for a small demo. We kept the essence: docs-as-contract,
  per-file ownership, an ADR log. See DECISIONS §0013.

## Git

- Short, human commit subjects; no AI attribution in git or PRs.

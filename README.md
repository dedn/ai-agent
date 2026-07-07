# AI agent on local mimOE

A small AI agent that connects to a **local, OpenAI-compatible mimOE inference
endpoint** using the **BYO Framework** approach — the model runs entirely on-device
(private, offline, zero per-token cost).

The goal is not complexity but a **working** agent whose **approach, tooling choices,
and component wiring** are easy to explain. Delivered as two solutions.

## Structure

```
solution-1-raw-agent/   # raw OpenAI SDK: manual agent loop, native tool calling
solution-2-rag/         # RAG over your documents (LangChain + Chroma)
AGENTS.md               # conventions for contributors (human or AI)
DECISIONS.md            # why the project is built this way (ADR log)
CLAUDE.md               # short orientation pointer
run.sh                  # launcher: ./run.sh 1 | 2 | ingest
```

## Quick start (solution 1)

```bash
cd solution-1-raw-agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # adjust to your mimOE URL / loaded model
.venv/bin/python agent.py
```

Requires mimOE Studio running with a tool-use-capable model **loaded**
(default `qwen3-4b`).

## Quick start (solution 2 — RAG over your documents)

```bash
cd solution-2-rag
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# put your PDFs in documents/, then:
.venv/bin/python ingest.py      # index your documents
.venv/bin/python agent.py       # ask questions about them
```

## Run (after setup)

Once each solution's venv exists (the steps above), launch from the repo root — the
script picks the right venv and directory automatically:

```bash
./run.sh 1        # solution 1 (raw agent)
./run.sh 2        # solution 2 (RAG document agent)
./run.sh ingest   # (re)index solution 2's documents/
```

See each solution's `README.md` for details, [AGENTS.md](AGENTS.md) for conventions,
and [DECISIONS.md](DECISIONS.md) for the reasoning behind each choice.

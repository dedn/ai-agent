# AI agent on local mimOE

A small AI agent that connects to a **local, OpenAI-compatible mimOE inference
endpoint** using the **BYO Framework** approach — the model runs entirely on-device
(private, offline, zero per-token cost).

The goal is not complexity but a **working** agent whose **approach, tooling choices,
and component wiring** are easy to explain. Delivered as two solutions.

## Structure

```
solution-1-raw-agent/   # raw OpenAI SDK: manual agent loop, native tool calling
solution-2-rag/         # LangChain + Chroma RAG agent (planned)
AGENTS.md               # conventions for contributors (human or AI)
DECISIONS.md            # why the project is built this way (ADR log)
CLAUDE.md               # short orientation pointer
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

See [solution-1-raw-agent/README.md](solution-1-raw-agent/README.md) for details,
[AGENTS.md](AGENTS.md) for conventions, and [DECISIONS.md](DECISIONS.md) for the
reasoning behind each choice.

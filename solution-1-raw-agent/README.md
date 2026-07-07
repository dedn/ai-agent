# Solution 1 — Raw API agent on local mimOE

A minimal AI agent that connects to a **local mimOE inference endpoint** (an
OpenAI-compatible API running entirely on-device) using the **raw `openai`
Python SDK** — no agent framework. It demonstrates the mechanics of an agent
"by hand": native tool calling and the agent loop.

This is the **transparent / fundamentals** solution. A second solution
(`../solution-2-rag`) builds a RAG agent with LangChain to show the framework path.

## Purpose

Answer a user's questions in a terminal chat and, when a task needs a tool,
**let the model decide on its own** to call one — a `calculator` (pure local code)
or `get_weather` (a package/API wrapper). Everything runs locally via mimOE:
private, offline, zero per-token cost.

## Important rules

- **Raw SDK by design** — no LangChain here. Every moving part (message list, tool
  schema, agent loop, dispatch) is explicit and explainable. That is the point of
  this solution.
- **BYO Framework = swap `base_url`.** mimOE is OpenAI-compatible, so the same code
  talks to Ollama / cloud by changing one setting in `config.py`.
- **One tool = one file** under `tools/` (logic + its OpenAI schema together);
  `tools/__init__.py` is the only aggregator. No generic names.
- **Model is a variable** (`MIMOE_MODEL`) and must match the model **loaded** in
  mimOE (default `qwen3-4b`).
- **qwen3 is a reasoning model** — `/no_think` in the system prompt + `strip_think()`
  keep the output (and stored history) clean of `<think>` blocks.
- **Tool description = contract** — the model sends exactly what the `description`
  says (e.g. the calculator demands pure arithmetic; percentages → multiplication).
- Reliability knobs live in `config.py`: `temperature=0.2`, `MAX_ITERATIONS` backstop.

Full rationale for each choice: [../DECISIONS.md](../DECISIONS.md).

## Do not

- **Do not hardcode the machine IP** in the endpoint — use `localhost` (the LAN IP
  changes with the network; `localhost` does not).
- **Do not use `eval()`** in the calculator — it parses an AST so the model can't run
  arbitrary code through the tool.
- **Do not remove `<think>` stripping** — qwen3 emits reasoning blocks; unstripped
  they pollute the answer and the memory.
- **Do not read, print, or commit `.env`** — use `.env.example`.
- **Do not expect a bigger model to be fast** — `qwen3-4b` is ~9× slower than 1.7b
  on modest hardware; that's an accepted trade-off for reliability.

## What it does

```
You: What is 128 * 47 + 15?
  [tool] calculator({'expression': '128 * 47 + 15'}) = 6031
Agent: The result is 6031.

You: What's the weather in Prague?
  [tool] get_weather({'city': 'Prague'}) = Prague: overcast, 28.9°C, wind 17.3 km/h
Agent: In Prague it's overcast, 28.9°C, wind 17.3 km/h.
```

## Architecture — how the components connect

```
      ┌─────────────────────────── agent.py ───────────────────────────┐
      │  OpenAI() client  ──base_url──►  mimOE  (local, qwen3-4b)        │
      │      │                                                          │
      │  agent loop:  send messages + tools schema                      │
      │      │                                                          │
      │      ├─ model returns tool_calls?  ── no ──►  final answer      │
      │      │                                                          │
      │      └─ yes ─► run tool (tools/) ─► append result ─► loop       │
      └─────────────────────────────────────────────────────────────────┘
```

- **mimOE** = the runtime (model server). Text-in → text-out. It is **not** the
  agent — we bring the agent logic ourselves (BYO Framework).
- **`config.py`** — connection settings + design knobs (leaf module; imports nothing
  from us).
- **`tools/`** — one file per tool (`calculator.py`, `weather.py`): the Python
  function + its OpenAI schema; `__init__.py` builds `TOOLS_SCHEMA` + `TOOL_FUNCTIONS`.
  The model reads each description to decide *when* to call it; **we** execute it.
- **`agent.py`** — the `OpenAI` client (pointed at mimOE), the system prompt, and the
  **agent loop**: send messages + tools → if the model asks for a tool, run it, append
  the result, call the model again → repeat until a plain answer. Short-term memory is
  the growing `messages` list; optional level-2 summarization is in
  `compact_history()` (off by default).

## Run it

Prereqs: mimOE Studio running with a model **loaded** (default `qwen3-4b`), Python 3.11+.

```bash
cd solution-1-raw-agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # adjust if your mimOE URL/model differ
.venv/bin/python agent.py
```

Commands inside the chat: `/exit` to quit, `/reset` to clear memory.

## Files

| File | Role |
|---|---|
| `config.py` | connection settings + design knobs (temperature, max iterations, compaction) |
| `tools/calculator.py` | calculator tool (own code, AST-safe) + its schema |
| `tools/weather.py` | weather tool (via `requests` + Open-Meteo) + its schema |
| `tools/__init__.py` | aggregates `TOOLS_SCHEMA` + `TOOL_FUNCTIONS` |
| `agent.py` | client, system prompt, agent loop, `<think>` strip, compaction, chat |
| `.env.example` | mimOE connection template |

## Limitations & next steps

- Two tools (calculator, weather) — enough to show tool routing and that a tool can
  be own code **or** a package/API. More tools = a new file in `tools/` + 2 lines.
- Short-term memory only (the message list). Level-2 summarization exists but is off
  by default — for a short demo the context window is plenty (see DECISIONS §0012).
- Streaming is omitted on purpose (it complicates parsing tool calls; mimOE supports
  `"stream": true`).
- The natural next step is retrieval over your own documents — see Solution 2 (RAG).

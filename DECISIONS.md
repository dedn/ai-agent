# DECISIONS — decision log (ADR)

Why the project is built the way it is. Each entry: context → decision → why →
trade-off.

---

### 0001 — Two solutions instead of one
- **Context:** the assignment asks for a "simple" agent, but judgment and the ability
  to explain are what's assessed.
- **Decision:** ship TWO solutions — raw (mechanics by hand) and RAG (via a framework).
- **Why:** the contrast shows both under-the-hood understanding and command of an
  industry tool; each complements the other.
- **Trade-off:** more work, but a stronger demonstration of skills.

### 0002 — mimOE as a local inference server via BYO Framework
- **Decision:** use mimOE only as an OpenAI-compatible endpoint; bring our own agent
  (BYO), connecting by swapping `base_url`.
- **Why:** this is the essence of the task; the code stays provider-agnostic
  (mimOE ↔ Ollama ↔ cloud is a one-line change).
- **Trade-off:** we don't use mimOE's native agent-mims — but that isn't required.

### 0003 — Solution 1 = raw openai SDK, Solution 2 = LangChain
- **Why:** raw gives transparency and line-by-line explainability; LangChain gives the
  breadth of ready-made building blocks (RAG, retriever).
- **Rejected:** LlamaIndex (RAG-only, less general), CrewAI (multi-agent — overkill for
  a single agent).

### 0004 — Model qwen3-4b (over 1.7b)
- **Decision:** use qwen3-4b.
- **Why:** more reliable tool calling and summarization; fixed the 1.7b failure to
  recall a fact after history compaction.
- **Trade-off:** ~9× slower (~8.7 vs ~80 tok/s). For a demo, reliability wins.

### 0005 — temperature = 0.2
- **Why:** an agent needs predictability and stable tool calling, not creativity. Low
  temperature = reproducibility.

### 0006 — MAX_ITERATIONS backstop
- **Context:** the agent (not the model) can loop forever on vague inputs (LOOP).
- **Decision:** a hard cap on agent-loop steps.
- **Why:** guarantees termination even when the model stalls.

### 0007 — /no_think + strip_think for qwen3
- **Context:** qwen3 is a reasoning model that emits `<think>…</think>`.
- **Decision:** `/no_think` in the system prompt (less generation) + a regex that
  strips the block on output AND before storing to history.
- **Why:** a clean answer for the user, fewer tokens, clean memory.

### 0008 — Traceable base_url (mimik-airouter)
- **Decision:** hit `/mimik-airouter/openai/v1` rather than the direct
  `/mimik-ai/openai/v1`.
- **Why:** requests appear in the **Traces** tab — the agent loop, tool calls, and
  tokens are visible. Great for the demo. The response is functionally identical.

### 0009 — One tool = one file (tools/ package)
- **Decision:** `tools/calculator.py`, `tools/weather.py`, with `tools/__init__.py`
  aggregating.
- **Why:** clear per-file ownership — the path is obvious, tools are isolated, and
  adding one = a new file + 2 lines. Makes the codebase easy for the next contributor
  (human or agent) to navigate.

### 0010 — Calculator by hand (AST), weather via a package
- **Decision:** `calculator` is own code on the standard library (AST parsing, no
  `eval`); `get_weather` wraps the `requests` package + Open-Meteo.
- **Why:** (1) AST instead of `eval` = safety (the model can't run arbitrary code);
  (2) two tools of different nature show that a tool can be your own code OR a call to
  a package/API.

### 0011 — Embeddings not via mimOE (for RAG, solution 2)
- **Context:** the qwen3 model loaded in mimOE can't embed (`pooling type NONE`), and
  there's no embed model in the Recommended list.
- **Decision:** embed locally via `sentence-transformers` / Chroma's default; RAG uses
  vector search only.
- **Rejected (as next steps):** hybrid search + BM25/RRF, reranking, GraphRAG —
  overkill for a single FAQ source.

### 0012 — Level-2 history compaction: implemented but off
- **Decision:** auto-summarize old history when the window fills to ≥80%; default
  `COMPACT_ENABLED = False`.
- **Why:** a short demo fits the window; on a smaller model summarization is unreliable
  (hallucinates, loses detail). Kept in the code to demonstrate the concept.

### 0013 — No CI machinery
- **Decision:** skip husky / lint-staged / pre-commit docs gate / import-boundary
  linting.
- **Why:** overkill for a five-file demo. We took the essence of an AI-first setup
  (docs-as-contract, ownership, ADR log) without the heavy tooling. Cutting the excess
  to fit the scale is itself a decision.

### 0015 — Cap output length (max_tokens) and harden reasoning cleanup
- **Context:** on a hard/ambiguous prompt ("weather in every US state") the model ran
  away — reasoning in an unclosed `<think>` until it hit the server's ~2048-token
  default, leaking the raw reasoning into the answer.
- **Decision:** set `max_tokens` (512); strip an unclosed `<think>` too; fall back to a
  clear message if only reasoning was produced.
- **Why:** bounds runaway generation (cost, latency, garbage) and degrades gracefully.
  Answers here are short, so 512 is generous. A real small-model failure mode, handled.

### 0014 — Tool description = contract
- **Context:** for "15% of 2400" the model sent `"15% of 2400"` to the calculator,
  which failed to parse.
- **Decision:** tighten the `description` — "pure math only; convert percentages to
  multiplication (`2400 * 0.15`)".
- **Why:** the model sends exactly what the description says. A precise tool
  description directly controls agent behavior (context engineering in action).

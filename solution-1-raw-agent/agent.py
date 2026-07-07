"""Solution 1 — a minimal AI agent on the raw OpenAI SDK + local mimOE.

Demonstrates agent mechanics "by hand":
  - connecting to mimOE by swapping base_url (BYO Framework);
  - native function calling (the model decides whether to call a tool);
  - agent loop: model requests a tool -> we run it -> return result -> repeat;
  - a backstop against runaway loops via an iteration cap.
"""
import itertools
import json
import re
import sys
import threading
import time

from openai import OpenAI

import config
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

# An "OpenAI" client that points at local mimOE (the essence of BYO Framework).
client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)

# Token bookkeeping, taken from `usage` in each API response.
_last_prompt_tokens = 0        # window fill on the last call -> auto-compaction + display
_turn_prompt_tokens = 0        # summed input tokens for the current turn
_turn_completion_tokens = 0    # summed output tokens for the current turn

# The system prompt sets the role. /no_think disables qwen3's reasoning mode.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely in plain text, without LaTeX or "
    "formulas. For any arithmetic ALWAYS use the calculator tool instead of computing "
    "in your head. For weather questions use the get_weather tool. /no_think"
)


class Spinner:
    """A small stdout spinner shown while a blocking call runs (TTY only)."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread = None
        self._active = sys.stdout.isatty()   # no animation when piped/redirected

    def __enter__(self):
        if self._active:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {frame} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.08)

    def __exit__(self, *exc):
        if self._active:
            self._stop.set()
            self._thread.join()
            sys.stdout.write("\r\033[K")   # clear the spinner line
            sys.stdout.flush()


def usage_line() -> str:
    """One-line context/token summary for the turn just finished."""
    window = config.MAX_CONTEXT_TOKENS
    pct = (100 * _last_prompt_tokens / window) if window else 0
    return (f"  [ctx {_last_prompt_tokens}/{window} ({pct:.0f}%) "
            f"| this turn: {_turn_prompt_tokens} in + {_turn_completion_tokens} out]")


def _track_usage(response) -> None:
    """Update token counters from an API response."""
    global _last_prompt_tokens, _turn_prompt_tokens, _turn_completion_tokens
    if response.usage:
        _last_prompt_tokens = response.usage.prompt_tokens
        _turn_prompt_tokens += response.usage.prompt_tokens
        _turn_completion_tokens += response.usage.completion_tokens


def strip_think(text: str) -> str:
    """Strip qwen3's <think>...</think> reasoning markup.

    Even with /no_think the model sometimes emits an empty or UNCLOSED <think>
    (e.g. it rambled and got cut off at max_tokens mid-reasoning), so:
      1) remove well-formed <think>...</think> blocks;
      2) remove an unclosed <think> that runs to the end (cut-off reasoning);
      3) remove any orphan <think> / </think> tags.
    """
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)   # closed blocks
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)           # unclosed (cut off mid-think)
    text = re.sub(r"</?think>", "", text)                            # orphan tags
    return text.strip()


def compact_history(history: list) -> None:
    """Level-2 memory: fold the old part of the dialogue into a short summary.

    Off by default (config.COMPACT_ENABLED). Triggers when the history grows past the
    threshold. NOT RAG, just text compaction so the window doesn't overflow. Mutates
    history IN PLACE (slice assignment) so the reference in main stays valid.
    """
    if not config.COMPACT_ENABLED:
        return
    # Trigger: window >= threshold (by real tokens) OR too many messages.
    token_limit = config.COMPACT_AT_FRACTION * config.MAX_CONTEXT_TOKENS
    if _last_prompt_tokens < token_limit and len(history) <= config.COMPACT_AFTER_MESSAGES:
        return

    # Safe boundary: the start of the last user turn. Avoids splitting assistant->tool
    # pairs and keeps the most recent exchange verbatim.
    cut = None
    for i in range(len(history) - 1, 0, -1):
        if history[i]["role"] == "user":
            cut = i
            break
    if cut is None or cut <= 1:
        return

    old = history[1:cut]   # between the system prompt and the last user turn
    convo = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in old if m.get("content"))
    print(f"  [memory] window is full (~{_last_prompt_tokens} tokens) - summarizing older history...")

    with Spinner("summarizing"):
        summary = client.chat.completions.create(
            model=config.MODEL,
            messages=[{
                "role": "user",
                "content": "Summarize the key facts of this dialogue (names, numbers, topics) "
                           "in 2-3 sentences. Summary only, no preamble. /no_think\n\n" + convo,
            }],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        ).choices[0].message.content
    summary = strip_think(summary)

    # In-place mutation: replace the whole old chunk with a single summary message.
    history[1:cut] = [{"role": "system", "content": f"Summary of the earlier dialogue: {summary}"}]


def run_agent(user_input: str, history: list) -> str:
    """One agent "turn": run the agent loop until a final answer.

    history is the message list (short-term memory). We mutate it in place so the
    conversation remembers context across turns.
    """
    global _turn_prompt_tokens, _turn_completion_tokens
    compact_history(history)   # level-2: auto-compact if the window is nearly full
    history.append({"role": "user", "content": user_input})
    _turn_prompt_tokens = 0    # reset per-turn token counters
    _turn_completion_tokens = 0

    for _ in range(config.MAX_ITERATIONS):
        with Spinner("thinking"):
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=history,
                tools=TOOLS_SCHEMA,          # give the model a "menu" of tools
                tool_choice="auto",          # the model decides whether to call a tool
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
        _track_usage(response)               # window fill + turn totals
        msg = response.choices[0].message

        # Store the model's reply in history ALREADY without <think>, to avoid
        # accumulating clutter and wasting context. May include a tool request.
        content = strip_think(msg.content)
        assistant_msg = {"role": "assistant", "content": content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        history.append(assistant_msg)

        # No tool request -> this is the final text answer. If the model only produced
        # (now-stripped) reasoning and left nothing, fall back to a clear message.
        if not msg.tool_calls:
            return content or "(couldn't produce a clean answer - try rephrasing or narrowing the request)"

        # Otherwise run each requested tool and return its result. A malformed tool
        # call (bad JSON args, wrong params) is caught and returned as text so the
        # model can recover instead of crashing the loop.
        for tc in msg.tool_calls:
            name = tc.function.name
            fn = TOOL_FUNCTIONS.get(name)
            try:
                args = json.loads(tc.function.arguments)
                result = fn(**args) if fn else f"unknown tool: {name}"
            except Exception as exc:
                result = f"tool error: {exc}"
            print(f"  [tool] {name}({tc.function.arguments})")   # show the call; result flows into the answer
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
        # ...and loop again: the model sees the result and continues.

    # Hit the cap -> likely a runaway loop (LOOP). Stop.
    return "Reached the step limit (possible LOOP) - stopping."


def main() -> None:
    print("mimOE agent (solution 1, raw OpenAI SDK).")
    print("Commands: /exit to quit, /reset to clear memory.\n")

    # Short-term memory = the message list, starting with the system prompt.
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "exit", "quit"):
            print("Bye!")
            break
        if user_input.lower() == "/reset":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(memory cleared)\n")
            continue

        try:
            answer = run_agent(user_input, history)
        except Exception as exc:
            answer = f"(error talking to mimOE: {exc})"
        print(f"Agent: {answer}")
        print(usage_line() + "\n")


if __name__ == "__main__":
    main()

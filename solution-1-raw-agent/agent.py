"""Solution 1 — a minimal AI agent on the raw OpenAI SDK + local mimOE.

Demonstrates agent mechanics "by hand":
  - connecting to mimOE by swapping base_url (BYO Framework);
  - native function calling (the model decides whether to call a tool);
  - agent loop: model requests a tool -> we run it -> return result -> repeat;
  - a backstop against runaway loops via an iteration cap.
"""
import json
import re

from openai import OpenAI

import config
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

# An "OpenAI" client that points at local mimOE (the essence of BYO Framework).
client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)

# How many tokens the last request used (usage.prompt_tokens from the API).
# Auto-compaction uses it to gauge how full the context window is.
_last_prompt_tokens = 0

# The system prompt sets the role. /no_think disables qwen3's reasoning mode.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely in plain text, without LaTeX or "
    "formulas. For any arithmetic ALWAYS use the calculator tool instead of computing "
    "in your head. For weather questions use the get_weather tool. /no_think"
)


def strip_think(text: str) -> str:
    """Strip qwen3's <think>...</think> reasoning markup.

    Even with /no_think the model sometimes emits an empty or UNCLOSED <think>, so:
      1) remove well-formed <think>...</think> blocks;
      2) remove orphan <think> / </think> tags (their content is the answer itself).
    """
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
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

    summary = client.chat.completions.create(
        model=config.MODEL,
        messages=[{
            "role": "user",
            "content": "Summarize the key facts of this dialogue (names, numbers, topics) "
                       "in 2-3 sentences. Summary only, no preamble. /no_think\n\n" + convo,
        }],
        temperature=config.TEMPERATURE,
    ).choices[0].message.content
    summary = strip_think(summary)

    # In-place mutation: replace the whole old chunk with a single summary message.
    history[1:cut] = [{"role": "system", "content": f"Summary of the earlier dialogue: {summary}"}]


def run_agent(user_input: str, history: list) -> str:
    """One agent "turn": run the agent loop until a final answer.

    history is the message list (short-term memory). We mutate it in place so the
    conversation remembers context across turns.
    """
    global _last_prompt_tokens
    compact_history(history)   # level-2: auto-compact if the window is nearly full
    history.append({"role": "user", "content": user_input})

    for _ in range(config.MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=history,
            tools=TOOLS_SCHEMA,          # give the model a "menu" of tools
            tool_choice="auto",          # the model decides whether to call a tool
            temperature=config.TEMPERATURE,
        )
        # Remember window fill; auto-compaction decisions are based on it.
        if response.usage:
            _last_prompt_tokens = response.usage.prompt_tokens
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

        # No tool request -> this is the final text answer.
        if not msg.tool_calls:
            return content

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
            print(f"  [tool] {name}({tc.function.arguments}) = {result}")   # transparency
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
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()

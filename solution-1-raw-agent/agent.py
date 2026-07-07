"""Solution 1 - a minimal AI agent on the raw OpenAI SDK + local mimOE.

Shows agent mechanics by hand: BYO-Framework connection (swap base_url), native
function calling, an agent loop with an iteration backstop, and short-term memory.
"""
import json
import re

from openai import OpenAI

import config
from console import Spinner, usage_line
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)

# token counters (usage line + compaction)
_last_prompt_tokens = 0
_turn_prompt_tokens = 0
_turn_completion_tokens = 0

# /no_think disables qwen3's reasoning mode.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely in plain text, without LaTeX or "
    "formulas. For any arithmetic ALWAYS use the calculator tool instead of computing "
    "in your head. For weather questions use the get_weather tool. /no_think"
)


def strip_think(text: str) -> str:
    """Remove qwen3's <think> reasoning: closed blocks, an unclosed tail, orphan tags."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _track_usage(response) -> None:
    global _last_prompt_tokens, _turn_prompt_tokens, _turn_completion_tokens
    if response.usage:
        _last_prompt_tokens = response.usage.prompt_tokens
        _turn_prompt_tokens += response.usage.prompt_tokens
        _turn_completion_tokens += response.usage.completion_tokens


def compact_history(history: list) -> None:
    """Level-2 memory (off by default): summarize old turns when the window fills.

    Cuts at the last user turn so assistant->tool pairs stay intact; mutates in place.
    """
    if not config.COMPACT_ENABLED:
        return
    token_limit = config.COMPACT_AT_FRACTION * config.MAX_CONTEXT_TOKENS
    if _last_prompt_tokens < token_limit and len(history) <= config.COMPACT_AFTER_MESSAGES:
        return
    cut = next((i for i in range(len(history) - 1, 0, -1) if history[i]["role"] == "user"), None)
    if cut is None or cut <= 1:
        return

    convo = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in history[1:cut] if m.get("content"))
    print(f"  [memory] window full (~{_last_prompt_tokens} tokens) - summarizing...")
    with Spinner("summarizing"):
        summary = client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content":
                       "Summarize the key facts (names, numbers, topics) in 2-3 sentences. "
                       "Summary only, no preamble. /no_think\n\n" + convo}],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        ).choices[0].message.content
    history[1:cut] = [{"role": "system", "content": f"Summary of earlier turns: {strip_think(summary)}"}]


def run_agent(user_input: str, history: list) -> str:
    """Run the agent loop for one user turn and return the final answer."""
    global _turn_prompt_tokens, _turn_completion_tokens
    compact_history(history)
    history.append({"role": "user", "content": user_input})
    _turn_prompt_tokens = _turn_completion_tokens = 0

    for _ in range(config.MAX_ITERATIONS):
        with Spinner("thinking"):
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
        _track_usage(response)
        msg = response.choices[0].message

        # store without <think>; may carry a tool request
        content = strip_think(msg.content)
        assistant_msg = {"role": "assistant", "content": content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        history.append(assistant_msg)

        if not msg.tool_calls:
            return content or "(couldn't produce a clean answer - try rephrasing or narrowing the request)"

        # Run each tool; a bad call is returned as text so the model can recover.
        for tc in msg.tool_calls:
            name = tc.function.name
            fn = TOOL_FUNCTIONS.get(name)
            try:
                result = fn(**json.loads(tc.function.arguments)) if fn else f"unknown tool: {name}"
            except Exception as exc:
                result = f"tool error: {exc}"
            print(f"  [tool] {name}({tc.function.arguments})")
            history.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "Reached the step limit (possible LOOP) - stopping."


def main() -> None:
    print("mimOE agent (solution 1, raw OpenAI SDK).")
    print("Commands: /exit to quit, /reset to clear memory.\n")
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
        print(usage_line(_last_prompt_tokens, _turn_prompt_tokens, _turn_completion_tokens,
                         config.MAX_CONTEXT_TOKENS) + "\n")


if __name__ == "__main__":
    main()

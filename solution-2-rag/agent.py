"""Solution 2 - a RAG agent over your personal documents (LangChain + Chroma + mimOE).

The framework path: local documents indexed into Chroma, a LangChain agent that
decides when to search them, all on-device. Contrast with solution 1's raw-SDK approach.

Uses `langchain.agents.create_agent` - the LangChain 1.x agent API (one line, no graph
to author; LangGraph is the runtime under the hood).
"""
import re

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config
from tools import TOOLS

# BYO Framework via LangChain's ChatOpenAI.
llm = ChatOpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model=config.MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
)

SYSTEM_PROMPT = (
    "You are a personal assistant for the user's home documents (contracts, policies). "
    "For ANY question about the user's documents, call search_documents and answer ONLY "
    "from what it returns, citing the source file and page. If the documents don't cover "
    "it, say so plainly. Use list_documents to see what's available and add_document to "
    "index a new PDF. Answer concisely in plain text. /no_think"
)

agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)

# backstop against a runaway tool loop
_INVOKE_CONFIG = {"recursion_limit": 2 * config.MAX_ITERATIONS}

_THINK = re.compile(r"<think>.*?</think>|<think>.*$|</?think>", re.DOTALL)


def clean(text: str) -> str:
    """Strip qwen3's <think> markup for display."""
    return _THINK.sub("", text or "").strip()


def main() -> None:
    print("Personal document agent (solution 2, RAG over your docs).")
    print("Commands: /exit to quit.\n")
    messages = []   # conversation memory for this session

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in ("/exit", "exit", "quit"):
            print("Bye!")
            break

        messages.append(HumanMessage(question))
        try:
            result = agent.invoke({"messages": messages}, _INVOKE_CONFIG)
            messages = result["messages"]   # keep the full turn as memory
            answer = clean(messages[-1].content) or "(no answer)"
        except Exception as exc:
            answer = f"(error: {exc})"
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()

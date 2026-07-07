"""Connection settings for the local mimOE endpoint.

Read from .env (if present), otherwise defaults for a local mimOE Studio.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# base_url is the heart of the "BYO Framework" approach: the OpenAI client points
# at local mimOE instead of api.openai.com. Only the address changes; the code
# stays "OpenAI-style".
BASE_URL = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-airouter/openai/v1")

# Locally the key is not really validated; it is a stub from the API button.
API_KEY = os.getenv("MIMOE_API_KEY", "1234")

# Model loaded in mimOE (must be tool-use capable and match what is Loaded there).
# qwen3-4b is smarter (more reliable tools/summarization) but slower than 1.7b.
MODEL = os.getenv("MIMOE_MODEL", "qwen3-4b")

# Low temperature -> predictability and reliable tool calling (not creativity).
TEMPERATURE = 0.2

# Cap on output length: bounds runaway generation (e.g. the model rambling on a hard
# or ambiguous prompt). Our answers are short, so this is generous.
MAX_TOKENS = 512

# Hard cap on agent-loop steps: a backstop against runaway loops (LOOP problem).
MAX_ITERATIONS = 5

# --- Level-2 memory: optional history compaction -----------------------------
# When the context window fills up, the old part of the dialogue is AUTOMATICALLY
# folded into a short summary. This is NOT RAG, just text summarization.
# OFF by default: a short demo fits the window; enable it to showcase.
COMPACT_ENABLED = False
MAX_CONTEXT_TOKENS = 12000    # qwen3 window size (from /models)
COMPACT_AT_FRACTION = 0.8     # primary trigger: compact when >=80% of the window is used
COMPACT_AFTER_MESSAGES = 40   # coarse fallback trigger by message count

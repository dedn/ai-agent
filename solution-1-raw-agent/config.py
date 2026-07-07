"""Connection settings for the local mimOE endpoint (read from .env, else defaults)."""
import os

from dotenv import load_dotenv

load_dotenv()

# BYO Framework: point the OpenAI client at local mimOE (only the URL changes).
BASE_URL = os.getenv("MIMOE_BASE_URL", "http://localhost:8083/mimik-airouter/openai/v1")
API_KEY = os.getenv("MIMOE_API_KEY", "1234")     # local stub, not validated
MODEL = os.getenv("MIMOE_MODEL", "qwen3-4b")     # must match the model Loaded in mimOE
TEMPERATURE = 0.2                                # predictability over creativity
MAX_TOKENS = 512                                 # bound runaway generation
MAX_ITERATIONS = 5                               # backstop against a runaway tool loop

# Level-2 memory: optional history compaction, OFF by default (a short demo fits the window).
COMPACT_ENABLED = False
MAX_CONTEXT_TOKENS = 12000
COMPACT_AT_FRACTION = 0.8      # compact when the window is >=80% full
COMPACT_AFTER_MESSAGES = 40

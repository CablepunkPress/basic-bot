import os

CONVERSATION_COLLECTION = "basic-bot-sessions"

# Sliding context window
WINDOW_FLOOR = int(os.environ.get("WINDOW_FLOOR", "20"))     # minimum messages in the sliding window
WINDOW_CEILING = int(os.environ.get("WINDOW_CEILING", "40"))  # fold triggers when messages reach this count

# Summarization
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "1000"))
SUMMARY_MIN_CHARS = int(os.environ.get("SUMMARY_MIN_CHARS", "40"))
SUMMARY_MODEL = "claude-haiku-4-5-20251001"

# Models
DEFAULT_MAX_TOKENS = 1024
THINKING_MAX_TOKENS = 16384
EXTENDED_BUDGET_TOKENS = 5000
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"

# Storage backend
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "firestore")

# Long-term memory (RAG) — embedding provider config
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "vertex")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_LOCATION = os.environ.get("EMBEDDING_LOCATION", "us-east5")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

# Tools
TOOL_BOX_ENABLED = os.getenv("TOOL_BOX_ENABLED", "true").lower() == "true"
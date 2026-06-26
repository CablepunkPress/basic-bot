import os

CONVERSATION_COLLECTION = "basic-bot-sessions"
MESSAGE_LIMIT = int(os.environ.get("MESSAGE_LIMIT", "20")) # floor (minimum N last messages loaded each turn)
SUMMARY_INTERVAL = int(os.environ.get("SUMMARY_INTERVAL", "20")) # MESSAGE_LIMIT + SUMMARY_INTERVAL = ceiling (at ceiling, SUMMARY_INTERVAL N folds for memory consolidation; at fold, sliding context window return to MESSAGE_LIMIT N) 
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "1000"))
SUMMARY_MIN_CHARS = int(os.environ.get("SUMMARY_MIN_CHARS", "40"))
DEFAULT_MAX_TOKENS = 1024
THINKING_MAX_TOKENS = 16384
EXTENDED_BUDGET_TOKENS = 5000
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"
SUMMARY_MODEL = "claude-haiku-4-5-20251001"

# Long-term memory (RAG) — embedding provider config
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "vertex")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_LOCATION = os.environ.get("EMBEDDING_LOCATION", "us-east5")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

# Tools
TOOL_BOX_ENABLED = os.getenv("TOOL_BOX_ENABLED", "true").lower() == "true"

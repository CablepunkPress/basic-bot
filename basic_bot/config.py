import os

# Sliding context window
WINDOW_FLOOR = int(os.environ.get("WINDOW_FLOOR", "20"))     # minimum messages in the sliding window
WINDOW_CEILING = int(os.environ.get("WINDOW_CEILING", "40"))  # fold triggers when messages reach this count

# Summarization
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "2000"))
SUMMARY_MIN_CHARS = int(os.environ.get("SUMMARY_MIN_CHARS", "40"))

# Local llama.cpp ports
EMBEDDING_PORT = int(os.environ.get("EMBEDDING_PORT", "11333"))
SUMMARY_PORT = int(os.environ.get("SUMMARY_PORT", "11444"))
CHAT_PORT = int(os.environ.get("CHAT_PORT", "11555"))

# Local llama.cpp URLs
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:11333")
SUMMARY_URL = os.environ.get("SUMMARY_URL", "http://localhost:11444")
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:11555")

# Storage backend — local SQLite by default; 
# cloud deploys set STORAGE_BACKEND=firestore
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "sqlite")

# LOCAL Long-term memory (RAG) — local llama-server by default; 
# cloud deploys set EMBEDDING_PROVIDER=vertex
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))

# CLOUD Long-term memory (RAG) — values for cloud deploys
# local llama-server by default; cloud deploys set EMBEDDING_PROVIDER=vertex
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_LOCATION = os.environ.get("EMBEDDING_LOCATION", "us-east5")

# History (UI loads last N messages at startup for user reference)
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "10"))

# Tools — Basic Bot has a tool belt; downstream bots have an additional tool box; auto-detects tool box tools when true
TOOL_BOX_ENABLED = os.getenv("TOOL_BOX_ENABLED", "true").lower() == "true"

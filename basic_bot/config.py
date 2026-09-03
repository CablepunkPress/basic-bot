import os

# Sliding context window
WINDOW_FLOOR = int(os.environ.get("WINDOW_FLOOR", "20")) # minimum messages in the sliding window
WINDOW_CEILING = int(os.environ.get("WINDOW_CEILING", "40")) # fold triggers when messages reach this count

# Summary validation
SUMMARY_MIN_CHARS = int(os.environ.get("SUMMARY_MIN_CHARS", "40"))

# Local llama.cpp ports
EMBEDDING_PORT = int(os.environ.get("EMBEDDING_PORT", "11333"))
SUMMARY_PORT = int(os.environ.get("SUMMARY_PORT", "11444"))
CHAT_PORT = int(os.environ.get("CHAT_PORT", "11555"))

# Local llama.cpp URLs
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:11333")
SUMMARY_URL = os.environ.get("SUMMARY_URL", "http://localhost:11444")
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:11555")

# History (UI loads last N messages at startup for user reference)
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "10"))

# Tools — agent ships with tool_belt; plugin tools are the tool_box; auto-detects plugin agent tools/ when true
TOOL_BOX_ENABLED = os.getenv("TOOL_BOX_ENABLED", "true").lower() == "true"

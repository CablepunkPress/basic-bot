"""Engine defaults.

Override these values in Bountiful agent's config.toml.

Consumers must use `import basic_bot.config as config` and
then `config.X`.
"""

# Sliding context window
WINDOW_FLOOR = 20
WINDOW_CEILING = 40

# Summary validation
SUMMARY_MIN_CHARS = 40

# Local llama.cpp ports
EMBEDDING_PORT = 11333
SUMMARY_PORT = 11444
CHAT_PORT = 11555

# Local llama.cpp URLs
EMBEDDING_URL = "http://localhost:11333"
SUMMARY_URL = "http://localhost:11444"
CHAT_URL = "http://localhost:11555"

# LocalProvider HTTP request timeout
REQUEST_TIMEOUT = 600

# Number of most similar past turn pairs from semantic search
RAG_RESULT_LIMIT = 5

# History (UI loads last N messages at startup)
HISTORY_LIMIT = 10

# Tools: agent ships with tool_belt; plugin tools are the tool_box; auto-detects plugin agent tools/ when true
TOOL_BOX_ENABLED = True


def apply_overrides(overrides: dict) -> None:
    """Override defaults from the agent's config.toml.

    Called once at startup by launch.py before anything else
    imports from this module. Keys in config.toml are matched
    case-insensitively to the uppercase constants above.
    """
    import basic_bot.config as _self
    for key, value in overrides.items():
        upper_key = key.upper()
        if hasattr(_self, upper_key):
            setattr(_self, upper_key, value)

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

# Local llama.cpp ports. This machine runs a llama-server on each, and
# the engine reaches them at http://localhost:{port}. To avoid a
# conflict with another program, change the port; the address follows.
EMBEDDING_PORT = 11333
SUMMARY_PORT = 11444
CHAT_PORT = 11555

# LocalProvider HTTP request timeout
REQUEST_TIMEOUT = 1200

# Number of most similar past turn pairs from semantic search
RAG_RESULT_LIMIT = 5

# History (UI loads last N messages at startup)
HISTORY_LIMIT = 10

# Tools: agent ships with tool_belt; plugin tools are the tool_box; auto-detects plugin agent tools/ when true
TOOL_BOX_ENABLED = True

# Log model reasoning/thinking content to the agent's log file
LOG_REASONING = False


def apply_overrides(overrides: dict) -> None:
    """Override defaults from the agent's config.toml.

    Called once by Session.open() before the engine builds the
    runtime. Keys in config.toml are matched case-insensitively to
    the uppercase constants above.
    """
    import basic_bot.config as _self
    for key, value in overrides.items():
        upper_key = key.upper()
        if hasattr(_self, upper_key):
            setattr(_self, upper_key, value)

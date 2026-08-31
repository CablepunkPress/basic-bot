"""Build Orchestrator.

Called by build.py in 'bountiful' repo.

Entry point for `python -m basic_bot.`.
"""

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.infrastructure.models import (
    download_chat_model,
    download_embedding_model,
    download_summary_model,
)


def main() -> None:
    print("  checking build tools")
    check_prerequisites()

    # TODO: detect hardware

    print("  llama.cpp (CPU)")
    build()

    print("  models")
    download_embedding_model()
    download_summary_model()
    download_chat_model()


if __name__ == "__main__":
    main()

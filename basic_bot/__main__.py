"""Entry point for `python -m basic_bot.`.

Called by build.py in bountiful repo.
"""

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.infrastructure.models import download_embedding_model


def main() -> None:
    print("  checking build tools")
    check_prerequisites()

    # TODO: detect hardware

    print("  llama.cpp (CPU)")
    build()

    print("  embedding model")
    download_embedding_model()
    # TODO: download_summary_model()
    # TODO: download_chat_model()


if __name__ == "__main__":
    main()

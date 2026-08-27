"""Entry point for `python -m basic_bot.infrastructure`."""

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.infrastructure.models import download_embedding_model


def main() -> None:
    print("  checking build tools")
    check_prerequisites()

    print("  llama.cpp (CPU)")
    build()

    print("  embedding model")
    download_embedding_model()


if __name__ == "__main__":
    main()
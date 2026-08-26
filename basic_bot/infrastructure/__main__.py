"""Entry point for `python -m basic_bot.infra`."""

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.infrastructure.models import download


def main() -> None:
    print("  checking build tools")
    check_prerequisites()

    print("  llama.cpp (CPU)")
    build()

    print(f"  embedding model")
    download()


if __name__ == "__main__":
    main()

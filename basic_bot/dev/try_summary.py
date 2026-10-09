"""Try the summary prompt on a real conversation, without folding.

Run from an agent's directory, with the agent's Python, while the agent
itself isn't running (the chat and summary models don't fit in memory
together):

    PYTHONPATH=~/myRepositories/basic-bot \\
      .venv/bin/python ~/myRepositories/basic-bot/dev/try_summary.py

Options:
    --runs N           how many summaries to write (default 1)
    --temperature T    override the profile's summary temperature

It summarizes exactly what the next fold would: the current summary plus
the next batch of messages after it. Each result is printed with its
length and timing. Nothing is saved.

PYTHONPATH makes Python load basic_bot from the repository instead of
the agent's installed copy, so edits to summary.py can be tried without
committing or reinstalling.
"""

import argparse
import logging
import socket
import sys
import time
import tomllib
from pathlib import Path

import basic_bot.config as config
from basic_bot.factory import create_runtime
from basic_bot.infrastructure.server import SUMMARY, ServerError, start, stop
from basic_bot.session import DEFAULT_USER
from basic_bot.summary import SUMMARY_MAX_PARAGRAPHS, summarize_batch


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _shape(text: str) -> tuple[int, int]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return len(text.split()), len(paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(
    description="Try the summary prompt on a real conversation, without folding.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    agent_path = Path.cwd()
    if not (agent_path / "dashboard.json").exists():
        sys.exit("Run this from an agent's directory, the folder with dashboard.json.")

    # The same overrides the agent itself would use
    config_file = agent_path / "config.toml"
    if config_file.exists():
        config.apply_overrides(tomllib.loads(config_file.read_text()))

    if _port_in_use(config.CHAT_PORT):
        sys.exit(
            "The chat server is running. Quit the agent first: the chat and "
            "summary models don't fit in memory together."
        )

    logging.basicConfig(level=logging.WARNING)
    runtime = create_runtime(agent_path)
    store = runtime.store

    # Exactly what the next fold would summarize
    state = store.get_state(DEFAULT_USER)
    boundary = state["summarized_through"]
    fold_size = config.WINDOW_CEILING - config.WINDOW_FLOOR
    chunk = store.get_messages_after(DEFAULT_USER, boundary)[:fold_size]
    if not chunk:
        sys.exit("There are no messages after the current summary to fold.")
    batch = [{"role": m["role"], "content": m["content"]} for m in chunk]

    sampling = dict(runtime.summary_sampling)
    if args.temperature is not None:
        sampling["temperature"] = args.temperature

    old_words, old_paragraphs = _shape(state["summary"])
    print(f"Current summary: through seq {boundary}, "
          f"{old_words} words in {old_paragraphs} paragraphs")
    print(f"Folding seq {chunk[0]['seq']}–{chunk[-1]['seq']} "
          f"({len(chunk)} messages), temperature {sampling.get('temperature')}")
    if len(chunk) < fold_size:
        print(f"(A real fold waits for {fold_size} messages; this uses {len(chunk)}.)")

    try:
        start(SUMMARY)
    except ServerError as e:
        sys.exit(f"ERROR: {e}")

    try:
        for run in range(1, args.runs + 1):
            started = time.monotonic()
            result = summarize_batch(
                runtime.summary_provider, state["summary"], batch, sampling,
            )
            seconds = time.monotonic() - started
            words, paragraphs = _shape(result)
            within = "within" if paragraphs <= SUMMARY_MAX_PARAGRAPHS else "OVER"

            print(f"\n=== Run {run}: {words} words, {paragraphs} paragraphs "
                  f"({within} the limit of {SUMMARY_MAX_PARAGRAPHS}), {seconds:.0f} s ===\n")
            print(result or "(empty: the summary was rejected as too short)")
    finally:
        stop(SUMMARY)


if __name__ == "__main__":
    main()

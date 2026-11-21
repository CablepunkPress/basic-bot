"""Backups of an agent's database.

A backup is made each time a session starts, once the chat server is
ready. It goes in the backups folder of the agent's data folder,
named with the UTC time it was made, such as
alice-backup-2026-11-20T223000Z.db, so the names sort by age. Only the
newest config.BACKUPS_KEPT are kept, and always at least one.

The copy is made with SQLite's own backup, which is consistent even
while the database is open. It's written under a temporary name and
renamed once it's complete, so a backup cut short is never kept as
one.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import basic_bot.config as config
from basic_bot.store import MessageStore

logger = logging.getLogger(__name__)

FOLDER = "backups"


def back_up(store: MessageStore, data_dir: Path, agent_id: str) -> None:
    """Back up the store, then remove the backups past the limit."""
    folder = data_dir / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = folder / f"{agent_id}-backup-{stamp}.db"
    partial = path.with_suffix(".partial")

    started = time.monotonic()
    store.backup(partial)
    partial.replace(path)
    elapsed = time.monotonic() - started

    removed = _prune(folder, agent_id)
    logger.info(
        "Backed up the database: %s, %.1f MB in %.1f s%s",
        path.name, path.stat().st_size / 1e6, elapsed,
        f"; removed {removed} older" if removed else "",
    )


def _prune(folder: Path, agent_id: str) -> int:
    """Remove unfinished backups and all but the newest kept, and count them."""
    for leftover in folder.glob(f"{agent_id}-backup-*.partial"):
        leftover.unlink()
    backups = sorted(folder.glob(f"{agent_id}-backup-*.db"))
    old = backups[:-max(1, config.BACKUPS_KEPT)]
    for path in old:
        path.unlink()
    return len(old)

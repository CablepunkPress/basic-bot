"""Factory for building bot runtimes.

create_runtime() assembles a BotRuntime from an agent directory and
package name. Web framework setup (FastAPI, Flask) is the caller's
responsibility — the factory builds the runtime, not the routes.
"""

import json
import logging
import os
from pathlib import Path

from basic_bot.config import STORAGE_BACKEND
from basic_bot.runtime import BotRuntime
from basic_bot.tools import build_registry

logger = logging.getLogger(__name__)


def _build_store(backend: str, collection: str):
    """Create the MessageStore for the configured backend.

    Imports are deferred so a local install never pulls in google.cloud
    and a cloud deploy never pulls in sqlite3 internals.
    """
    if backend == "firestore":
        from basic_bot.store_firestore import FirestoreMessageStore
        return FirestoreMessageStore(collection)

    if backend == "sqlite":
        from basic_bot.store_sqlite import SQLiteMessageStore
        from basic_bot.config import SQLITE_DIR
        db_path = os.path.join(SQLITE_DIR, f"{collection}.db")
        return SQLiteMessageStore(db_path)

    raise ValueError(f"Unknown storage backend: {backend!r}")


def create_runtime(agent_path: str | Path, package_name: str) -> BotRuntime:
    """Build a fully configured bot runtime.

    Args:
        agent_path: Directory containing persona.md and dashboard.json.
        package_name: Python package name for tool discovery. Transitional —
            will be replaced by filesystem-based tool loading from
            agent_path/tools/ in a future version.

    Returns:
        A BotRuntime with store, persona, tools, and identity configured.
    """
    agent_path = Path(agent_path)

    # Dashboard — agent identity
    dashboard_path = agent_path / "dashboard.json"
    dashboard = json.loads(dashboard_path.read_text())
    agent_id = dashboard["id"]

    logging.basicConfig(
        format=f"%(asctime)s - [{agent_id}] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    # Persona — user-authored, in the agent directory
    persona_path = agent_path / "persona.md"
    persona_text = persona_path.read_text().strip()

    # Capabilities — engine-owned, always from basic_bot package
    engine_dir = Path(__file__).parent
    capabilities_path = engine_dir / "instructions" / "capabilities.md"
    capabilities_text = capabilities_path.read_text().strip()

    persona = persona_text + "\n\n" + capabilities_text

    # Collection from agent ID
    collection = agent_id

    # Build storage backend
    store = _build_store(STORAGE_BACKEND, collection)

    # Build tool registry
    tool_registry = build_registry(package_name)

    return BotRuntime(
        agent_id=agent_id,
        agent_path=agent_path,
        package_name=package_name,
        store=store,
        persona=persona,
        tool_registry=tool_registry,
        dashboard=dashboard,
    )

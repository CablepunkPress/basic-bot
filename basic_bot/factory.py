"""Factory for building bot applications.

create_app() takes a package name and a collection/database scope and
returns a fully configured FastAPI application — routes, memory, tools,
and the fold lifecycle all wired up.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request

from basic_bot.chat import build_tool_registry, chat_with_claude
from basic_bot.config import (
    DEFAULT_MODEL,
    HISTORY_LIMIT,
    STORAGE_BACKEND,
    WINDOW_CEILING,
    WINDOW_FLOOR,
)
from basic_bot.fold import build_metadata, fold_rag, fold_summary, should_fold
from basic_bot.memory import get_messages
from basic_bot.models import MODELS
from basic_bot.runtime import BotRuntime

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


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(package_name: str, collection: str) -> FastAPI:
    """Build a fully configured FastAPI bot application.

    Args:
        package_name: Python package name (e.g., "basic_bot", "cablepunk").
            Locates persona, tools, and dashboard config.
        collection: Firestore collection or database scope for conversation data.
            Required with no default — prevents accidental cross-contamination.

    Tool belt (basic_bot.tool_belt) is always loaded. If the downstream bot has a
    tool_box package, its tools are auto-discovered. Disable with TOOL_BOX_ENABLED=false.
    """
    logging.basicConfig(
        format=f"%(asctime)s - [{package_name}] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    package = import_module(package_name)
    if package.__file__ is None:
        raise RuntimeError(f"Package '{package_name}' has no __file__ — cannot locate resources")
    package_dir = Path(package.__file__).parent

    # Load persona
    persona_path = package_dir / "instructions" / "persona.md"
    persona = persona_path.read_text().strip()

    # Build storage backend
    store = _build_store(STORAGE_BACKEND, collection)

    # Build tool registry
    tool_registry = build_tool_registry(package_name)

    runtime = BotRuntime(
        package_name=package_name,
        store=store,
        persona=persona,
        tool_registry=tool_registry,
    )

    # Load dashboard config
    dashboard_path = package_dir.parent / "dashboard.json"
    dashboard = json.loads(dashboard_path.read_text())
    agent_id = dashboard["id"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("=" * 50)
        logger.info("%s starting (collection: %s)", agent_id, collection)
        logger.info("Storage backend: %s", STORAGE_BACKEND)
        logger.info("Fold window: %d–%d messages", WINDOW_FLOOR, WINDOW_CEILING)
        logger.info("Default model: %s", DEFAULT_MODEL)
        logger.info(
            "Available models: %s",
            ", ".join(cfg["display_name"] for cfg in MODELS.values()),
        )
        logger.info("SDK: anthropic %s", anthropic.__version__)
        logger.info(
            "Tools: %d [%s]",
            len(tool_registry),
            ", ".join(tool_registry.keys()) or "none",
        )
        logger.info("History limit: %d messages", HISTORY_LIMIT)
        logger.info("=" * 50)

        if STORAGE_BACKEND == "firestore":
            try:
                from google.cloud import firestore
                db = firestore.Client()
                reg = dict(dashboard)
                reg["backendUrl"] = os.environ.get("BACKEND_URL", "")
                db.collection("agents").document(agent_id).set(reg)
                logger.info("Registered agent '%s' in Firestore", agent_id)
            except Exception:
                logger.exception("Failed to register agent — dashboard may be stale")

        yield
        logger.info("%s shutting down", agent_id)

    app = FastAPI(lifespan=lifespan)

    @app.post("/chat")
    async def chat(request: Request):
        try:
            data = await request.json()
            user_id = data.get("session_id")
            message = data.get("message", "").strip()
            model_id = data.get("model", DEFAULT_MODEL)
            effort = data.get("effort")
            thinking = data.get("thinking", False)

            if not message:
                raise HTTPException(status_code=400, detail="Message is required")
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id is required")

            logger.info(
                "Chat — user: %s, model: %s (%d chars)",
                user_id, model_id, len(message),
            )

            result = await chat_with_claude(
                runtime, user_id, message, model_id, effort, thinking,
            )

            metadata = build_metadata(result)
            seq = runtime.store.save_turn(
                user_id, message, result["reply"], metadata=metadata,
            )
            logger.info("Saved turn (seq %d–%d)", seq, seq + 1)

            # Fold lifecycle: check → RAG (sync) → summary (background)
            fold_state = should_fold(runtime.store, user_id)
            if fold_state:
                chunk = fold_rag(runtime.store, user_id, fold_state)
                if chunk:
                    asyncio.create_task(
                        asyncio.to_thread(
                            fold_summary,
                            runtime.store,
                            user_id,
                            fold_state["summary"],
                            chunk,
                        )
                    )
                    logger.info("Fold triggered for user %s — RAG done, summary in background", user_id)

            return {
                "response": result["reply"],
                "seq": seq,
                "model_used": result["model_used"],
                "display_name": result["display_name"],
                "effort": result["effort"],
                "thinking": result["thinking"],
                "fallback": result["fallback"],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Chat error: %s", str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/history")
    async def history(request: Request):
        """Return the last N messages with sequence numbers and metadata.

        Query params:
            session_id: required — identifies the conversation
            limit: optional — override HISTORY_LIMIT (capped at WINDOW_CEILING)
        """
        user_id = request.query_params.get("session_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        limit_param = request.query_params.get("limit")
        if limit_param:
            try:
                limit = min(int(limit_param), WINDOW_CEILING)
            except ValueError:
                limit = HISTORY_LIMIT
        else:
            limit = HISTORY_LIMIT

        messages = get_messages(runtime.store, user_id, limit)

        return {
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "seq": msg["seq"],
                    "metadata": msg.get("metadata"),
                }
                for msg in messages
            ],
            "count": len(messages),
        }

    @app.get("/models")
    async def models_endpoint():
        return {
            "models": {
                mid: {
                    "display_name": cfg["display_name"],
                    "effort_levels": cfg["effort_levels"],
                    "thinking_type": cfg["thinking_type"],
                }
                for mid, cfg in MODELS.items()
            },
            "default": DEFAULT_MODEL,
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": agent_id}

    @app.get("/")
    async def root():
        return {"agent": agent_id, "status": "running"}

    return app

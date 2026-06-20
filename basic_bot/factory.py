"""Factory for building bot applications.

create_app() takes a package name and a Firestore collection and returns
a fully configured FastAPI application — routes, memory, tools, and
dashboard registration all wired up.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request

from basic_bot.chat import build_tool_registry, chat_with_claude, maybe_summarize
from basic_bot.config import DEFAULT_MODEL
from basic_bot.memory import db, save_turn
from basic_bot.models import MODELS
from basic_bot.runtime import BotRuntime

logger = logging.getLogger(__name__)


def create_app(package_name: str, collection: str, tool_chest: str = "basic") -> FastAPI:
    """Build a fully configured FastAPI bot application.

    Args:
        package_name: Python package name (e.g., "basic_bot", "cablepunk").
            Locates persona, tools, and dashboard config.
        collection: Firestore collection for conversation data (messages, summaries, rag).
            Required with no default — prevents accidental cross-contamination.
        tool_chest: Built-in tools, or Built-in + Plugin tools (values: "basic" or "extended").
            Built-in tools are Basic Bot's "belt" folder; Plugin tools are downstream agents "tools" folder.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    package = import_module(package_name)
    if package.__file__ is None:
        raise RuntimeError(f"Package '{package_name}' has no __file__ — cannot locate resources")
    package_dir = Path(package.__file__).parent

    # Load persona
    persona_path = package_dir / "instructions" / "persona.md"
    persona = persona_path.read_text().strip()

    # Build tool registry
    tool_registry = build_tool_registry(package_name, tool_chest)

    runtime = BotRuntime(
        package_name=package_name,
        collection=collection,
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
        logger.info("=" * 50)

        try:
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

            seq = save_turn(user_id, message, result["reply"], collection)
            logger.info("Saved turn (seq %d–%d)", seq, seq + 1)

            if maybe_summarize(user_id, collection):
                logger.info("Summary updated for user %s", user_id)

            return {
                "response": result["reply"],
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
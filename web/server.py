"""Local Flask server for Basic Bot.

Single user, no auth, no proxy. Serves the chat UI and talks
directly to the bot's core logic. SQLite by default.
"""

import asyncio
import logging
import os
import threading
from importlib import import_module
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from basic_bot.chat import build_tool_registry, chat_with_claude
from basic_bot.config import (
    DEFAULT_MODEL,
    HISTORY_LIMIT,
    WINDOW_CEILING,
)
from basic_bot.fold import build_metadata, fold_rag, fold_summary, should_fold
from basic_bot.memory import get_messages
from basic_bot.models import MODELS
from basic_bot.runtime import BotRuntime

logger = logging.getLogger(__name__)

DEFAULT_USER = "local"


def create_local_app(
    package_name: str = "basic_bot",
    collection: str = "basic-bot-sessions",
) -> Flask:
    """Build a Flask app for local use.

    Args:
        package_name: Python package to load persona and tools from.
        collection: Used to derive the SQLite database filename.
    """
    logging.basicConfig(
        format=f"%(asctime)s - [{package_name}] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    # Load persona
    package = import_module(package_name)
    if package.__file__ is None:
        raise RuntimeError(f"Package '{package_name}' has no __file__")
    package_dir = Path(package.__file__).parent

    persona_path = package_dir / "instructions" / "persona.md"
    persona = persona_path.read_text().strip()

    # Build SQLite store
    sqlite_dir = os.environ.get(
        "SQLITE_DIR", os.path.expanduser("~/.basic-bot"),
    )
    db_path = os.path.join(sqlite_dir, f"{collection}.db")

    from basic_bot.store_sqlite import SQLiteMessageStore
    store = SQLiteMessageStore(db_path)

    # Build tool registry
    tool_registry = build_tool_registry(package_name)

    runtime = BotRuntime(
        package_name=package_name,
        store=store,
        persona=persona,
        tool_registry=tool_registry,
    )

    # Flask app with templates and static from web/
    web_dir = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(web_dir / "templates"),
        static_folder=str(web_dir / "static"),
    )

    logger.info("=" * 50)
    logger.info("%s starting locally (collection: %s)", package_name, collection)
    logger.info("Storage: SQLite (%s)", db_path)
    logger.info("Default model: %s", DEFAULT_MODEL)
    logger.info(
        "Tools: %d [%s]",
        len(tool_registry),
        ", ".join(tool_registry.keys()) or "none",
    )
    logger.info("=" * 50)

    # --- Routes ---

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400

        model_id = data.get("model", DEFAULT_MODEL)
        effort = data.get("effort")
        thinking = data.get("thinking", False)
        user_id = DEFAULT_USER

        logger.info(
            "Chat — model: %s (%d chars)", model_id, len(message),
        )

        try:
            result = asyncio.run(
                chat_with_claude(
                    runtime, user_id, message, model_id, effort, thinking,
                )
            )
        except Exception as e:
            logger.exception("Chat error: %s", str(e))
            return jsonify({"error": "Internal server error"}), 500

        metadata = build_metadata(result)
        seq = runtime.store.save_turn(
            user_id, message, result["reply"], metadata=metadata,
        )
        logger.info("Saved turn (seq %d–%d)", seq, seq + 1)

        # Fold lifecycle
        fold_state = should_fold(runtime.store, user_id)
        if fold_state:
            chunk = fold_rag(runtime.store, user_id, fold_state)
            if chunk:
                thread = threading.Thread(
                    target=fold_summary,
                    args=(runtime.store, user_id, fold_state["summary"], chunk),
                    daemon=True,
                )
                thread.start()
                logger.info(
                    "Fold triggered — RAG done, summary in background",
                )

        return jsonify({
            "response": result["reply"],
            "seq": seq,
            "model_used": result["model_used"],
            "display_name": result["display_name"],
            "effort": result["effort"],
            "thinking": result["thinking"],
            "fallback": result["fallback"],
        })

    @app.route("/history", methods=["GET"])
    def history():
        user_id = DEFAULT_USER

        limit_param = request.args.get("limit")
        if limit_param:
            try:
                limit = min(int(limit_param), WINDOW_CEILING)
            except ValueError:
                limit = HISTORY_LIMIT
        else:
            limit = HISTORY_LIMIT

        messages = get_messages(runtime.store, user_id, limit)

        return jsonify({
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
        })

    @app.route("/models", methods=["GET"])
    def models_endpoint():
        return jsonify({
            "models": {
                mid: {
                    "display_name": cfg["display_name"],
                    "effort_levels": cfg["effort_levels"],
                    "thinking_type": cfg["thinking_type"],
                }
                for mid, cfg in MODELS.items()
            },
            "default": DEFAULT_MODEL,
        })

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})

    return app

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request, HTTPException

from basic_bot.config import DEFAULT_MODEL
from basic_bot.chat import chat_with_claude
from basic_bot.memory import db, save_turn
from basic_bot.models import MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Startup
    logger.info("=" * 50)
    logger.info("basic-bot starting")
    logger.info("Default model: %s", DEFAULT_MODEL)
    logger.info(
        "Available models: %s",
        ", ".join(cfg["display_name"] for cfg in MODELS.values())
    )
    logger.info("SDK: anthropic %s", anthropic.__version__)
    logger.info("Memory: Firestore")

    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard.json"
    try:
        dashboard = json.loads(dashboard_path.read_text())
        dashboard["backendUrl"] = os.environ.get("BACKEND_URL", "")
        agent_id = dashboard["id"]
        db.collection("agents").document(agent_id).set(dashboard)
        logger.info("Registered agent '%s' in Firestore", agent_id)
    except Exception:
        logger.exception("Failed to register agent — dashboard may be stale")

    logger.info("=" * 50)

    yield

    # Shutdown
    logger.info("basic-bot shutting down")


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_id = data.get("session_id")
        message = data.get("message", "").strip()
        agent_id = data.get("agent_id", "basic-bot")
        model_id = data.get("model", DEFAULT_MODEL)
        effort = data.get("effort")
        thinking = data.get("thinking", False)

        logger.info(
            "Received chat request - agent_id: %s, user_id: %s, model: %s",
            agent_id, user_id, model_id
        )

        if not message:
            logger.warning("Empty message received")
            raise HTTPException(status_code=400, detail="Message is required")

        if not user_id:
            logger.warning("No user_id provided")
            raise HTTPException(status_code=400, detail="user_id is required")

        logger.info("Message from user %s (%d chars)", user_id, len(message))

        result = await chat_with_claude(
            user_id, message, model_id, effort, thinking
        )

        seq = save_turn(user_id, message, result["reply"])
        logger.info("Saved turn for user %s (seq %d-%d)", user_id, seq, seq + 1)

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
        logger.exception("Chat endpoint error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def models():
    return {
        "models": {
            model_id: {
                "display_name": config["display_name"],
                "effort_levels": config["effort_levels"],
                "thinking_type": config["thinking_type"],
            }
            for model_id, config in MODELS.items()
        },
        "default": DEFAULT_MODEL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "basic-bot"}


@app.get("/")
async def root():
    return {"message": "Basic Bot API", "status": "running"}
import logging
from typing import cast

import anthropic
from anthropic.types import MessageParam, TextBlock
from fastapi import FastAPI, Request, HTTPException
from google.cloud import firestore

# ----------------------
# Model config
# ----------------------
MODELS = {
    "claude-haiku-4-5-20251001": {
        "display_name": "Haiku 4.5",
        "effort_levels": None,
        "thinking_type": "extended",
    },
    "claude-sonnet-4-6": {
        "display_name": "Sonnet 4.6",
        "effort_levels": ["low", "medium", "high", "max"],
        "thinking_type": "adaptive",
    },
    "claude-opus-4-6": {
        "display_name": "Opus 4.6",
        "effort_levels": ["low", "medium", "high", "max"],
        "thinking_type": "adaptive",
    },
    "claude-opus-4-7": {
        "display_name": "Opus 4.7",
        "effort_levels": ["low", "medium", "high", "xhigh", "max"],
        "thinking_type": "adaptive",
    },
    "claude-opus-4-8": {
        "display_name": "Opus 4.8",
        "effort_levels": ["low", "medium", "high", "xhigh", "max"],
        "thinking_type": "adaptive",
    },
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"

# ----------------------
# App config
# ----------------------
CONVERSATION_COLLECTION = "basic-bot-sessions"
MESSAGE_LIMIT = 20
DEFAULT_MAX_TOKENS = 1024
THINKING_MAX_TOKENS = 8192
EXTENDED_BUDGET_TOKENS = 5000

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------
# Initialize clients
# ----------------------
# Reads ANTHROPIC_API_KEY from the environment automatically.
client = anthropic.Anthropic()

# Reads project from Cloud Run metadata / local ADC.
db = firestore.Client()


# ----------------------
# Conversation memory
# ----------------------
def save_message(user_id: str, role: str, content: str):
    """Save a message to the user's conversation in Firestore."""
    db.collection(CONVERSATION_COLLECTION).document(user_id)\
        .collection("messages").add({
            "role": role,
            "content": content,
            "created_at": firestore.SERVER_TIMESTAMP,
        })


def get_messages(user_id: str, limit: int = MESSAGE_LIMIT) -> list[dict[str, str]]:
    """Get recent messages for a user, in chronological order."""
    messages_ref = (
        db.collection(CONVERSATION_COLLECTION)
        .document(user_id)
        .collection("messages")
    )
    query = messages_ref.order_by(
        "created_at", direction=firestore.Query.DESCENDING
    ).limit(limit)

    docs = list(query.stream())
    docs.reverse()

    messages = []
    for doc in docs:
        data = doc.to_dict()
        messages.append({"role": data["role"], "content": data["content"]})
    return messages


# ----------------------
# Helpers
# ----------------------
def build_system_prompt(model_id: str) -> str:
    """Build the system prompt with memory and model awareness."""
    display_name = MODELS.get(model_id, MODELS[DEFAULT_MODEL])["display_name"]
    return (
        "You are Basic Bot, a helpful and friendly general-purpose AI assistant. "
        "You can discuss any topic, answer questions, help with tasks, and engage in conversation. "
        "Be concise, clear, and helpful. Keep responses conversational and friendly. "
        "You have conversation memory and can remember what users have told you "
        "earlier in the conversation. "
        f"You are currently running on {display_name}."
    )


def build_api_kwargs(
    model_id: str,
    system_prompt: str,
    messages: list[MessageParam],
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Build kwargs for client.messages.create based on model capabilities."""
    model_config = MODELS[model_id]

    kwargs: dict = {
        "model": model_id,
        "system": system_prompt,
        "messages": messages,
    }

    # Effort — only for models that support it, only if the level is valid
    if effort and model_config["effort_levels"]:
        if effort in model_config["effort_levels"]:
            kwargs["output_config"] = {"effort": effort}
        else:
            logger.warning(
                "Effort '%s' not supported by %s, ignoring",
                effort, model_config["display_name"]
            )

    # Thinking — two regimes depending on model generation
    if thinking and model_config["thinking_type"] == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["max_tokens"] = THINKING_MAX_TOKENS
    elif thinking and model_config["thinking_type"] == "extended":
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": EXTENDED_BUDGET_TOKENS,
        }
        kwargs["max_tokens"] = THINKING_MAX_TOKENS
    else:
        kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

    return kwargs


def extract_reply(response) -> str:  # type: ignore[no-untyped-def]
    """Extract the text reply from a Claude response, skipping any thinking blocks."""
    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text
    raise ValueError("No text block found in response")


# ----------------------
# Chat with Claude
# ----------------------
async def chat_with_claude(
    user_id: str,
    user_message: str,
    model_id: str = DEFAULT_MODEL,
    effort: str | None = None,
    thinking: bool = False,
) -> dict:
    """Chat with Claude using conversation memory. Returns reply and metadata."""

    logger.info("Processing chat request for user: %s", user_id)

    # Get conversation history from Firestore
    memory = get_messages(user_id)
    logger.info("Retrieved %d messages from history", len(memory))

    # Build messages list
    messages: list[MessageParam] = [
        cast(MessageParam, {"role": msg["role"], "content": msg["content"]})
        for msg in memory
    ]
    messages.append({"role": "user", "content": user_message})

    # Validate model, fall back to default if unknown
    if model_id not in MODELS:
        logger.warning("Unknown model '%s', using default", model_id)
        model_id = DEFAULT_MODEL

    model_config = MODELS[model_id]
    system_prompt = build_system_prompt(model_id)

    logger.info(
        "Sending request to %s (effort=%s, thinking=%s) with %d messages",
        model_config["display_name"], effort, thinking, len(messages)
    )

    # Try the requested model
    try:
        kwargs = build_api_kwargs(model_id, system_prompt, messages, effort, thinking)
        response = client.messages.create(**kwargs)
        reply = extract_reply(response)

        logger.info("Received response from %s (%d chars)",
                     model_config["display_name"], len(reply))

        return {
            "reply": reply,
            "model_used": model_id,
            "display_name": model_config["display_name"],
            "effort": effort,
            "thinking": thinking,
            "fallback": False,
        }

    except Exception:
        logger.exception(
            "Error with %s, attempting fallback to %s",
            model_config["display_name"], MODELS[FALLBACK_MODEL]["display_name"]
        )

        # If we're already on the fallback model in bare mode, nothing left to try
        if model_id == FALLBACK_MODEL and not effort and not thinking:
            raise

        # Fallback: safest possible call — no effort, no thinking
        try:
            fallback_prompt = build_system_prompt(FALLBACK_MODEL)
            fallback_kwargs = build_api_kwargs(
                FALLBACK_MODEL, fallback_prompt, messages
            )
            response = client.messages.create(**fallback_kwargs)
            reply = extract_reply(response)

            logger.info(
                "Fallback to %s succeeded (%d chars)",
                MODELS[FALLBACK_MODEL]["display_name"], len(reply)
            )

            return {
                "reply": reply,
                "model_used": FALLBACK_MODEL,
                "display_name": MODELS[FALLBACK_MODEL]["display_name"],
                "effort": None,
                "thinking": False,
                "fallback": True,
            }

        except Exception:
            logger.exception("Fallback also failed")
            raise


# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 50)
    logger.info("basic-bot starting")
    logger.info("Default model: %s", DEFAULT_MODEL)
    logger.info(
        "Available models: %s",
        ", ".join(cfg["display_name"] for cfg in MODELS.values())
    )
    logger.info("SDK: anthropic %s", anthropic.__version__)
    logger.info("Memory: Firestore (%s)", CONVERSATION_COLLECTION)
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("basic-bot shutting down")


# ----------------------
# Routes
# ----------------------
@app.post("/chat")
async def chat(request: Request):
    """Handle chat requests from dashboard proxy."""
    try:
        data = await request.json()
        user_id = data.get("session_id")        # Opaque user ID from proxy
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

        logger.info("Message from user %s: %s", user_id, message[:50])

        # Get AI response with model selection and fallback
        result = await chat_with_claude(
            user_id, message, model_id, effort, thinking
        )

        # Save both turns to Firestore
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", result["reply"])
        logger.info("Saved conversation turn for user: %s", user_id)

        # Response includes metadata for the frontend log
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
    """Return available models and their capabilities for the frontend."""
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
    """Health check endpoint."""
    return {"status": "healthy", "agent": "basic-bot"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Basic Bot API", "status": "running"}
import logging
from typing import cast

import anthropic
from anthropic.types import MessageParam, TextBlock
from fastapi import FastAPI, Request, HTTPException
from google.cloud import firestore

# ----------------------
# Config
# ----------------------
MODEL_NAME = "claude-haiku-4-5-20251001"
CONVERSATION_COLLECTION = "basic-bot-sessions"
MESSAGE_LIMIT = 20

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
# Chat with Claude
# ----------------------
async def chat_with_claude(user_id: str, user_message: str) -> str:
    """Chat with Claude using conversation memory."""

    logger.info("Processing chat request for user: %s", user_id)

    # Get conversation history from Firestore
    memory = get_messages(user_id)
    logger.info("Retrieved %d messages from history", len(memory))

    # System prompt is a top-level parameter in the Messages API, not a message.
    system_instruction = (
        "You are Basic Bot, a helpful and friendly general-purpose AI assistant. "
        "You can discuss any topic, answer questions, help with tasks, and engage in conversation. "
        "Be concise, clear, and helpful. Keep responses conversational and friendly."
    )

    # Build the messages list from history. Stored keys are already
    # "role" and "content", matching the API format directly.
    messages: list[MessageParam] = [
        cast(MessageParam, {"role": msg["role"], "content": msg["content"]})
        for msg in memory
    ]

    # Add the current user turn.
    messages.append({"role": "user", "content": user_message})

    logger.info("Sending request to Claude with %d messages in context", len(messages))

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system=system_instruction,
            messages=messages,
        )

        # No tools and no thinking, so the first content block is the text reply.
        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise ValueError(f"Expected text response, got {type(block).__name__}")
        reply = block.text

        logger.info("Received response from Claude (%d chars)", len(reply))
        return reply

    except Exception:
        logger.exception("Anthropic API error")
        raise


# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

# ----------------------
# Startup/Shutdown Events
# ----------------------
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 50)
    logger.info("basic-bot starting")
    logger.info("Model: %s", MODEL_NAME)
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
        user_id = data.get("session_id")  # Opaque user ID from proxy
        message = data.get("message", "").strip()
        agent_id = data.get("agent_id", "basic-bot")

        logger.info("Received chat request - agent_id: %s, user_id: %s",
                     agent_id, user_id)

        if not message:
            logger.warning("Empty message received")
            raise HTTPException(status_code=400, detail="Message is required")

        if not user_id:
            logger.warning("No user_id provided")
            raise HTTPException(status_code=400, detail="user_id is required")

        logger.info("Message from user %s: %s", user_id, message[:50])

        # Get AI response (loads history from Firestore internally)
        logger.info("Calling Claude for user: %s", user_id)
        reply = await chat_with_claude(user_id, message)

        # Save both turns to Firestore
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", reply)
        logger.info("Saved conversation turn for user: %s", user_id)

        return {"response": reply}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "basic-bot"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Basic Bot API", "status": "running"}
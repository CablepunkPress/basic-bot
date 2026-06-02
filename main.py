from fastapi import FastAPI, Request, HTTPException
import sqlite3
import secrets
import logging
import anthropic

# ----------------------
# Config
# ----------------------
MODEL_NAME = "claude-haiku-4-5-20251001"

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------
# Initialize Anthropic client
# ----------------------
# Reads ANTHROPIC_API_KEY from the environment automatically.
client = anthropic.Anthropic()

# ----------------------
# Database setup (short-term memory)
# ----------------------
DB_PATH = "/tmp/sessions.db"

def init_db():
    """Initialize SQLite database for session storage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)

# ----------------------
# Session management
# ----------------------
class SessionManager:
    """Manages conversation sessions and message history."""
    
    @staticmethod
    def create_session(user_id: str):
        """Create a new session for a user."""
        session_id = secrets.token_urlsafe(16)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
            (session_id, user_id)
        )
        conn.commit()
        conn.close()
        logger.info("Created new session for user: %s", user_id)
        return session_id
    
    @staticmethod
    def session_exists(session_id: str) -> bool:
        """Check if session exists."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    @staticmethod
    def get_user_for_session(session_id: str) -> str:
        """Get user_id for a session."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    @staticmethod
    def save_message(session_id: str, role: str, content: str):
        """Save a message to the database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_messages(session_id: str, limit: int = 20) -> list:
        """Get recent messages for a session."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT role, content FROM messages
               WHERE session_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row[0], "text": row[1]} for row in reversed(rows)]

session_manager = SessionManager()

# ----------------------
# Chat with Claude
# ----------------------
async def chat_with_claude(session_id: str, user_message: str) -> str:
    """Chat with Claude using session memory."""

    logger.info("Processing chat request for session: %s", session_id)

    # Get conversation history
    memory = session_manager.get_messages(session_id, limit=20)
    logger.info("Retrieved %d messages from history", len(memory))

    # System prompt is a top-level parameter in the Messages API, not a message.
    system_instruction = (
        "You are Basic Bot, a helpful and friendly general-purpose AI assistant. "
        "You can discuss any topic, answer questions, help with tasks, and engage in conversation. "
        "Be concise, clear, and helpful. Keep responses conversational and friendly."
    )

    # Stored roles are already "user"/"assistant" — exactly what the API wants,
    # so no remapping like the Gemini "model" role we used before.
    messages = [
        {"role": msg["role"], "content": msg["text"]}
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
        reply = response.content[0].text

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
    """Initialize database on startup."""
    init_db()
    logger.info("=" * 50)
    logger.info("basic-bot starting")
    logger.info("Model: %s", MODEL_NAME)
    logger.info("SDK: anthropic %s", anthropic.__version__)
    logger.info("Database: %s (ephemeral)", DB_PATH)
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
        session_id = data.get("session_id")  # This is user_id from proxy
        message = data.get("message", "").strip()
        agent_id = data.get("agent_id", "basic-bot")
        
        logger.info("Received chat request - agent_id: %s, session_id: %s", agent_id, session_id)
        
        if not message:
            logger.warning("Empty message received")
            raise HTTPException(status_code=400, detail="Message is required")
        
        if not session_id:
            logger.warning("No session_id provided")
            raise HTTPException(status_code=400, detail="session_id (user_id) is required")
        
        logger.info("Message from user %s: %s", session_id, message[:50])
        
        # Check if session exists, create if not
        if not session_manager.session_exists(session_id):
            session_id = session_manager.create_session(session_id)
            logger.info("Created new session for user: %s", session_id)
        else:
            logger.info("Continuing existing session: %s", session_id)
        
        # Get AI response
        logger.info("Calling Claude for session: %s", session_id)
        reply = await chat_with_claude(session_id, message)
        
        # Save messages
        session_manager.save_message(session_id, "user", message)
        logger.info("Saved user message to database")
        
        session_manager.save_message(session_id, "assistant", reply)
        logger.info("Saved assistant message to database")
        
        logger.info("Request completed successfully for user: %s", session_id)
        
        return {
            "response": reply
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy", "agent": "basic-bot"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Basic Bot API", "status": "running"}
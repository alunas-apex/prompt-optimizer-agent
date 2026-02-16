"""Telegram Bot Server for the Prompt Optimizer Agent.

Runs a FastAPI application that receives Telegram webhook updates and
dispatches them to the handlers defined in handlers.py.

Usage:
    # Set TELEGRAM_BOT_TOKEN in .env, then:
    uvicorn telegram-bot.bot_server:app --host 0.0.0.0 --port 8443

    # Register the webhook with Telegram:
    curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://<YOUR_DOMAIN>/webhook"
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from handlers import (
    analyze_command,
    help_command,
    optimize_command,
    plain_text_handler,
    start_command,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(_PROJECT_ROOT / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_PATH = "/webhook"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set — the bot will not function.")


# ---------------------------------------------------------------------------
# Telegram Application (python-telegram-bot)
# ---------------------------------------------------------------------------

def _build_telegram_app() -> Application:
    """Build and configure the python-telegram-bot Application."""
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .updater(None)       # We handle updates via webhook, not polling
        .build()
    )

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("optimize", optimize_command))

    # Plain text fallback — treat any non-command text as a prompt to optimize
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text_handler))

    return app


telegram_app = _build_telegram_app()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and shut down the Telegram application."""
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Telegram bot started (webhook mode)")
    yield
    await telegram_app.stop()
    await telegram_app.shutdown()
    logger.info("Telegram bot stopped")


app = FastAPI(
    title="Prompt Optimizer Telegram Bot",
    description="Webhook receiver for the Prompt Optimizer Telegram bot",
    lifespan=lifespan,
)


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request) -> Response:
    """Receive an update from Telegram and dispatch it."""
    data = await request.json()
    update = Update.de_json(data=data, bot=telegram_app.bot)
    await telegram_app.process_update(update)
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "bot_configured": bool(BOT_TOKEN)}


# ---------------------------------------------------------------------------
# Entry point (for direct execution / development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BOT_PORT", "8443"))
    uvicorn.run(app, host="0.0.0.0", port=port)

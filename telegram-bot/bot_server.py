"""Telegram Bot Server for the Prompt Optimizer Agent.

Supports two modes:
  - **Polling mode** (default): The bot long-polls the Telegram API for
    updates.  No public URL required — ideal for development and environments
    without a stable domain.
  - **Webhook mode**: Runs a FastAPI server that receives Telegram webhook
    POSTs.  Activate by setting WEBHOOK_URL in the environment.

Usage:
    # Polling mode (default):
    python -m telegram-bot.bot_server

    # Webhook mode:
    WEBHOOK_URL=https://yourdomain.com/webhook python -m telegram-bot.bot_server
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

# Ensure the project root and bot directory are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BOT_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _BOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # empty → polling mode
WEBHOOK_PATH = "/webhook"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set — the bot will not function.")


# ---------------------------------------------------------------------------
# Shared: register handlers on an Application builder
# ---------------------------------------------------------------------------

def _register_handlers(app: Application) -> None:
    """Attach all command and message handlers to *app*."""
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("optimize", optimize_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text_handler))


# ---------------------------------------------------------------------------
# Polling mode
# ---------------------------------------------------------------------------

def run_polling() -> None:
    """Start the bot in long-polling mode (no public URL needed)."""
    logger.info("Starting bot in POLLING mode …")

    app = Application.builder().token(BOT_TOKEN).build()
    _register_handlers(app)

    # Delete any leftover webhook so polling works
    async def _clear_webhook(app: Application) -> None:
        await app.bot.delete_webhook(drop_pending_updates=True)
        me = await app.bot.get_me()
        logger.info("Bot identity: @%s (id=%s)", me.username, me.id)

    app.post_init = _clear_webhook

    app.run_polling(drop_pending_updates=True)


# ---------------------------------------------------------------------------
# Webhook mode (FastAPI)
# ---------------------------------------------------------------------------

def run_webhook() -> None:
    """Start the bot in webhook mode behind a FastAPI server."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Request, Response
    from telegram import Update

    logger.info("Starting bot in WEBHOOK mode (url=%s) …", WEBHOOK_URL)

    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .updater(None)
        .build()
    )
    _register_handlers(telegram_app)

    _bot_initialized = False

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _bot_initialized
        try:
            await telegram_app.initialize()
            await telegram_app.start()
            await telegram_app.bot.set_webhook(url=WEBHOOK_URL + WEBHOOK_PATH)
            _bot_initialized = True
            logger.info("Telegram bot started (webhook mode)")
        except Exception:
            logger.warning(
                "Could not fully initialize Telegram bot (API may be unreachable). "
                "The server will start anyway.",
                exc_info=True,
            )
        yield
        if _bot_initialized:
            await telegram_app.bot.delete_webhook()
            await telegram_app.stop()
            await telegram_app.shutdown()
            logger.info("Telegram bot stopped")

    fastapi_app = FastAPI(
        title="Prompt Optimizer Telegram Bot",
        description="Webhook receiver for the Prompt Optimizer Telegram bot",
        lifespan=lifespan,
    )

    @fastapi_app.post(WEBHOOK_PATH)
    async def telegram_webhook(request: Request) -> Response:
        nonlocal _bot_initialized
        if not _bot_initialized:
            try:
                await telegram_app.initialize()
                await telegram_app.start()
                _bot_initialized = True
            except Exception:
                logger.error("Cannot initialize Telegram bot", exc_info=True)
                return Response(status_code=503, content="Bot not initialized")
        data = await request.json()
        update = Update.de_json(data=data, bot=telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)

    @fastapi_app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "bot_configured": bool(BOT_TOKEN),
            "bot_initialized": _bot_initialized,
        }

    import uvicorn

    port = int(os.getenv("BOT_PORT", "8443"))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not BOT_TOKEN:
        logger.error("Cannot start: TELEGRAM_BOT_TOKEN is not set in .env")
        sys.exit(1)

    if WEBHOOK_URL:
        run_webhook()
    else:
        run_polling()

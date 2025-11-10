from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
import logging
import asyncio

logger = logging.getLogger(__name__)


def make_bot_lifespan(token: Optional[str]):
    """Return an asynccontextmanager suitable for FastAPI `lifespan` parameter.

    This function tries to import `python-telegram-bot` lazily. If the
    package is not installed, a no-op lifespan is returned and a warning
    is logged.
    """

    try:
        # Local import so the app can run even if the library isn't installed
        from telegram import Update
        from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

        async def _start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            logger.info("Received /start command from %s", getattr(update.effective_user, 'username', None))
            print(f"[bot] /start from {getattr(update.effective_user, 'id', None)}")
            if update.message:
                await update.message.reply_text("Ciao dal bot!")

        async def _message_logger(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Log incoming messages (text and non-text)."""
            # quick visibility: always print to stdout in addition to logger
            try:
                print(f"[bot] got update: {update}")
            except Exception:
                print("[bot] got update (unprintable)")
            logger.info("_message_logger invoked for update: %s", update)
            try:
                msg = update.effective_message
                user = update.effective_user
                if msg is None:
                    logger.info("Received update without message: %s", update)
                    return

                # Prefer .text for textual messages
                text = getattr(msg, 'text', None)
                if text:
                    logger.info("Telegram message from %s (%s): %s", getattr(user, 'username', None) or getattr(user, 'full_name', None), getattr(user, 'id', None), text)
                else:
                    # Non-text message (photo, sticker, etc.)
                    logger.info("Telegram non-text message from %s (%s): %s", getattr(user, 'username', None) or getattr(user, 'full_name', None), getattr(user, 'id', None), msg)
            except Exception:
                logger.exception("Error while logging incoming telegram message")
    except Exception as e:  # ImportError or other issues
        logger.warning("python-telegram-bot not available: %s", e)

        @asynccontextmanager
        async def _noop_lifespan(app: FastAPI):
            # Bot not available; just yield and do nothing
            yield

        return _noop_lifespan

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not token:
            # No token provided: skip starting the bot
            logger.info("TELEGRAM_BOT_TOKEN not set: Telegram bot disabled")
            yield
            return
        app.state.bot_app = ApplicationBuilder().token(token).build()
        # Register handlers: /start and message logger
        app.state.bot_app.add_handler(CommandHandler("start", _start_handler))
        # /get_over_match: send grouped over-1.5 matches for today
        async def _get_over_match(update, context):
            # lazy import the service to avoid startup issues
            try:
                from app.services.over_match_service import get_today_over_groups
            except Exception as e:
                logger.exception("Failed to import over_match_service: %s", e)
                await update.message.reply_text("Errore interno: servizio non disponibile")
                return

            groups = get_today_over_groups(min_confidence=0.8)
            if not groups:
                await update.message.reply_text("Nessun match Over 1.5 >=80% trovato per oggi.")
                return

            # send one message per group (shifted time)
            for shifted_time, matches in groups.items():
                lines = [f"Match time (shifted +50m): {shifted_time}"]
                for m in matches:
                    lines.append(f"- {m}")
                text = "\n".join(lines)
                # prefer replying in chat
                chat_id = update.effective_chat.id if update.effective_chat else None
                if chat_id:
                    await context.bot.send_message(chat_id=chat_id, text=text)
                else:
                    await update.message.reply_text(text)

        app.state.bot_app.add_handler(CommandHandler("get_over_match", _get_over_match))
        # Log all text messages that are not commands
        app.state.bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _message_logger))
        # Also log non-text updates (photos, stickers, etc.) via a broad filter
        app.state.bot_app.add_handler(MessageHandler(~filters.TEXT, _message_logger))
        # Catch-all: ensure we log anything missed by the above (helps debugging)
        try:
            app.state.bot_app.add_handler(MessageHandler(filters.ALL, _message_logger))
        except Exception:
            # Older/newer versions of the filters API may not have ALL; ignore if absent
            logger.debug("filters.ALL not available; skip catch-all handler")

        logger.info("Registered Telegram handlers: start, message logger(s)")
        # Initialize and start the bot application (polling)
        await app.state.bot_app.initialize()
        await app.state.bot_app.start()
        # Try to explicitly start polling if an updater is available (compat shim)
        try:
            updater = getattr(app.state.bot_app, 'updater', None)
            if updater is not None:
                start_polling = getattr(updater, 'start_polling', None)
                if callable(start_polling):
                    maybe_coro = start_polling()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                    logger.info("Started updater polling via updater.start_polling()")
        except Exception:
            logger.debug("No explicit updater.start_polling() performed (not available or failed)", exc_info=True)
        logger.info("Bot Telegram avviato (polling started)")

        try:
            yield
        finally:
            # Graceful shutdown
            await app.state.bot_app.stop()
            await app.state.bot_app.shutdown()
            logger.info("Bot Telegram spento")

    return lifespan

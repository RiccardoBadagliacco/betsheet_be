"""Notifier for Over 1.5 matches grouped by end-of-first-half time.

This module prefers to send messages using the in-memory Application created
in `app.main` (if available). If not available, it falls back to the HTTP API
using requests executed in a thread to avoid blocking the event loop.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import json
from datetime import datetime, timedelta
import logging
import asyncio
import requests
from app.core.settings import settings

logger = logging.getLogger(__name__)


def _parse_match_time(mt_str: str) -> datetime | None:
    if not mt_str:
        return None
    try:
        # Try ISO datetime first
        return datetime.fromisoformat(mt_str)
    except Exception:
        # Try parsing time only HH:MM[:SS]
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                t = datetime.strptime(mt_str, fmt).time()
                # attach today's date (we only care about hour/minute)
                now = datetime.now()
                return datetime(year=now.year, month=now.month, day=now.day, hour=t.hour, minute=t.minute, second=t.second)
            except Exception:
                continue
    return None


def _group_matches_by_end_first_half(fixtures: Iterable[Dict[str, Any]]) -> Dict[str, list]:
    """Return dict keyed by end-of-first-half hour (HH:MM) -> list of fixtures."""
    groups: Dict[str, list] = {}
    for f in fixtures:
        try:
            recs = f.get("recommendations", {}).get("over_under") or []
        except Exception:
            recs = []

        over15_ok = False
        for r in recs:
            if r.get("market") == "Over 1.5" and r.get("confidence") is not None:
                try:
                    if float(r.get("confidence")) >= 0.8:
                        over15_ok = True
                        break
                except Exception:
                    continue

        if not over15_ok:
            continue

        mt = f.get("match_time")
        dt = _parse_match_time(mt)
        if not dt:
            md = f.get("match_date")
            if md:
                try:
                    dt = datetime.fromisoformat(md)
                except Exception:
                    dt = None
        if not dt:
            logger.debug("Skipping fixture without parsable match_time: %s", f.get("fixture_id"))
            continue

        end_first = dt + timedelta(minutes=50)
        key = end_first.strftime("%H:%M")
        groups.setdefault(key, []).append(f)

    return groups


async def send_grouped_notifications_from_file_async(path: str = "data/all_predictions.json") -> None:
    try:
        def _read():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)

        data = await asyncio.to_thread(_read)
    except FileNotFoundError:
        logger.warning("Predictions file not found: %s", path)
        return
    fixtures = data.get("fixtures", [])
    await send_grouped_notifications_async(fixtures)


async def send_grouped_notifications_async(fixtures: Iterable[Dict[str, Any]]) -> None:
    groups = _group_matches_by_end_first_half(fixtures)
    if not groups:
        logger.info("No matches with Over 1.5 >= 80%% found")
        return

    token: Optional[str] = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.MY_CHAT_ID
    if not token or not chat_id:
        logger.warning("Telegram token or MY_CHAT_ID not configured; skipping notifications")
        return

    bot_app = None
    try:
        from app.main import app as fastapi_app
        bot_app = getattr(getattr(fastapi_app, 'state', None), 'bot_app', None)
    except Exception:
        bot_app = None

    for key in sorted(groups.keys()):
        fixtures_list = groups[key]
        lines = [f"{f.get('home_team')} vs {f.get('away_team')}" for f in fixtures_list]
        text = "\n".join(lines)

        if bot_app:
            try:
                await bot_app.bot.send_message(chat_id=chat_id, text=text)
                logger.info("Sent notification via Application.bot for %s: %d matches", key, len(fixtures_list))
            except Exception as e:
                logger.exception("Error sending telegram message via Application.bot for %s: %s", key, e)
                # If chat not found, give guidance
                if 'chat not found' in str(e).lower():
                    logger.error("Chat not found. Verify settings.MY_CHAT_ID (%s) and that the bot has permission to message that chat.", chat_id)
        else:
            send_url = f"https://api.telegram.org/bot{token}/sendMessage"

            def _post():
                return requests.post(send_url, json={"chat_id": chat_id, "text": text}, timeout=10)

            try:
                resp = await asyncio.to_thread(_post)
                if resp.status_code != 200:
                    logger.error("Failed to send telegram message for %s: %s", key, resp.text)
                    try:
                        body = resp.json()
                        if body.get('description') and 'chat not found' in body.get('description'):
                            logger.error("Telegram API reports chat not found. Check settings.MY_CHAT_ID (%s) and that the bot can message the chat.", chat_id)
                    except Exception:
                        pass
                else:
                    logger.info("Sent notification via HTTP for %s: %d matches", key, len(fixtures_list))
            except Exception as e:
                logger.exception("Error sending telegram message for %s: %s", key, e)


def send_grouped_notifications_from_file(path: str = "data/all_predictions.json") -> None:
    """Compatibility wrapper: schedule async notifier in running loop or run it."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_grouped_notifications_from_file_async(path))
    except RuntimeError:
        asyncio.run(send_grouped_notifications_from_file_async(path))

def group_matches_from_file(path: str = "data/all_predictions.json") -> Dict[str, list]:
    """Synchronous helper: read predictions file and return grouped matches dict.

    Returns groups keyed by HH:MM -> list of fixture dicts.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logger.warning("Predictions file not found: %s", path)
        return {}
    fixtures = data.get("fixtures", [])
    return _group_matches_by_end_first_half(fixtures)

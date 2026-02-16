#!/usr/bin/env python3
"""
NOTIFICATIONS — Reliable Discord webhook delivery with retry and fallback.

Shared module used by auto_scan.py, position_monitor.py, and execute_trade.py.
If Discord is unreachable, alerts are saved to a local JSONL fallback file
so no opportunity is silently lost.

Features:
  - 3 retries with exponential backoff (2s, 4s, 8s)
  - Discord embed chunking (respects 10 embed / 6000 char limits)
  - JSONL fallback file when Discord is persistently down
  - Dry-run mode for testing
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

from log_setup import get_logger

logger = get_logger(__name__)

__all__ = ["send_discord_alert", "send_discord_embeds"]

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent
FALLBACK_FILE = PROJECT_ROOT / "alerts_fallback.jsonl"


def _get_discord_webhook() -> str:
    """Lazy lookup of Discord webhook URL.

    Reading at call time (not import time) ensures dotenv has been loaded
    by the calling module, fixing watchdog.py which imports notifications
    before loading .env.
    """
    return os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("DISCORD_WEBHOOK") or ""

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds: 2, 4, 8


async def _post_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
) -> bool:
    """POST to Discord with exponential backoff. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 204):
                    return True
                elif resp.status == 429:
                    # Rate limited — use Discord's retry_after if available
                    try:
                        data = await resp.json()
                        wait = data.get("retry_after", BACKOFF_BASE ** (attempt + 1))
                    except Exception:
                        wait = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Discord rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                else:
                    body = await resp.text()
                    logger.warning(f"Discord returned {resp.status}: {body[:200]} (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Discord request failed: {e} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))

    return False


def _save_to_fallback(embeds: list[dict], context: str = ""):
    """Append alert to JSONL fallback file when Discord is unreachable."""
    try:
        record = {
            "timestamp": datetime.now(ET).isoformat(),
            "context": context,
            "embeds": embeds,
        }
        with open(FALLBACK_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.warning(f"Discord unreachable — alert saved to {FALLBACK_FILE}")
    except Exception as e:
        logger.error(f"Failed to write fallback alert: {e}")


def _chunk_embeds(embeds: list[dict]) -> list[list[dict]]:
    """Split embeds into Discord-safe chunks (max 10 embeds, ~5500 chars per message)."""
    chunks = []
    current_chunk = []
    current_chars = 0

    for embed in embeds:
        desc_len = len(embed.get("description", "")) + len(embed.get("title", ""))
        if current_chars + desc_len > 5500 or len(current_chunk) >= 9:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(embed)
        current_chars += desc_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


UTC = ZoneInfo("UTC")


async def send_discord_alert(
    title: str,
    description: str,
    color: int = 0xFF6600,
    context: str = "",
):
    """
    Send a single Discord embed alert with retry and fallback.

    Used by position_monitor.py and execute_trade.py for simple alerts.
    """
    embed = {
        "title": title,
        "description": description,
        "color": color,
        # Discord expects UTC ISO 8601 timestamps for embed timestamps.
        # Using ET here caused Discord to display incorrect times.
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    await send_discord_embeds([embed], context=context)


async def send_discord_embeds(
    embeds: list[dict],
    dry_run: bool = False,
    context: str = "",
):
    """
    Send multiple Discord embeds with chunking, retry, and fallback.

    Used by auto_scan.py for multi-embed scan results.
    Falls back to JSONL file if Discord is persistently unreachable.
    """
    webhook_url = _get_discord_webhook()

    if not webhook_url:
        logger.warning("No DISCORD_WEBHOOK set — skipping notification")
        _save_to_fallback(embeds, context=context or "no_webhook")
        return

    if not embeds:
        return

    chunks = _chunk_embeds(embeds)

    if dry_run:
        print("\n[DRY RUN] Would send to Discord:")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Message {i+1} ({len(chunk)} embeds) ---")
            for embed in chunk:
                print(f"  Title: {embed.get('title', 'N/A')}")
                print(f"  Desc:  {embed.get('description', '')[:200]}...")
        return

    failed_embeds = []

    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            payload = {"embeds": chunk}
            success = await _post_with_retry(session, webhook_url, payload)

            if success:
                logger.debug(f"Discord alert sent ({len(chunk)} embeds)")
            else:
                logger.error(f"Discord delivery failed after {MAX_RETRIES} retries")
                failed_embeds.extend(chunk)

            # Rate limit spacing between chunk sends
            if len(chunks) > 1:
                await asyncio.sleep(1)

    # Save any failed embeds to fallback
    if failed_embeds:
        _save_to_fallback(failed_embeds, context=context or "retry_exhausted")

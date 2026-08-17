"""Shared OpenAI-compatible async client with hard timeouts."""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from app.core.config import Settings


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    timeout = httpx.Timeout(
        settings.openai_timeout_sec,
        connect=min(10.0, float(settings.openai_timeout_sec)),
    )
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=timeout,
        max_retries=settings.openai_max_retries,
    )

"""Loguru setup with readable console format and stdlib interception."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from app.core.config import Settings

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>{extra[request_id]}</yellow> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_CONFIGURED = False


def _ensure_request_id(record: dict) -> None:
    record["extra"].setdefault("request_id", "-")


class InterceptHandler(logging.Handler):
    """Forward stdlib logging (uvicorn/sqlalchemy) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_logging(settings: Settings | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = (settings.log_level if settings else "INFO").upper()

    logger.remove()
    logger.configure(patcher=_ensure_request_id)
    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=settings.app_env == "dev" if settings else False,
        enqueue=False,
    )

    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(level)

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "asyncio",
        "httpx",
        "httpcore",
        "openai",
    ):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False
        logging.getLogger(name).setLevel(level)

    if level not in {"DEBUG", "TRACE"}:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    _CONFIGURED = True
    logger.info(
        "Logging configured | level={} env={}",
        level,
        getattr(settings, "app_env", "?"),
    )

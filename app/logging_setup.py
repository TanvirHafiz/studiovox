"""Logging configuration. Every job also gets its own log file under jobs/<id>/job.log."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.config import config

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging() -> logging.Logger:
    config.ensure_dirs()
    logger = logging.getLogger("studiovox")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(stream_handler)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = config.logs_dir / f"studiovox_{ts}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger


def job_logger(job_id: str, job_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"studiovox.job.{job_id}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(job_dir / "job.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    return logger

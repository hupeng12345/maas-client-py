from __future__ import annotations

import logging
from typing import TextIO


def get_logger(name: str = "maas_client", *, level: int = logging.INFO, stream: TextIO | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]

"""Centralised logging configuration.

Use ``get_logger(__name__)`` in modules instead of bare ``print``.
``setup_logging`` is called once from ``pipeline.py`` (or notebooks) and
attaches a single console handler with timestamps + module names.
"""
from __future__ import annotations
import logging
import sys

_CONFIGURED = False


def setup_logging(level: int | str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname).1s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("plosone")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'plosone' namespace."""
    if not name.startswith("plosone"):
        if name.startswith("src."):
            name = "plosone." + name[4:]
        else:
            name = f"plosone.{name}"
    return logging.getLogger(name)

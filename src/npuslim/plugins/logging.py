"""Shared logging helpers for NPUSlim plugin lifecycle messages."""

from __future__ import annotations

from loguru import logger as _logger

PATCH_LOG_PREFIX = "[NPUSlimPatch]"


def _prefix_patch_log(record: dict) -> None:
    record["message"] = f"{PATCH_LOG_PREFIX} {record['message']}"


patch_logger = _logger.patch(_prefix_patch_log)

"""MSC: mode-switched token-set control (P0)."""

from __future__ import annotations

__all__ = ["SAFE_MODES", "SAFETY_STATES"]

SAFETY_STATES = ("REFUSE", "DENY", "RELATED_SAFE", "FOLLOW", "INVALID")
SAFE_MODES = ("REFUSE", "DENY", "RELATED_SAFE")

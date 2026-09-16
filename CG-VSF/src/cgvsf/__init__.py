"""CG-VSF: counterexample-guided visual safety falsification (P0)."""

from __future__ import annotations

__all__ = ["FAIL_MODES"]

FAIL_MODES = ("REFUSE", "DENY", "RELATED_SAFE", "GARBAGE")

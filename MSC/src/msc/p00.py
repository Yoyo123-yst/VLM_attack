"""P0-0: tokenize frozen phrase banks. Tokenizer only; no attack."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .protocol import load_frozen
from .phrases import public_tokensets
from .verdict import p00_verdict


def load_tokenizer(model_path: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    return processor.tokenizer


def run_p00(
    *,
    frozen: Optional[Mapping[str, Any]] = None,
    tokenizer=None,
) -> Dict[str, Any]:
    blob = dict(frozen or load_frozen())
    tok = tokenizer or load_tokenizer(blob["model"]["local_path"])
    tokensets = public_tokensets(tok, blob)
    gate = p00_verdict(tokensets, frozen_gates=blob["gates"])
    return {"tokensets": tokensets, "verdict": gate}

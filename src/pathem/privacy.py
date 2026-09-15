"""Never print raw model text. Hash or drop it unless store_raw_outputs is on."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


RAW_KEYS = ("text", "raw_output", "rhc_text", "safe_text", "prefix_text", "y")


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def strip_raw_text(row: Mapping[str, Any], keep_chars: int = 0) -> dict[str, Any]:
    out = dict(row)
    blob = out.get("text") or out.get("raw_output") or ""
    if blob:
        out["output_hash"] = text_hash(str(blob))
        out["chars"] = len(str(blob))
    for key in RAW_KEYS:
        if key in out:
            if keep_chars > 0 and key == "text":
                out["text_head"] = str(out[key])[:keep_chars]
            out.pop(key, None)
    judge = out.get("judge")
    if isinstance(judge, dict) and "text" in judge:
        judge = dict(judge)
        judge.pop("text", None)
        out["judge"] = judge
    return out


def console_safe(row: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "run_id",
        "method",
        "query_id",
        "carrier_id",
        "delta_id",
        "random_seed",
        "token_count",
        "terminal_label",
        "core_rhc",
        "response_mode",
        "safety",
        "output_hash",
        "chars",
        "status",
    )
    stripped = strip_raw_text(row)
    return {k: stripped[k] for k in keep if k in stripped}

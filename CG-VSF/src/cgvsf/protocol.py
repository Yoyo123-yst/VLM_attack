"""Load and hash the frozen P0 protocol. Thresholds are not editable at runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
HERE = Path(__file__).resolve().parents[2]
FROZEN_PATH = HERE / "CGVSF_P0_FROZEN.json"


def load_frozen(path: Path = FROZEN_PATH) -> Dict[str, Any]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not blob.get("frozen"):
        raise RuntimeError("CG-VSF freeze file is not marked frozen")
    if blob.get("scope", {}).get("store_raw_outputs") is not False:
        raise RuntimeError("store_raw_outputs must be false")
    return blob


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def judge_source_hash() -> str:
    files = [
        ROOT / "src" / "n0" / "axes.py",
        ROOT / "src" / "p0" / "judge.py",
        ROOT / "src" / "pathem" / "labels.py",
    ]
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
        h.update(b"\n")
    return h.hexdigest()


def parse_cell(spec: str) -> Tuple[str, str]:
    q, c = spec.split(":", 1)
    return q, c


def hard_cells(frozen: Dict[str, Any] | None = None) -> Tuple[Tuple[str, str], ...]:
    blob = frozen or load_frozen()
    return tuple(parse_cell(s) for s in blob["scope"]["hard_cells"])


def assert_query_unsealed(query_id: str, sealed: Iterable[str] | None = None) -> None:
    q = str(query_id)
    if q.startswith("h"):
        try:
            n = int(q[1:])
        except ValueError:
            n = -1
        if 83 <= n <= 130:
            raise RuntimeError(f"sealed query {q}")
    if sealed and q in set(sealed):
        raise RuntimeError(f"sealed query {q}")

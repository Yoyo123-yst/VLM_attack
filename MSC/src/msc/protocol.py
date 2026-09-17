"""Load and hash the frozen MSC P0 protocol. Thresholds are not editable at runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
HERE = Path(__file__).resolve().parents[2]
FROZEN_PATH = HERE / "MSC_P0_FROZEN.json"
BANK_NAMES = ("REFUSE", "DENY", "RELATED_SAFE", "FOLLOW")


def load_frozen(path: Path = FROZEN_PATH) -> Dict[str, Any]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not blob.get("frozen"):
        raise RuntimeError("MSC freeze file is not marked frozen")
    if blob.get("scope", {}).get("store_raw_outputs") is not False:
        raise RuntimeError("store_raw_outputs must be false")
    if blob.get("states", {}).get("store_raw_outputs") is not False:
        raise RuntimeError("states.store_raw_outputs must be false")
    if blob.get("states", {}).get("no_latent_probe") is not True:
        raise RuntimeError("MSC must not use a latent probe")
    banks = blob.get("phrase_banks") or {}
    for name in BANK_NAMES:
        phrases = banks.get(name)
        if not isinstance(phrases, list) or not phrases:
            raise RuntimeError(f"frozen phrase bank {name} missing")
    if banks.get("harvest_from_hard_cells") or banks.get("harvest_from_sealed"):
        raise RuntimeError("phrase harvest from hard/sealed is forbidden")
    if banks.get("harvest_from_calib_into_freeze"):
        raise RuntimeError("calibration must not write phrases back into the freeze")
    return blob


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def is_sealed_query(query_id: str) -> bool:
    q = str(query_id)
    if q.startswith("h"):
        try:
            n = int(q[1:])
        except ValueError:
            n = -1
        if 83 <= n <= 130:
            return True
    return False


def assert_query_unsealed(query_id: str, sealed: Iterable[str] | None = None) -> None:
    if is_sealed_query(query_id):
        raise RuntimeError(f"sealed query {query_id}")
    if sealed and str(query_id) in set(sealed):
        raise RuntimeError(f"sealed query {query_id}")


def phrase_banks(frozen: Dict[str, Any] | None = None) -> Dict[str, list]:
    blob = frozen or load_frozen()
    return {name: list(blob["phrase_banks"][name]) for name in BANK_NAMES}

"""JSONL is the source of truth. Markdown is a projection. Atomic writes only."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

BANNED_KEYS = frozenset(
    {"text", "token_text", "raw", "output_text", "generation", "response", "decoded"}
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def strip_banned(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_banned(v) for k, v in obj.items() if k not in BANNED_KEYS}
    if isinstance(obj, list):
        return [strip_banned(x) for x in obj]
    return obj


def contains_banned(obj: Any) -> bool:
    if isinstance(obj, dict):
        if any(k in BANNED_KEYS for k in obj):
            return True
        return any(contains_banned(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_banned(x) for x in obj)
    return False


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = dict(row)
    if contains_banned(raw):
        raise RuntimeError("refusing to log banned raw-text keys")
    clean = strip_banned(raw)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = []
    for row in rows:
        raw = dict(row)
        if contains_banned(raw):
            raise RuntimeError("refusing to log banned raw-text keys")
        lines.append(json.dumps(strip_banned(raw), ensure_ascii=False))
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_status(
    path: Path,
    *,
    stage: str,
    state: str,
    frozen_sha: str,
    git_commit: str,
    started: str,
    completed: int = 0,
    total: int = 0,
    failed: int = 0,
    skipped: int = 0,
    latest: Optional[Mapping[str, Any]] = None,
    verdict: str = "PENDING",
    evidence: str = "",
    next_cmd: str = "",
) -> None:
    latest = latest or {}
    body = "\n".join(
        [
            "# MSC Experiment Status",
            "",
            f"- Stage: {stage}",
            f"- State: {state}",
            f"- Frozen config SHA256: `{frozen_sha}`",
            f"- Git commit: `{git_commit}`",
            f"- Started at: {started}",
            f"- Updated at: {now()}",
            "",
            "## Progress",
            f"| {completed} | {total} | {failed} | {skipped} |",
            "",
            "## Latest Result",
            f"- Cell/state: {latest.get('cell_id', '—')}",
            f"- Method: {latest.get('method', '—')}",
            f"- Mode transition: {latest.get('before_mode', '—')} → {latest.get('after_mode', '—')}",
            f"- core_rhc: {latest.get('core_rhc', '—')}",
            f"- Budget used: {latest.get('backward_used', '—')}",
            "",
            "## Gate",
            f"- Current verdict: {verdict}",
            f"- Evidence: {evidence or '—'}",
            f"- Next allowed command: `{next_cmd or '—'}`",
            "",
        ]
    )
    atomic_write_text(path, body)


def append_run_log(path: Path, event: str, fields: Mapping[str, Any]) -> None:
    lines = [f"## {now()} — {event}", ""]
    for k, v in fields.items():
        if k in BANNED_KEYS:
            continue
        lines.append(f"- {k}: {v}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.flush()

"""Prefix rows for committor training. Label is the full-trajectory terminal."""

from __future__ import annotations

from typing import Any, Sequence

from .privacy import text_hash

PREFIX_STAGES = ("early", "mid", "late", "pre_eos")


def prefix_cuts(n: int) -> dict[str, int]:
    n = int(n)
    if n <= 0:
        return {k: 0 for k in PREFIX_STAGES}
    return {
        "early": max(1, n // 4),
        "mid": max(1, n // 2),
        "late": max(1, (3 * n) // 4),
        "pre_eos": max(1, n - 1 if n > 1 else 1),
    }


def split_words(text: str) -> list[str]:
    return (text or "").strip().split()


def prefix_rows_from_text(
    text: str,
    terminal_label: str,
    core_rhc: bool,
    trajectory_id: str,
    query_id: str,
) -> list[dict[str, Any]]:
    """Build four prefixes. Supervision = full trajectory, not prefix judge."""
    words = split_words(text)
    cuts = prefix_cuts(len(words))
    rows = []
    for stage in PREFIX_STAGES:
        k = int(cuts[stage])
        pref = " ".join(words[:k])
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "query_id": query_id,
                "stage": stage,
                "prefix_word_count": k,
                "full_word_count": len(words),
                "frac": (k / len(words)) if words else 0.0,
                "prefix_hash": text_hash(pref),
                "terminal_label": terminal_label,
                "y_rhc": int(bool(core_rhc)),
            }
        )
    return rows


def prefix_rows_from_ids(
    token_ids: Sequence[int],
    terminal_label: str,
    core_rhc: bool,
    trajectory_id: str,
    query_id: str,
) -> list[dict[str, Any]]:
    ids = [int(x) for x in token_ids]
    cuts = prefix_cuts(len(ids))
    rows = []
    for stage in PREFIX_STAGES:
        k = int(cuts[stage])
        pref = ids[:k]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "query_id": query_id,
                "stage": stage,
                "prefix_token_count": k,
                "full_token_count": len(ids),
                "frac": (k / len(ids)) if ids else 0.0,
                "prefix_id_hash": text_hash(",".join(str(i) for i in pref)),
                "terminal_label": terminal_label,
                "y_rhc": int(bool(core_rhc)),
            }
        )
    return rows


def assert_group_split(rows: Sequence[dict[str, Any]], train_q, val_q) -> None:
    """No query and no trajectory may appear in both train and val."""
    train_q, val_q = set(train_q), set(val_q)
    if train_q & val_q:
        raise RuntimeError("query overlap in committor split")
    train_t = {r["trajectory_id"] for r in rows if r["query_id"] in train_q}
    val_t = {r["trajectory_id"] for r in rows if r["query_id"] in val_q}
    if train_t & val_t:
        raise RuntimeError("trajectory leaked across committor split")

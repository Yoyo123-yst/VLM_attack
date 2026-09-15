"""Query-disjoint PathEM splits on CR-0 cells. Sealed IDs stay unread."""

from __future__ import annotations

from typing import Any

from .protocol import PILOT_CARRIERS, PILOT_QUERIES, SEALED, assert_query_allowed

# Stage 1 committor uses CR-0 greedy labels as weak prefixes only after Stage 0.
# Query-disjoint: train/val never share a query_id.
COMMITTOR_TRAIN = ("h49", "h56", "h66")
COMMITTOR_VAL = ("h64", "h72")
PILOT_ATTACK = ("h49", "h53", "h56", "h64", "h66", "h72")
# Held-out evaluation stays empty until Stage 3/4. Do not use h76/h80/h81 yet
# as a fake test set during Stage 0–1 hyperparameter work.
HELDOUT_EVAL: tuple[str, ...] = ()

CR0_LEFTOVER = ("h61", "h68", "h74", "h76", "h80", "h81")


def split_map() -> dict[str, tuple[str, ...]]:
    return {
        "committor_train": COMMITTOR_TRAIN,
        "committor_validation": COMMITTOR_VAL,
        "pilot_attack": PILOT_ATTACK,
        "heldout_evaluation": HELDOUT_EVAL,
        "cr0_leftover_unused": CR0_LEFTOVER,
        "sealed": SEALED,
        "carriers": PILOT_CARRIERS,
    }


def assert_no_leak() -> None:
    train, val = set(COMMITTOR_TRAIN), set(COMMITTOR_VAL)
    if train & val:
        raise RuntimeError("committor train/val query overlap")
    for q in list(train) + list(val) + list(PILOT_ATTACK) + list(HELDOUT_EVAL):
        assert_query_allowed(q, PILOT_QUERIES)
        if q in SEALED:
            raise RuntimeError(f"sealed query leaked: {q}")


def cr0_cell_id(query_id: str, carrier_id: str, restart: int) -> str:
    return f"{query_id}:{carrier_id}:e16:r{int(restart)}"

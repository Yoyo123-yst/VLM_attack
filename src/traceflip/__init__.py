"""TraceFlip public surface.

Import order matters only for documentation; every submodule is importable
without loading a model, so the CPU test-suite can exercise the algebra.
"""

from .protocol import (
    ALPHA,
    DECODE,
    EPS,
    EVAL_SEEDS,
    FLIP_KAPPA,
    KEEP_KAPPA,
    OPT_SEEDS,
    PILOT_CARRIERS,
    PILOT_QUERIES,
    SEALED,
    SUCCESS,
    assert_query_allowed,
    assert_seed_split,
)

__all__ = [
    "ALPHA",
    "DECODE",
    "EPS",
    "EVAL_SEEDS",
    "FLIP_KAPPA",
    "KEEP_KAPPA",
    "OPT_SEEDS",
    "PILOT_CARRIERS",
    "PILOT_QUERIES",
    "SEALED",
    "SUCCESS",
    "assert_query_allowed",
    "assert_seed_split",
]

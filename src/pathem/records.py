"""Trajectory records. Raw text is hashed; optional path only if explicitly enabled."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .privacy import strip_raw_text, text_hash


RECORD_FIELDS = (
    "run_id",
    "method",
    "model_id",
    "query_id",
    "carrier_id",
    "delta_id",
    "parent_delta_id",
    "random_seed",
    "decoding_config_id",
    "trajectory_id",
    "parent_trajectory_id",
    "branch_depth",
    "token_count",
    "prefix_checkpoints",
    "sequence_log_probability",
    "committor_scores",
    "importance_weight",
    "terminal_label",
    "judge_scores",
    "output_hash",
    "raw_output_path_optional",
    "wall_time",
    "gpu_memory",
    "status",
)


@dataclass
class TrajectoryRecord:
    run_id: str
    method: str
    model_id: str = "Qwen2-VL-7B-Instruct"
    query_id: str = ""
    carrier_id: str = ""
    delta_id: Optional[str] = None
    parent_delta_id: Optional[str] = None
    random_seed: Optional[int] = None
    decoding_config_id: str = "std_t095_p092"
    trajectory_id: str = ""
    parent_trajectory_id: Optional[str] = None
    branch_depth: int = 0
    token_count: int = 0
    prefix_checkpoints: list[int] = field(default_factory=list)
    sequence_log_probability: Optional[float] = None
    committor_scores: list[float] = field(default_factory=list)
    importance_weight: float = 1.0
    terminal_label: str = "OTHER"
    judge_scores: dict[str, Any] = field(default_factory=dict)
    output_hash: Optional[str] = None
    raw_output_path_optional: Optional[str] = None
    wall_time: float = 0.0
    gpu_memory: float = 0.0
    status: str = "ok"
    core_rhc: bool = False
    response_mode: Optional[str] = None
    safety: Optional[str] = None

    def attach_text(self, text: str, store_raw: bool = False) -> None:
        self.output_hash = text_hash(text or "")
        self.token_count = max(self.token_count, len((text or "").split()))
        if store_raw:
            raise RuntimeError("store_raw_outputs is disabled in Stage 0")


def record_schema() -> tuple[str, ...]:
    return RECORD_FIELDS


def serialize_record(rec: TrajectoryRecord) -> dict[str, Any]:
    blob = asdict(rec)
    return strip_raw_text(blob)

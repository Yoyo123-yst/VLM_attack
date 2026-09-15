"""Load PathEM YAML. Unknown keys are kept; required keys are checked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .protocol import (
    ALPHA,
    DECODE,
    EPS,
    EVAL_SEEDS,
    IMAGE_SIZE,
    MAX_NEW_TOKENS,
    OPT_SEEDS,
    PILOT_CARRIERS,
    PILOT_QUERIES,
    SEED_BASE,
    STORE_RAW_OUTPUTS,
    VAL_SEEDS,
    assert_seed_split,
)

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
DEFAULT_CFG = ROOT / "configs" / "pathem" / "pilot.yaml"


def load_cfg(path: Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CFG
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = dict(raw)
    cfg.setdefault("epsilon", EPS)
    cfg.setdefault("alpha", ALPHA)
    cfg.setdefault("image_size", IMAGE_SIZE)
    cfg.setdefault("generation_max_tokens", MAX_NEW_TOKENS)
    cfg.setdefault("temperature", DECODE["temperature"])
    cfg.setdefault("top_p", DECODE["top_p"])
    cfg.setdefault("optimization_seeds", list(OPT_SEEDS))
    cfg.setdefault("validation_seeds", list(VAL_SEEDS))
    cfg.setdefault("evaluation_seeds", list(EVAL_SEEDS))
    cfg.setdefault("query_ids", list(PILOT_QUERIES))
    cfg.setdefault("carrier_ids", list(PILOT_CARRIERS))
    cfg.setdefault("store_raw_outputs", STORE_RAW_OUTPUTS)
    cfg.setdefault("seed_base", SEED_BASE)
    assert_seed_split(
        cfg["optimization_seeds"],
        cfg["validation_seeds"],
        cfg["evaluation_seeds"],
    )
    if cfg.get("store_raw_outputs"):
        raise RuntimeError("store_raw_outputs must stay false in Stage 0")
    cfg["_path"] = str(cfg_path)
    return cfg

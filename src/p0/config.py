from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
DEFAULT_CFG = ROOT / "configs" / "p0.yaml"


def load_cfg(path: Path | None = None, scale: str = "mini") -> Dict[str, Any]:
    cfg_path = path or DEFAULT_CFG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    if scale not in cfg["scale"]:
        raise ValueError(f"unknown scale {scale}")
    overlay = cfg["scale"][scale]
    cfg["active_scale"] = scale
    cfg["n_harmful_discover"] = overlay["n_harmful_discover"]
    cfg["n_harmful_holdout"] = overlay["n_harmful_holdout"]
    cfg["n_benign"] = overlay["n_benign"]
    cfg["n_carriers_train"] = overlay["n_carriers_train"]
    cfg["n_carriers_test"] = overlay["n_carriers_test"]
    cfg["layers"]["candidates"] = overlay["layers"]
    cfg["subspace"]["ranks"] = overlay["ranks"]
    cfg["attack"]["sampler_steps"] = overlay["sampler_steps"]
    cfg["attack"]["attack_steps"] = overlay["attack_steps"]
    cfg["model"]["max_new_tokens"] = overlay["max_new_tokens"]
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / scale)
    return cfg


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    p.add_argument("--scale", type=str, default="mini", choices=["mini", "full"])
    p.add_argument(
        "--stage",
        type=str,
        default="smoke",
        choices=[
            "smoke",
            "probe",
            "traces",
            "patch",
            "subspace",
            "causal",
            "attack",
            "report",
            "mini",
            "full",
        ],
    )
    return p


def ensure_out(cfg: Dict[str, Any]) -> Path:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out

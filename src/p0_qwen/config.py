from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
DEFAULT_CFG = ROOT / "configs" / "p0_qwen.yaml"


def load_cfg(path: Path | None = None, scale: str = "full") -> Dict[str, Any]:
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
    return cfg


def setting_out(cfg: Dict[str, Any], setting: str | None = None) -> Path:
    base = Path(cfg["output_dir"])
    if setting:
        return base / cfg["active_scale"] / setting
    return base / cfg["active_scale"]


def ensure_out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=str, default=str(DEFAULT_CFG))
    p.add_argument("--scale", type=str, default="full", choices=["mini", "full"])
    p.add_argument(
        "--stage",
        type=str,
        default="smoke",
        choices=[
            "smoke",
            "calibrate",
            "probe",
            "traces",
            "patch",
            "subspace",
            "causal",
            "attack",
            "report",
            "full",
            "integrity",
        ],
    )
    p.add_argument("--setting", type=str, default="", choices=["", "native", "prefix"])
    p.add_argument(
        "--smoke-one",
        action="store_true",
        help="attack: 1 unseen test query x eps=8/255 x 4 methods",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="ignore cached patch/causal/attack JSON even if provenance matches",
    )
    return p

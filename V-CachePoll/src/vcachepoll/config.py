from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
DEFAULT_CFG = ROOT / "configs" / "p0.yaml"


def load_cfg(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = path or DEFAULT_CFG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return deepcopy(cfg)


def out_dir(cfg: Dict[str, Any]) -> Path:
    path = Path(cfg["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path

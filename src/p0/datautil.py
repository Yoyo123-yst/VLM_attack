from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple

from PIL import Image

from .catalog import all_pairs


CARRIER_JSON = Path("/root/autodl-tmp/multimodal_attack_project/data/p0/carriers.json")


def load_carriers(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    spec = json.loads(CARRIER_JSON.read_text())
    root = Path(spec["root"])
    items = []
    for row in spec["images"]:
        path = root / row["file"]
        items.append({**row, "path": str(path)})
    train = [x for x in items if x["split"] == "train"][: cfg["n_carriers_train"]]
    test = [x for x in items if x["split"] == "test"][: cfg["n_carriers_test"]]
    return train, test


def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def all_query_pairs() -> List[Dict[str, Any]]:
    return all_pairs()


def save_json(path: Path, obj: Any) -> None:
    """Atomically replace a JSON checkpoint.

    Long GPU runs may be interrupted while saving.  Writing directly to the
    destination can leave a truncated file that cannot be resumed, so write
    and fsync a sibling temporary file before ``os.replace``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

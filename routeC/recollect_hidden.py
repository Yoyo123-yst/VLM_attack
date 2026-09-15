#!/usr/bin/env python3
"""P0.4 (GPU standby — DO NOT RUN in Phase-0 A段).

Recollect `last_user` hidden states for the 42 DENY records + same-cell
REFUSE/ANSWER controls, saving each as routeC/out/deny_hidden/<record_id>.npz.

Hidden keys: L12/L16/L20/L24:last_user. Pure forward pass (~1h GPU).

Run only after A段 results are reviewed. Not executed automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from n1.preflight import all_carrier_index  # noqa: E402
from p0.catalog import all_pairs  # noqa: E402
from p0.datautil import load_json, open_image  # noqa: E402
from p0_qwen.config import load_cfg  # noqa: E402
from p0_qwen.model import QwenP0  # noqa: E402
from p0_qwen.patching import _x01_from_delta  # noqa: E402

OUT = ROOT / "routeC" / "out" / "deny_hidden"
CR0 = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
N1R = ROOT / "outputs" / "n1r_fast" / "candidates.json"
LAYERS = [12, 16, 20, 24]


def _records(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(c.get("records", []))


def main() -> None:
    cfg = load_cfg(scale="full")
    wrapper = QwenP0(cfg, setting="native")
    carriers = all_carrier_index()
    query_text = {p["id"]: p["query"] for p in all_pairs()}

    cr0 = _records(load_json(CR0))
    n1r = _records(load_json(N1R))
    all_records = cr0 + n1r

    deny = [r for r in all_records if r.get("response_mode") == "DENY"]
    # same-cell controls: same (query, carrier) with REFUSE or ANSWER mode
    deny_cells = {(r["query_id"], r["carrier_id"]) for r in deny}
    controls = [
        r
        for r in all_records
        if (r["query_id"], r["carrier_id"]) in deny_cells
        and r.get("response_mode") in {"REFUSE", "ANSWER"}
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    for r in deny + controls:
        rid = r["record_id"]
        if rid in seen:
            continue
        seen.add(rid)
        img = open_image(carriers[r["carrier_id"]]["path"])
        x01 = _x01_from_delta(wrapper, img, r.get("delta_path"))
        q = query_text.get(r["query_id"], "")
        hidden = wrapper.collect_hidden(
            img, q, layers=LAYERS, x01=x01
        )
        np.savez(OUT / f"{rid}.npz", **{k: v.numpy() for k, v in hidden.items()})
        print(f"P0.4 saved {rid}", flush=True)
    print(f"P0.4 done: {len(seen)} records", flush=True)


if __name__ == "__main__":
    main()

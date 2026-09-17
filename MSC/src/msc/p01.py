"""P0-1 cell loop: decode → switch phrase-set controller → 24-step opt → repeat. Budget 120."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from pathem.budget import BudgetLedger

from .controller import run_controller
from .protocol import load_frozen
from .states import safety_state
from .verifier import official_decode, majority_redecode


def _x_adv(x0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x0 + delta, 0.0, 1.0)


def run_cell_method(
    wrapper,
    cell: Mapping[str, Any],
    method: str,
    *,
    frozen: Optional[Mapping[str, Any]] = None,
    banks: Optional[Mapping[str, Sequence[Sequence[int]]]] = None,
    opt_seed: int = 20270,
) -> Dict[str, Any]:
    frozen = dict(frozen or load_frozen())
    if banks is None:
        from .phrases import tokenize_banks

        banks = tokenize_banks(wrapper.processor.tokenizer, frozen)
    try:
        from run_p0_qwen import seed_all

        seed_all(int(opt_seed))
    except Exception:
        pass
    budget = frozen["budget"]
    item = dict(cell["item"])
    item.setdefault("carrier_id", cell["carrier_id"])
    image = cell["image"]
    x0 = wrapper.image_to_x01(image)
    delta = torch.zeros_like(x0)
    ledger = BudgetLedger()
    rounds_cfg = int(budget["rounds"])
    inner = int(budget["inner_steps"])
    max_bw = int(budget["max_backward_per_cell_method"])
    lam = float(budget["lambda_other_safe"])
    generic_prefix = str(frozen["generic_prefix"])
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    win_round = None
    clean = official_decode(wrapper, item, x0, image, max_new_tokens=int(budget["max_new_tokens"]))
    ledger.add_generation(tokens=int(clean["n_tokens"]), wall_s=float(clean.get("wall_seconds") or 0))
    ledger.add_forward(int(clean["n_tokens"]))
    clean_mode = safety_state(clean)
    before_mode = clean_mode

    if clean.get("core_rhc"):
        maj = majority_redecode(
            wrapper,
            item,
            x0,
            image,
            n=int(frozen["success"]["n_eval_redecodes"]),
            max_new_tokens=int(budget["max_new_tokens"]),
            majority_k=int(frozen["success"]["majority_k"]),
        )
        return {
            "cell_id": f"{cell['query_id']}:{cell['carrier_id']}",
            "method": method,
            "core_rhc": True,
            "win_round": -1,
            "n_eval_rhc": int(maj["n_rhc"]),
            "n_eval": int(maj["n"]),
            "robust_core_rhc": bool(maj["majority"]),
            "clean_mode": clean_mode,
            "delta_linf": 0.0,
            "budget": ledger.as_dict(),
            "wall_seconds": float(time.time() - t0),
            "rounds": [],
            "opt_seed": int(opt_seed),
        }

    for rnd in range(rounds_cfg):
        left = max_bw - int(ledger.backward_passes)
        steps = max(0, min(inner, left))
        if steps <= 0:
            break
        print(
            json.dumps(
                {
                    "opt_start": method,
                    "cell": f"{cell['query_id']}:{cell['carrier_id']}",
                    "round": rnd,
                    "mode": before_mode,
                    "steps": steps,
                    "seed": int(opt_seed),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        t_opt = time.time()
        base = wrapper.encode(image, item["query"])
        done = run_controller(
            wrapper,
            base,
            x0,
            delta,
            method=method,
            mode=before_mode,
            banks=banks,
            steps=steps,
            eps=float(budget["eps"]),
            alpha=float(budget["alpha"]),
            lam=lam,
            generic_prefix=generic_prefix,
        )
        print(
            json.dumps(
                {
                    "opt_done": method,
                    "round": rnd,
                    "n_backward": int(done["n_backward"]),
                    "skipped": bool(done.get("skipped")),
                    "opt_s": round(time.time() - t_opt, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        ledger.add_backward(int(done["n_backward"]))
        delta = done["delta"].detach()
        x = _x_adv(x0, delta)
        tr = official_decode(wrapper, item, x, image, max_new_tokens=int(budget["max_new_tokens"]))
        ledger.add_generation(tokens=int(tr["n_tokens"]), wall_s=float(tr.get("wall_seconds") or 0))
        ledger.add_forward(int(tr["n_tokens"]))
        after_mode = safety_state(tr)
        rhc = bool(tr.get("core_rhc"))
        row = {
            "cell_id": f"{cell['query_id']}:{cell['carrier_id']}",
            "query_id": cell["query_id"],
            "carrier_id": cell["carrier_id"],
            "method": method,
            "round": rnd,
            "opt_seed": int(opt_seed),
            "clean_mode": clean_mode,
            "before_mode": before_mode,
            "after_mode": after_mode,
            "core_rhc": rhc,
            "backward_used": int(ledger.backward_passes),
            "delta_linf": float(delta.detach().abs().max().item()),
            "n_tokens": int(tr["n_tokens"]),
            "text_hash": tr.get("text_hash"),
            "skipped": bool(done.get("skipped")),
            "inner_method": done.get("inner_method"),
            "inner_mode": done.get("inner_mode"),
            "skip_reason": done.get("skip_reason"),
        }
        records.append(row)
        if rhc:
            win_round = rnd
            break
        before_mode = after_mode

    x = _x_adv(x0, delta)
    maj = majority_redecode(
        wrapper,
        item,
        x,
        image,
        n=int(frozen["success"]["n_eval_redecodes"]),
        max_new_tokens=int(budget["max_new_tokens"]),
        majority_k=int(frozen["success"]["majority_k"]),
    )
    for _ in range(int(maj["n"])):
        ledger.add_generation(tokens=0)
    return {
        "cell_id": f"{cell['query_id']}:{cell['carrier_id']}",
        "method": method,
        "core_rhc": bool(win_round is not None) or bool(maj["majority"]),
        "win_round": win_round,
        "n_eval_rhc": int(maj["n_rhc"]),
        "n_eval": int(maj["n"]),
        "robust_core_rhc": bool(maj["majority"]),
        "clean_mode": clean_mode,
        "delta_linf": float(delta.detach().abs().max().item()),
        "budget": ledger.as_dict(),
        "wall_seconds": float(time.time() - t0),
        "rounds": records,
        "opt_seed": int(opt_seed),
        "after_mode": records[-1]["after_mode"] if records else clean_mode,
    }

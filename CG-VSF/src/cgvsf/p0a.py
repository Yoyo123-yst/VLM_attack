"""P0-A cell loop: decode → certificate → 24-step opt → repeat. Budget 120."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from p0.model import COMPLY_WORDS, REFUSE_WORDS, token_id_list
from pathem.budget import BudgetLedger
from traceflip.flip import generic_pgd, refusal_margin_objective, targeted_prefix_objective
from traceflip.gate import encode_gate_prefix
from traceflip.run import GENERIC_PREFIX_TEXT

from .certificates import extract_failure_certificate, prefix_tag, recurrence, update_certificate_set
from .objective import accumulated_objective, sequential_lse_pgd
from .protocol import load_frozen
from .verifier_adapter import decode_prefix_text, official_decode


def _x_adv(x0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x0 + delta, 0.0, 1.0)


def _opt_objective(method: str, certs: Sequence[Mapping[str, Any]], wrapper, round_i: int):
    if method == "refusal_margin":
        refuse = token_id_list(wrapper.processor, REFUSE_WORDS)
        comply = token_id_list(wrapper.processor, COMPLY_WORDS)

        def objective(w, inputs, delta, x0):
            return refusal_margin_objective(w, inputs, delta, x0, refuse, comply)

        return objective

    if method == "targeted_prefix_earlystop":
        ids = encode_gate_prefix(wrapper.processor, GENERIC_PREFIX_TEXT)

        def objective(w, inputs, delta, x0):
            return targeted_prefix_objective(w, inputs, delta, x0, ids)

        return objective

    if method == "gateflip_fair":
        full = encode_gate_prefix(wrapper.processor, GENERIC_PREFIX_TEXT)
        k = min(max(int(round_i) + 1, 1), len(full))
        ids = full[:k]

        def objective(w, inputs, delta, x0):
            return targeted_prefix_objective(w, inputs, delta, x0, ids)

        return objective

    if method == "last_certificate":
        ids_list = [list(certs[-1]["token_ids"])] if certs else []
    else:
        ids_list = [list(c["token_ids"]) for c in certs]
    return accumulated_objective(ids_list)


def run_cell_method(
    wrapper,
    cell: Mapping[str, Any],
    method: str,
    *,
    frozen: Optional[Mapping[str, Any]] = None,
    opt_seed: int = 20260,
    eval_seed: int = 40460,
) -> Dict[str, Any]:
    frozen = dict(frozen or load_frozen())
    try:
        from run_p0_qwen import seed_all

        seed_all(int(opt_seed))
    except Exception:
        pass
    budget = frozen["budget"]
    cert_cfg = frozen["certificates"]
    item = dict(cell["item"])
    item.setdefault("carrier_id", cell["carrier_id"])
    image = cell["image"]
    x0 = wrapper.image_to_x01(image)
    delta = torch.zeros_like(x0)
    ledger = BudgetLedger()
    rounds_cfg = int(budget["rounds"])
    inner = int(budget["inner_steps"])
    max_bw = int(budget["max_backward_per_cell_method"])
    C: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    win_round = None
    robust = None
    clean = official_decode(wrapper, item, x0, image, max_new_tokens=int(budget["max_new_tokens"]))
    ledger.add_generation(tokens=int(clean["n_tokens"]), wall_s=float(clean.get("wall_seconds") or 0))
    ledger.add_forward(int(clean["n_tokens"]))
    clean_mode = str(clean.get("terminal_label") or clean.get("certificate_mode"))
    before_mode = clean_mode
    exact = method == "accumulated_exact_only"

    def extract_cert(tr) -> Optional[Dict[str, Any]]:
        return extract_failure_certificate(
            tr.get("token_ids") or [],
            decode_prefix=lambda ids: decode_prefix_text(wrapper.processor, ids),
            label_fn=prefix_tag,
            max_len=int(cert_cfg["max_len"]),
            fallback_len=int(cert_cfg["fallback_len"]),
            stable_run=int(cert_cfg["stable_run"]),
            exact_only=exact,
        )

    if clean.get("core_rhc"):
        return {
            "cell_id": f"{cell['query_id']}:{cell['carrier_id']}",
            "method": method,
            "core_rhc": True,
            "win_round": -1,
            "robust_core_rhc": True,
            "clean_mode": clean_mode,
            "n_certificates": 0,
            "certificate_modes": [],
            "delta_linf": 0.0,
            "budget": ledger.as_dict(),
            "wall_seconds": float(time.time() - t0),
            "rounds": [],
            "opt_seed": int(opt_seed),
            "eval_seed": int(eval_seed),
        }

    cert0 = extract_cert(clean)
    C = update_certificate_set([], cert0, method=method, max_set=int(cert_cfg["max_set"]))

    for rnd in range(rounds_cfg):
        left = max_bw - int(ledger.backward_passes)
        steps = max(0, min(inner, left))
        if steps <= 0:
            break
        print(
            json.dumps(
                {
                    "opt_start": method,
                    "round": rnd,
                    "n_certs": len(C),
                    "cert_lens": [c.get("length") for c in C],
                    "steps": steps,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        t_opt = time.time()
        base = wrapper.encode(image, item["query"])
        if method in {"last_certificate", "accumulated_certificate", "accumulated_exact_only"}:
            ids_list = [list(c["token_ids"]) for c in C if c.get("token_ids")]
            if method == "last_certificate":
                ids_list = ids_list[-1:] if ids_list else []
            if len(ids_list) <= 2:
                obj = accumulated_objective(ids_list)
                done = generic_pgd(
                    wrapper,
                    base,
                    x0,
                    delta,
                    obj,
                    steps=steps,
                    eps=float(budget["eps"]),
                    alpha=float(budget["alpha"]),
                )
            else:
                done = sequential_lse_pgd(
                    wrapper,
                    base,
                    x0,
                    delta,
                    ids_list,
                    steps=steps,
                    eps=float(budget["eps"]),
                    alpha=float(budget["alpha"]),
                )
        else:
            obj = _opt_objective(method, C, wrapper, rnd)
            done = generic_pgd(
                wrapper,
                base,
                x0,
                delta,
                obj,
                steps=steps,
                eps=float(budget["eps"]),
                alpha=float(budget["alpha"]),
            )
        print(
            json.dumps(
                {
                    "opt_done": method,
                    "round": rnd,
                    "n_backward": int(done["n_backward"]),
                    "opt_s": round(time.time() - t_opt, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        ledger.add_backward(int(done["n_backward"]))
        delta = done["delta"].detach()
        x = _x_adv(x0, delta)
        tr = official_decode(
            wrapper, item, x, image, max_new_tokens=int(budget["max_new_tokens"])
        )
        ledger.add_generation(tokens=int(tr["n_tokens"]), wall_s=float(tr.get("wall_seconds") or 0))
        ledger.add_forward(int(tr["n_tokens"]))
        after_mode = str(tr.get("terminal_label") or tr.get("certificate_mode"))
        rhc = bool(tr.get("core_rhc"))
        cert = None if rhc else extract_cert(tr)
        rec_flags = recurrence(C, cert)
        row = {
            "cell_id": f"{cell['query_id']}:{cell['carrier_id']}",
            "query_id": cell["query_id"],
            "carrier_id": cell["carrier_id"],
            "method": method,
            "round": rnd,
            "opt_seed": int(opt_seed),
            "eval_seed": None,
            "clean_mode": clean_mode,
            "before_mode": before_mode,
            "after_mode": after_mode,
            "certificate_hash": None if cert is None else cert["hash"],
            "certificate_mode": None if cert is None else cert["mode"],
            "certificate_len": None if cert is None else cert["length"],
            "certificate_token_ids": None if cert is None else list(cert["token_ids"]),
            "n_certificates": len(C),
            "old_certificate_recurrence": bool(
                rec_flags["hash_recurrence"] or rec_flags["mode_recurrence"]
            ),
            "hash_recurrence": rec_flags["hash_recurrence"],
            "mode_recurrence": rec_flags["mode_recurrence"],
            "core_rhc": rhc,
            "backward_used": int(ledger.backward_passes),
            "delta_linf": float(delta.detach().abs().max().item()),
            "decoder_alignment_ok": True,
            "n_tokens": int(tr["n_tokens"]),
            "text_hash": tr.get("text_hash"),
            "state_key": f"{cell['query_id']}-{cell['carrier_id']}__{method}__r{rnd}",
            "delta": delta.detach().cpu().half().clone(),
        }
        records.append(row)
        if rhc:
            win_round = rnd
            try:
                from run_p0_qwen import seed_all

                seed_all(int(eval_seed))
            except Exception:
                pass
            tr_eval = official_decode(
                wrapper, item, x, image, max_new_tokens=int(budget["max_new_tokens"])
            )
            ledger.add_generation(
                tokens=int(tr_eval["n_tokens"]), wall_s=float(tr_eval.get("wall_seconds") or 0)
            )
            robust = bool(tr_eval.get("core_rhc"))
            records[-1]["eval_seed"] = int(eval_seed)
            records[-1]["robust_core_rhc"] = robust
            break
        C = update_certificate_set(
            C, cert, method=method, max_set=int(cert_cfg["max_set"])
        )
        records[-1]["n_certificates"] = len(C)
        before_mode = after_mode

    return {
        "cell_id": f"{cell['query_id']}:{cell['carrier_id']}",
        "method": method,
        "core_rhc": bool(win_round is not None),
        "win_round": win_round,
        "robust_core_rhc": robust,
        "clean_mode": clean_mode,
        "n_certificates": len(C),
        "certificate_modes": [c["mode"] for c in C],
        "delta_linf": float(delta.detach().abs().max().item()),
        "budget": ledger.as_dict(),
        "wall_seconds": float(time.time() - t0),
        "rounds": records,
        "opt_seed": int(opt_seed),
        "eval_seed": int(eval_seed),
    }

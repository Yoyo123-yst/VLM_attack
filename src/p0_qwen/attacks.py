"""Per-sample visual PGD on Qwen. No universal delta."""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from p0.judge import benign_utility, harmful_compliance
from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list
from p0.subspace import state_shift

from .model import QwenP0
from .vision import eot_scale, tv_loss


def pgd_one(
    wrapper: QwenP0,
    img,
    query: str,
    steps: int,
    eps: float,
    alpha: float,
    loss_fn: Callable,
    tv_weight: float,
    eot: Tuple[float, float],
) -> torch.Tensor:
    x0 = wrapper.image_to_x01(img)
    packed = wrapper.encode(img, query)
    ids, attn = packed["input_ids"], packed.get("attention_mask")
    delta = torch.zeros_like(x0)
    wrapper.model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Gradient checkpointing on 4-bit Qwen increased fragmentation and OOM'd
    # the first PGD step; keep a vanilla autograd.grad on delta only.
    try:
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            x = torch.clamp(x0 + delta, 0.0, 1.0)
            x_e = eot_scale(x, eot[0], eot[1])
            pv, grid = wrapper.patchify(x_e)
            loss = loss_fn(pv, ids, attn, grid) + tv_weight * tv_loss(x)
            grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
            if grad is None:
                raise RuntimeError("attack got no image gradient")
            delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
            delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
            del loss, grad, pv, x, x_e
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return delta.detach()
    finally:
        try:
            wrapper.model.gradient_checkpointing_disable()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def refusal_loss_fn(wrapper: QwenP0, refuse_ids, comply_ids):
    def fn(pv, ids, attn, grid):
        logits = wrapper.first_token_logits(pv, ids, attn, grid)
        return margin_from_logits(logits, refuse_ids, comply_ids)

    return fn


def state_loss_fn(wrapper: QwenP0, layer: int, U: np.ndarray, mu_ref: np.ndarray, mu_jb: np.ndarray):
    u = torch.from_numpy(U.astype(np.float32)).to(wrapper.device)
    mref = torch.from_numpy(mu_ref.astype(np.float32)).to(wrapper.device)
    mjb = torch.from_numpy(mu_jb.astype(np.float32)).to(wrapper.device)

    def fn(pv, ids, attn, grid):
        h = wrapper.hidden_last_user(pv, ids, layer, attn, grid).float()
        z = h @ u
        return torch.norm(z - mjb) - torch.norm(z - mref)

    return fn


def screen_attack_clean(
    wrapper: QwenP0,
    queries: List[Dict[str, Any]],
    img,
) -> Dict[str, Any]:
    """Clean (no-δ) labels for attack-test queries. ASR denominator is REF only."""
    rows = []
    for q in tqdm(queries, desc="attack-clean-ref"):
        text = wrapper.generate(img, q["query"])
        j = harmful_compliance(text, q["topic_keywords"], q["query"])
        label = j.get("label")
        eligible = label == "REF"
        rows.append(
            {
                "query_id": q["id"],
                "clean_label": label,
                "clean_refusal": float(j.get("refusal") or 0.0),
                "clean_relevance": float(j.get("relevance") or 0.0),
                "eligible_for_attack": bool(eligible),
                "text": (j.get("text") or "")[:400],
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    n_ref = sum(1 for r in rows if r["eligible_for_attack"])
    n_rhc = sum(1 for r in rows if r["clean_label"] in {"RHC", "JB"})
    n_related = sum(1 for r in rows if r["clean_label"] == "RELATED_SAFE")
    n_fail = sum(1 for r in rows if r["clean_label"] == "FAIL")
    return {
        "split": "attack_test",
        "n_catalog": len(rows),
        "n_eligible": n_ref,
        "n_clean_rhc": n_rhc,
        "n_related_safe": n_related,
        "n_fail": n_fail,
        "eligible_ids": [r["query_id"] for r in rows if r["eligible_for_attack"]],
        "rows": rows,
    }


def evaluate_one(
    wrapper: QwenP0,
    img,
    q: Dict[str, Any],
    x01: Optional[torch.Tensor],
    kind: str,
    layer: Optional[int],
    U: Optional[np.ndarray],
    mu_ref: Optional[np.ndarray],
    mu_jb: Optional[np.ndarray],
) -> Dict[str, Any]:
    text = wrapper.generate(img, q["query"], x01=x01)
    if kind == "harmful":
        j = harmful_compliance(text, q["topic_keywords"], q["query"])
    else:
        j = benign_utility(text, q["topic_keywords"], q["query"])
    shift = None
    if layer is not None and U is not None and mu_ref is not None and mu_jb is not None:
        hid = wrapper.collect_hidden(img, q["query"], [layer], x01=x01)
        h = hid[f"L{layer}:last_user"].numpy()
        shift = state_shift(h, U, mu_ref, mu_jb)
    return {"query_id": q["id"], "judge": j, "state_shift": shift}


def run_per_sample(
    wrapper: QwenP0,
    cfg: Dict[str, Any],
    queries: List[Dict[str, Any]],
    benign_queries: List[Dict[str, Any]],
    img,
    layer: int,
    methods: Dict[str, Callable],
    U: np.ndarray,
    mu_ref: np.ndarray,
    mu_jb: np.ndarray,
    existing: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    steps = int(cfg["attack"]["attack_steps"])
    alpha = float(cfg["attack"]["alpha"])
    tvw = float(cfg["attack"]["tv_weight"])
    eot = tuple(cfg["attack"]["eot_scale"])
    eps_list = [float(x) for x in cfg["attack"]["eps_list"]]
    out: Dict[str, Any] = existing or {"layer": layer, "eps": {}, "benign_clean": []}
    out.setdefault("eps", {})
    out["layer"] = layer
    if not out.get("benign_clean"):
        out["benign_clean"] = []
        for bq in tqdm(benign_queries, desc="benign-clean"):
            out["benign_clean"].append(evaluate_one(wrapper, img, bq, None, "benign", layer, U, mu_ref, mu_jb))
        if on_progress:
            on_progress(out)

    for eps in eps_list:
        tag = f"{int(round(eps * 255))}/255"
        rec_eps = out["eps"].setdefault(tag, {"methods": {}})
        rec_eps.setdefault("methods", {})
        for name, fn in methods.items():
            done = rec_eps["methods"].get(name) or {}
            if done.get("harmful") and done.get("benign"):
                print(f"reuse {name} eps={tag}", flush=True)
                continue
            rows = []
            benign_rows = []
            for q in tqdm(queries, desc=f"{name} eps={tag}"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                delta = pgd_one(wrapper, img, q["query"], steps, eps, alpha, fn, tvw, eot)
                x01 = torch.clamp(wrapper.image_to_x01(img) + delta, 0.0, 1.0)
                rows.append(evaluate_one(wrapper, img, q, x01, "harmful", layer, U, mu_ref, mu_jb))
            for bq in tqdm(benign_queries, desc=f"{name} benign eps={tag}"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                delta = pgd_one(wrapper, img, bq["query"], steps, eps, alpha, fn, tvw, eot)
                x01 = torch.clamp(wrapper.image_to_x01(img) + delta, 0.0, 1.0)
                benign_rows.append(evaluate_one(wrapper, img, bq, x01, "benign", layer, U, mu_ref, mu_jb))
            rec_eps["methods"][name] = {"harmful": rows, "benign": benign_rows}
            if on_progress:
                on_progress(out)
        if "clean" not in rec_eps["methods"]:
            rec_eps["methods"]["clean"] = {
                "harmful": [
                    evaluate_one(wrapper, img, q, None, "harmful", layer, U, mu_ref, mu_jb) for q in queries
                ],
                "benign": out["benign_clean"],
            }
            if on_progress:
                on_progress(out)
        out["eps"][tag] = rec_eps
    return out


def default_methods(wrapper: QwenP0, layer: int, U, mu_ref, mu_jb, U_rand, U_fail, mu_fail):
    refuse_ids = token_id_list(wrapper.processor, REFUSE_WORDS)
    comply_ids = token_id_list(wrapper.processor, COMPLY_WORDS)
    methods = {
        "refusal_margin_pgd": refusal_loss_fn(wrapper, refuse_ids, comply_ids),
        "u_guided": state_loss_fn(wrapper, layer, U, mu_ref, mu_jb),
        "random_subspace": state_loss_fn(wrapper, layer, U_rand, mu_ref, mu_jb),
    }
    if U_fail is not None and mu_fail is not None:
        methods["fail_subspace"] = state_loss_fn(wrapper, layer, U_fail, mu_ref, mu_fail)
    return methods

"""P1: quota poisoning + eviction-set PGD on auxiliary image B."""

from __future__ import annotations

import gc
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .compressor import exchange_events, restore_u, restore_victim, summarize_keep
from .config import out_dir
from .io import load_json, save_json
from .losses import caa_b_loss, cage_b_loss, combined_loss, lamp_like_loss
from .mechlog import delta_a_is_zero, linf_ok, margin_crossing, mechanism_snapshot, quota_conserved
from .model import QwenMultiImage, load_pair_images, load_sample_images
from .probe import _ok, _select
from .vision import clip_delta, tv_loss, x01_to_pil


def _gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _empty() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _select_cfg(scores, owner, n_per, cfg, mode: str):
    return _select(scores, owner, n_per, cfg, mode)


def _metrics_from_scores(scores, owner, n_per, cfg, keep_clean, mode: str = "avtp") -> Dict[str, Any]:
    quota, keep = _select_cfg(scores, owner, n_per, cfg, mode)
    attacker = 1 if int(owner.max().item()) <= 1 else None
    ev = exchange_events(keep_clean, keep, owner, attacker=attacker)
    return {
        "r": [float(x) for x in quota.r.detach().cpu().tolist()],
        "k": [int(x) for x in quota.k.detach().cpu().tolist()],
        "i_bar": [float(x) for x in quota.i_bar.detach().cpu().tolist()],
        "kept": summarize_keep(keep, owner, n_images=len(n_per)),
        "events": ev,
        "keep": keep.detach(),
    }


def _J(ev: Dict[str, int], r: List[float], u_out: int = 0, w1: float = 2.0, w2: float = 2.0, w3: float = 8.0, w4: float = 10.0) -> float:
    r_aux = (sum(r[1:]) / max(len(r) - 1, 1)) if len(r) > 1 else 0.0
    return float(w1 * ev["a_out"] + w2 * ev["b_in"] + w3 * (r_aux - r[0]) + w4 * u_out)


def _attack_aux_idx(state_or_cfg, n_aux: int) -> List[int]:
    if isinstance(state_or_cfg, dict) and "attack" in state_or_cfg:
        perturb = str(state_or_cfg.get("attack", {}).get("perturb") or "single")
    else:
        perturb = "single"
    if perturb == "multi":
        return list(range(max(int(n_aux), 1)))
    return [0]


def _perturbed_xs(state: Dict[str, Any], delta) -> List:
    aux = list(state.get("x_aux") or [state["xB0"]])
    attack_aux = list(state.get("attack_aux") or [0])
    xs = [state["xA"]]
    if torch.is_tensor(delta):
        if delta.dim() == aux[0].dim() + 1:
            dmap = {int(i): delta[j] for j, i in enumerate(attack_aux)}
        else:
            dmap = {int(attack_aux[0]): delta}
    else:
        dmap = {int(i): d for i, d in zip(attack_aux, delta)}
    for i, x0 in enumerate(aux):
        if i in dmap:
            xs.append(torch.clamp(x0 + dmap[i], 0.0, 1.0))
        else:
            xs.append(x0)
    return xs


def _tv_attacked(xs: List, attack_aux: List[int]):
    parts = [xs[1 + i] for i in attack_aux if 1 + i < len(xs)]
    if not parts:
        return xs[1]
    return parts[0]


def _log_prefix(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("attack", {}).get("log_prefix", "p1"))


def _result_path(cfg: Dict[str, Any], key: str, default: str) -> Path:
    raw = cfg.get("attack", {}).get(key)
    return Path(raw) if raw else (out_dir(cfg) / default)


def prepare_sample(wrapper: QwenMultiImage, pair: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    images = load_sample_images(pair)
    packed = wrapper.pack(images, pair["prompt"])
    grid = packed.tensors["image_grid_thw"]
    xs = [wrapper.x01_for_grid(im, grid[i]).detach() for i, im in enumerate(images)]
    xA, x_aux = xs[0], xs[1:]
    xB0 = x_aux[0]
    pv = wrapper.pixels_from_xs(xs).detach()
    packed = wrapper.with_pixels(packed, pv)
    n_from_pv = int(pv.shape[0]) // (2 * 2)
    if n_from_pv != sum(packed.n_per_image):
        raise RuntimeError(
            f"patchify tokens {n_from_pv} != packed visual {sum(packed.n_per_image)}"
        )
    owner = wrapper.owner(packed)
    with torch.no_grad():
        scores, h_clean, out = wrapper._forward_hidden(packed, pixel_values=pv)
        del out
    mode = str(cfg.get("compressor", {}).get("mode") or "avtp")
    if mode not in {"avtp", "global", "isolated", "query", "query_aware", "feather", "spatial"}:
        mode = "avtp"
    sel_mode = "avtp" if mode in {"query", "query_aware"} else mode
    quota, keep = _select_cfg(scores, owner, packed.n_per_image, cfg, sel_mode)
    u_idx = pair.get("u_idx")
    if u_idx:
        u_mask = torch.zeros_like(keep)
        u_mask[torch.tensor(u_idx, device=keep.device, dtype=torch.long)] = True
    else:
        # Default: highest-scoring A survivors (task evidence, not the first-to-drop tail).
        a_keep = torch.nonzero((owner == 0) & keep, as_tuple=False).flatten()
        n_top = max(1, int(round(0.25 * int(a_keep.numel()))))
        if a_keep.numel() == 0:
            u_mask = (owner == 0) & keep
        else:
            top = torch.topk(scores[a_keep], k=min(n_top, int(a_keep.numel())), largest=True).indices
            u_mask = torch.zeros_like(keep)
            u_mask[a_keep[top]] = True
    v_mask = owner >= 1
    gold = pair.get("ans_a_only") or pair.get("screen", {}).get("ans_a_only")
    attack_aux = _attack_aux_idx(cfg, n_aux=len(x_aux))
    mech_clean = mechanism_snapshot(
        scores, owner, packed.n_per_image, keep, keep, u_mask, quota.r, quota.k, quota.i_bar
    )
    return {
        "pair": pair,
        "packed": packed,
        "xA": xA,
        "xB0": xB0,
        "x_aux": x_aux,
        "attack_aux": attack_aux,
        "owner": owner,
        "keep_clean": keep.detach(),
        "u_mask": u_mask.detach(),
        "v_mask": v_mask.detach(),
        "h_clean": h_clean.detach(),
        "scores_clean": scores.detach(),
        "gold": gold,
        "r_clean": [float(x) for x in quota.r.detach().cpu().tolist()],
        "k_clean": [int(x) for x in quota.k.detach().cpu().tolist()],
        "i_bar_clean": [float(x) for x in quota.i_bar.detach().cpu().tolist()],
        "n_u": int(u_mask.sum().item()),
        "gold_ids": _gold_ids(wrapper, gold),
        "mech_clean": mech_clean,
        "delta_a_zero": True,
    }


def _gold_ids(wrapper: QwenMultiImage, gold: Optional[str]) -> torch.Tensor:
    tok = wrapper.processor.tokenizer
    text = (gold or "yes").strip() or "yes"
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if ids.numel() == 0:
        ids = torch.tensor([int(tok.eos_token_id or 0)], dtype=torch.long)
    return ids.to(device=wrapper.device)[:8]


def _aux_from_scores(scores, owner, n_per, cfg, u_mask, v_mask, h_adv, h_clean, xB):
    _loss, aux = combined_loss(
        scores=scores,
        owner=owner,
        n_per=n_per,
        u_mask=u_mask,
        v_mask=v_mask,
        h_adv=h_adv,
        h_clean=h_clean,
        xB=xB,
        r_base=float(cfg["compressor"]["r_base"]),
        alpha=float(cfg["compressor"]["alpha"]),
        r_min=float(cfg["compressor"]["r_min"]),
        r_max=float(cfg["compressor"]["r_max"]),
        lambda_q=0.0,
        lambda_e=0.0,
        lambda_v=0.0,
        lambda_p=0.0,
        kappa=float(cfg["attack"]["kappa"]),
        tau=float(cfg["attack"]["tau"]),
        tv_fn=tv_loss,
        lambda_c=0.0,
    )
    return aux


def _task_loss(wrapper, packed, pixel_values, gold_ids):
    """Untargeted Task-PGD: increase CE of teacher-forced gold tokens (PGD minimizes loss)."""
    from .compressor import layer_variation_score

    ids0 = packed.tensors["input_ids"]
    attn0 = packed.tensors.get("attention_mask", torch.ones_like(ids0))
    g = gold_ids.to(ids0.device).view(1, -1)
    sl = int(ids0.shape[1])
    if int(g.shape[1]) > 1:
        prefix = g[:, :-1]
        tensors = {
            **packed.tensors,
            "input_ids": torch.cat([ids0, prefix], dim=1),
            "attention_mask": torch.cat([attn0, torch.ones_like(prefix)], dim=1),
            "pixel_values": pixel_values,
        }
    else:
        tensors = {**packed.tensors, "pixel_values": pixel_values}
    out = wrapper.model(**tensors, output_hidden_states=True, use_cache=False)
    n = int(g.shape[1])
    logits = out.logits[0, sl - 1 : sl - 1 + n].float()
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target = g.view(-1)[: logits.shape[0]]
    # PGD descends. Untargeted Task-PGD must RAISE CE / lower p(gold).
    loss = -torch.nn.functional.cross_entropy(logits, target)
    captured = {i: out.hidden_states[i][0] for i in range(len(out.hidden_states))}
    seq_score = layer_variation_score(captured, wrapper.score_layers)
    scores = seq_score[packed.vis_index]
    h_adv = out.hidden_states[-1][0, -1]
    del out
    return loss, scores, h_adv


def _forward_loss(wrapper, state, delta, cfg):
    atk = cfg["attack"]
    kind = str(atk.get("loss_kind", "vcache"))
    xs = _perturbed_xs(state, delta)
    xB = _tv_attacked(xs, list(state.get("attack_aux") or [0]))
    pv = wrapper.pixels_from_xs(xs)
    if kind == "task":
        loss, scores, h_adv = _task_loss(wrapper, state["packed"], pv, state["gold_ids"])
        aux = _aux_from_scores(
            scores, state["owner"], state["packed"].n_per_image, cfg,
            state["u_mask"], state["v_mask"], h_adv, state["h_clean"], xB,
        )
        aux["L_task"] = loss.detach()
        return loss, aux, scores, xB, pv
    if kind in {"lamp", "lamp_like"}:
        task, scores, h_adv = _task_loss(wrapper, state["packed"], pv, state["gold_ids"])
        loss = task + lamp_like_loss(scores, state["owner"], h_adv, state["h_clean"])
        aux = _aux_from_scores(
            scores, state["owner"], state["packed"].n_per_image, cfg,
            state["u_mask"], state["v_mask"], h_adv, state["h_clean"], xB,
        )
        aux["L_lamp"] = loss.detach()
        aux["L_task"] = task.detach()
        return loss, aux, scores, xB, pv

    scores, h_adv = wrapper.visual_scores_grad(state["packed"], pv)
    if kind == "caa":
        loss = caa_b_loss(scores, state["owner"], state["keep_clean"])
        aux = _aux_from_scores(
            scores, state["owner"], state["packed"].n_per_image, cfg,
            state["u_mask"], state["v_mask"], h_adv, state["h_clean"], xB,
        )
        aux["L_caa"] = loss.detach()
        return loss, aux, scores, xB, pv
    if kind == "cage":
        loss = cage_b_loss(scores, state["owner"], state["keep_clean"])
        aux = _aux_from_scores(
            scores, state["owner"], state["packed"].n_per_image, cfg,
            state["u_mask"], state["v_mask"], h_adv, state["h_clean"], xB,
        )
        aux["L_cage"] = loss.detach()
        return loss, aux, scores, xB, pv
    if kind == "rank":
        loss, aux = combined_loss(
            scores=scores,
            owner=state["owner"],
            n_per=state["packed"].n_per_image,
            u_mask=state["u_mask"],
            v_mask=state["v_mask"],
            h_adv=h_adv,
            h_clean=state["h_clean"],
            xB=xB,
            r_base=float(cfg["compressor"]["r_base"]),
            alpha=float(cfg["compressor"]["alpha"]),
            r_min=float(cfg["compressor"]["r_min"]),
            r_max=float(cfg["compressor"]["r_max"]),
            lambda_q=float(atk["lambda_q"]),
            lambda_e=float(atk["lambda_e"]),
            lambda_v=float(atk.get("lambda_v", 0.0)),
            lambda_p=float(atk.get("lambda_p", 0.0)),
            kappa=float(atk["kappa"]),
            tau=float(atk["tau"]),
            tv_fn=tv_loss,
            lambda_c=0.0,
        )
        return loss, aux, scores, xB, pv
    loss, aux = combined_loss(
        scores=scores,
        owner=state["owner"],
        n_per=state["packed"].n_per_image,
        u_mask=state["u_mask"],
        v_mask=state["v_mask"],
        h_adv=h_adv,
        h_clean=state["h_clean"],
        xB=xB,
        r_base=float(cfg["compressor"]["r_base"]),
        alpha=float(cfg["compressor"]["alpha"]),
        r_min=float(cfg["compressor"]["r_min"]),
        r_max=float(cfg["compressor"]["r_max"]),
        lambda_q=float(atk["lambda_q"]),
        lambda_e=float(atk["lambda_e"]),
        lambda_v=float(atk["lambda_v"]),
        lambda_p=float(atk["lambda_p"]),
        kappa=float(atk["kappa"]),
        tau=float(atk["tau"]),
        tv_fn=tv_loss,
        lambda_c=float(atk.get("lambda_c", 0.0)),
    )
    return loss, aux, scores, xB, pv


def _log_step(pair_id: str, step: int, steps: int, loss, aux, ev, elapsed: float, u_out: int = 0, prefix: str = "p1") -> Dict[str, Any]:
    r = [float(x) for x in aux["r"].detach().cpu().tolist()]
    row = {
        "step": step,
        "loss": float(loss.detach().cpu()),
        "L_quota": float(aux["L_quota"].detach().cpu()),
        "L_evict": float(aux["L_evict"].detach().cpu()),
        "L_crit": float(aux["L_crit"].detach().cpu()) if "L_crit" in aux else None,
        "L_value": float(aux["L_value"].detach().cpu()),
        "L_perc": float(aux["L_perc"].detach().cpu()),
        "r": r,
        "k": [int(x) for x in aux["k"].detach().cpu().tolist()],
        "a_out": ev["a_out"],
        "b_in": ev["b_in"],
        "u_out": u_out,
        "swap": ev["swap_ba"],
        "J": _J(ev, r, u_out=u_out),
        "elapsed_s": elapsed,
        "mem_gb": _gpu_mem_gb(),
    }
    print(
        f"{prefix} {pair_id} {step}/{steps} L={row['loss']:.3f} "
        f"Lq={row['L_quota']:+.4f} Le={row['L_evict']:.3f} "
        f"rA={r[0]:.3f} rB={sum(r[1:]) / max(len(r) - 1, 1):.3f} a_out={ev['a_out']} u_out={u_out} "
        f"mem={row['mem_gb']}",
        flush=True,
    )
    return row


def _delta_leaf(state: Dict[str, Any]) -> torch.Tensor:
    attack_aux = list(state.get("attack_aux") or [0])
    if len(attack_aux) > 1:
        return torch.zeros_like(torch.stack([state["x_aux"][i] for i in attack_aux]))
    return torch.zeros_like(state["xB0"])


def _delta_anchor(state: Dict[str, Any], delta: torch.Tensor) -> torch.Tensor:
    attack_aux = list(state.get("attack_aux") or [0])
    if torch.is_tensor(delta) and delta.dim() == state["xB0"].dim() + 1:
        return torch.stack([state["x_aux"][i] for i in attack_aux])
    return state["xB0"]


def exchange_search(wrapper, state, delta, cfg) -> torch.Tensor:
    atk = cfg["attack"]
    eps = float(atk["eps"])
    scales = [float(s) for s in atk.get("search_scales", [0.5, 0.75, 1.0, 1.25, 1.5])]
    x0 = _delta_anchor(state, delta if torch.is_tensor(delta) else delta[0])
    d0 = delta.detach() if torch.is_tensor(delta) else torch.stack([d.detach() for d in delta])
    best = clip_delta(x0, d0, eps)
    best_j = -1e9
    linf = float(best.abs().amax().clamp(min=1e-8))
    cands = [clip_delta(x0, best * s, eps) for s in scales]
    cands.append(clip_delta(x0, best * (eps / linf), eps))
    with torch.no_grad():
        for cand in cands:
            xs = _perturbed_xs(state, cand)
            pv = wrapper.pixels_from_xs(xs)
            scores = wrapper.visual_scores(wrapper.with_pixels(state["packed"], pv))
            met = _metrics_from_scores(scores, state["owner"], state["packed"].n_per_image, cfg, state["keep_clean"])
            u_out = int((state["u_mask"].to(met["keep"].device) & ~met["keep"]).sum().item())
            j = _J(met["events"], met["r"], u_out=u_out)
            if j > best_j:
                best_j = j
                best = cand
    return best


def evaluate_final(wrapper, state, delta, cfg) -> Dict[str, Any]:
    pair = state["pair"]
    gold = state["gold"]
    xs = _perturbed_xs(state, delta)
    pv = wrapper.pixels_from_xs(xs).detach()
    packed = wrapper.with_pixels(state["packed"], pv)
    scores = wrapper.visual_scores(packed)
    owner = state["owner"]
    n_per = packed.n_per_image
    avtp = _metrics_from_scores(scores, owner, n_per, cfg, state["keep_clean"], "avtp")
    glob = _metrics_from_scores(scores, owner, n_per, cfg, state["keep_clean"], "global")
    ans_full = wrapper.generate(packed, keep_visual=None)
    ans_avtp = wrapper.generate(packed, keep_visual=avtp["keep"])
    keep_iso = _select_cfg(scores, owner, n_per, cfg, "isolated")[1]
    ans_iso = wrapper.generate(packed, keep_visual=keep_iso)
    keep_restore = restore_victim(avtp["keep"], state["keep_clean"], owner, victim=0)
    ans_restore = wrapper.generate(packed, keep_visual=keep_restore)
    u_mask = state["u_mask"].to(avtp["keep"].device)
    keep_restore_u = restore_u(avtp["keep"], u_mask)
    ans_restore_u = wrapper.generate(packed, keep_visual=keep_restore_u)
    full_ok = _ok(ans_full, pair, gold=gold)
    avtp_ok = _ok(ans_avtp, pair, gold=gold)
    iso_ok = _ok(ans_iso, pair, gold=gold)
    restore_a_ok = _ok(ans_restore, pair, gold=gold)
    restore_u_ok = _ok(ans_restore_u, pair, gold=gold)
    u_out = int((u_mask & ~avtp["keep"]).sum().item())
    n_u = int(u_mask.sum().item())
    r_t = torch.tensor(avtp["r"], dtype=torch.float32)
    k_t = torch.tensor(avtp["k"], dtype=torch.long)
    i_t = torch.tensor(avtp["i_bar"], dtype=torch.float32)
    mech_adv = mechanism_snapshot(
        scores, owner, n_per, avtp["keep"], state["keep_clean"], u_mask, r_t, k_t, i_t
    )
    mech_clean = state.get("mech_clean")
    if mech_clean is None and state.get("scores_clean") is not None:
        mech_clean = mechanism_snapshot(
            state["scores_clean"],
            owner,
            n_per,
            state["keep_clean"],
            state["keep_clean"],
            u_mask,
            torch.tensor(state["r_clean"], dtype=torch.float32),
            torch.tensor(state["k_clean"], dtype=torch.long),
        )
    d_att = delta if torch.is_tensor(delta) else torch.stack(list(delta))
    eps = float(cfg.get("attack", {}).get("eps") or 16 / 255)
    r_base = float(cfg["compressor"]["r_base"])
    family = {}
    if bool(cfg.get("compressor", {}).get("eval_family")):
        for mode in ("avtp", "global", "isolated", "feather"):
            q, keep_m = _select_cfg(scores, owner, n_per, {**cfg, "compressor": {**cfg["compressor"], "match_k_total": True}}, mode)
            ans_m = wrapper.generate(packed, keep_visual=keep_m)
            family[mode] = {
                "k": [int(x) for x in q.k.detach().cpu().tolist()],
                "k_sum": int(keep_m.sum().item()),
                "ok": bool(_ok(ans_m, pair, gold=gold)),
            }
    return {
        "ans_full": ans_full,
        "ans_avtp": ans_avtp,
        "ans_isolated": ans_iso,
        "ans_restore_a": ans_restore,
        "ans_restore_u": ans_restore_u,
        "full_ok": bool(full_ok),
        "avtp_ok": bool(avtp_ok),
        "isolated_ok": bool(iso_ok),
        "restore_a_ok": bool(restore_a_ok),
        "restore_u_ok": bool(restore_u_ok),
        "avtp": {k: v for k, v in avtp.items() if k != "keep"},
        "global": {k: v for k, v in glob.items() if k != "keep"},
        "linf": float(d_att.detach().abs().amax().cpu()),
        "linf_ok": linf_ok(d_att, eps),
        "delta_a_zero": delta_a_is_zero(state["xA"], xs[0]),
        "quota_sum_ok": quota_conserved(r_t, r_base),
        "mech_clean": mech_clean,
        "mech_adv": mech_adv,
        "margin_cross": margin_crossing(mech_clean or {}, mech_adv),
        "family": family,
        "comp_only_fail": bool(full_ok and not avtp_ok),
        "n_u": n_u,
        "u_out": u_out,
        "u_evict_rate": (u_out / n_u) if n_u else 0.0,
    }


def pgd_one(
    wrapper: QwenMultiImage,
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    steps: Optional[int] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    atk = cfg["attack"]
    eps = float(atk["eps"])
    alpha = float(atk["alpha"])
    steps = int(steps if steps is not None else atk["steps"])
    search_every = int(atk.get("search_every", 10))
    pair_id = state["pair"]["pair_id"]
    delta = _delta_leaf(state)
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    oom_fallback = False
    step = 0
    while step < steps:
        step += 1
        delta = delta.detach().requires_grad_(True)
        try:
            with torch.enable_grad():
                loss, aux, scores, _xB, _pv = _forward_loss(wrapper, state, delta, cfg)
            grad = torch.autograd.grad(loss, delta, allow_unused=True)[0]
        except torch.cuda.OutOfMemoryError:
            _empty()
            if oom_fallback:
                raise
            oom_fallback = True
            print(f"{_log_prefix(cfg)} {pair_id} OOM at step {step}; retry with score_layers=[1]", flush=True)
            wrapper.score_layers = [1]
            step -= 1
            continue
        if grad is None:
            raise RuntimeError(f"{pair_id}: no image gradient")
        gnorm = float(grad.detach().float().abs().mean().cpu())
        delta = clip_delta(_delta_anchor(state, delta.detach()), delta.detach() - alpha * grad.sign(), eps)
        if search_every > 0 and step % search_every == 0:
            delta = exchange_search(wrapper, state, delta, cfg)

        with torch.no_grad():
            met = _metrics_from_scores(
                scores.detach(), state["owner"], state["packed"].n_per_image, cfg, state["keep_clean"]
            )
            ev = met["events"]
            u_out = int((state["u_mask"].to(met["keep"].device) & ~met["keep"]).sum().item())
        row = _log_step(
            pair_id, step, steps, loss, aux, ev, time.time() - t0,
            u_out=u_out, prefix=_log_prefix(cfg),
        )
        row["grad_abs_mean"] = gnorm
        row["oom_fallback_layer1"] = oom_fallback
        history.append(row)
        if progress_cb is not None:
            progress_cb(row)
        del loss, aux, scores, grad
        _empty()

    if search_every > 0:
        delta = exchange_search(wrapper, state, delta, cfg)
    evaled = evaluate_final(wrapper, state, delta, cfg)
    return {
        "pair_id": pair_id,
        "method": str(atk.get("loss_kind", "vcache")),
        "steps": steps,
        "history": history,
        "eval": evaled,
        "delta_linf": float(delta.abs().amax().cpu()),
        "oom_fallback_layer1": oom_fallback,
        "elapsed_s": time.time() - t0,
        "delta": delta.detach().cpu(),
        "xB_hw": list(state["xB0"].shape),
    }


def _load_kept(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = _result_path(cfg, "screen_path", "screen.json")
    if not path.is_file():
        raise FileNotFoundError(f"need screen json at {path}")
    kept = load_json(path)["kept"]
    if not kept:
        raise RuntimeError("screen kept zero pairs")
    return kept


def _write_progress(cfg: Dict[str, Any], blob: Dict[str, Any]) -> Path:
    dest = Path(cfg["attack"]["progress_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json(dest, blob)
    return dest


def _eta(done: int, total: int, elapsed: float) -> Optional[float]:
    if done <= 0:
        return None
    return (elapsed / done) * (total - done)


def stage_p1_smoke(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kept = _load_kept(cfg)[: int(cfg["attack"].get("n_smoke", 1))]
    Path(cfg["attack"]["delta_dir"]).mkdir(parents=True, exist_ok=True)
    wrapper = QwenMultiImage(cfg)
    rows = []
    t0 = time.time()
    for pair in kept:
        print(f"== {_log_prefix(cfg)} smoke {pair['pair_id']} ==", flush=True)
        state = prepare_sample(wrapper, pair, cfg)

        def _cb(row):
            _write_progress(
                cfg,
                {
                    "stage": "smoke",
                    "updated_at": _now(),
                    "pair_id": pair["pair_id"],
                    "step": row["step"],
                    "last": row,
                    "elapsed_s": time.time() - t0,
                },
            )

        out = pgd_one(wrapper, state, cfg, steps=int(cfg["attack"]["smoke_steps"]), progress_cb=_cb)
        delta = out.pop("delta")
        torch.save(delta, Path(cfg["attack"]["delta_dir"]) / f"{pair['pair_id']}_smoke.pt")
        ev = out["eval"]["avtp"]["events"]
        rows.append({k: v for k, v in out.items()})
        _write_progress(
            cfg,
            {
                "stage": "smoke",
                "updated_at": _now(),
                "elapsed_s": time.time() - t0,
                "n": len(rows),
                "last": {"pair_id": pair["pair_id"], "a_out": ev["a_out"], "b_in": ev["b_in"], "full_ok": out["eval"]["full_ok"]},
                "rows": rows,
            },
        )
        _empty()
    dest = _result_path(cfg, "smoke_path", "p1_smoke.json")
    save_json(dest, {"n": len(rows), "elapsed_s": time.time() - t0, "rows": rows})
    return {"n": len(rows), "path": str(dest), "elapsed_s": time.time() - t0}


def stage_p1_attack(cfg: Dict[str, Any], limit: Optional[int] = None, wrapper: Optional[QwenMultiImage] = None) -> Dict[str, Any]:
    kept = _load_kept(cfg)
    if limit is not None:
        kept = kept[: int(limit)]
    dest = _result_path(cfg, "result_path", "p1_attack.json")
    done_ids = set()
    rows: List[Dict[str, Any]] = []
    if dest.is_file():
        prev = load_json(dest)
        prev_rows = list(prev.get("rows") or [])
        rows = [r for r in prev_rows if r.get("eval") and not r.get("error")]
        done_ids = {r["pair_id"] for r in rows}
        print(f"resume {len(done_ids)} finished samples", flush=True)
    pending = [p for p in kept if p["pair_id"] not in done_ids]
    if wrapper is None:
        wrapper = QwenMultiImage(cfg)
    t0 = time.time()
    total = len(kept)
    already = len(done_ids)
    Path(cfg["attack"]["delta_dir"]).mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pending, start=1):
        print(f"== {_log_prefix(cfg)} attack {pair['pair_id']} ({already + i}/{total}) ==", flush=True)
        sample_t0 = time.time()
        state = None
        try:
            state = prepare_sample(wrapper, pair, cfg)

            def _cb(row, _pair=pair, _done=already + i, _total=total, _t0=t0):
                eta = _eta(_done - already, max(len(pending), 1), time.time() - _t0)
                _write_progress(
                    cfg,
                    {
                        "stage": "attack",
                        "updated_at": _now(),
                        "pair_id": _pair["pair_id"],
                        "done": _done - 1,
                        "total": _total,
                        "step": row["step"],
                        "eta_s": eta,
                        "last": row,
                    },
                )

            out = pgd_one(wrapper, state, cfg, progress_cb=_cb)
            delta = out.pop("delta")
            torch.save(delta, Path(cfg["attack"]["delta_dir"]) / f"{pair['pair_id']}.pt")
            out["error"] = None
        except Exception as exc:
            print(f"{_log_prefix(cfg)} {pair['pair_id']} FAILED {type(exc).__name__}: {exc}", flush=True)
            out = {
                "pair_id": pair["pair_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "eval": None,
                "elapsed_s": time.time() - sample_t0,
            }
            _empty()
        rows.append(out)
        elapsed = time.time() - t0
        done = already + i
        eta = _eta(i, len(pending), elapsed)
        blob = {
            "stage": "attack",
            "updated_at": _now(),
            "done": done,
            "total": total,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "last": {
                "pair_id": pair["pair_id"],
                "error": out.get("error"),
                "eval": None if out.get("eval") is None else {
                    "full_ok": out["eval"]["full_ok"],
                    "avtp_ok": out["eval"]["avtp_ok"],
                    "comp_only_fail": out["eval"]["comp_only_fail"],
                    "a_out": out["eval"]["avtp"]["events"]["a_out"],
                    "b_in": out["eval"]["avtp"]["events"]["b_in"],
                    "u_out": out["eval"].get("u_out"),
                    "n_u": out["eval"].get("n_u"),
                    "delta_r": [
                        out["eval"]["avtp"]["r"][j] - state["r_clean"][j]
                        for j in range(2)
                    ]
                    if out.get("eval") is not None
                    else None,
                },
            },
            "rows": rows,
        }
        # drop keep tensors already stripped
        save_json(dest, {"n": len(rows), "elapsed_s": elapsed, "rows": _strip(rows)})
        _write_progress(cfg, _strip(blob))
        _empty()
    save_json(dest, {"n": len(rows), "elapsed_s": time.time() - t0, "rows": _strip(rows)})
    return {"n": len(rows), "path": str(dest)}


def _strip(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in {"keep", "delta", "packed", "xA", "xB0", "x_aux", "owner", "u_mask", "v_mask", "h_clean", "keep_clean", "gold_ids", "scores_clean"}}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj

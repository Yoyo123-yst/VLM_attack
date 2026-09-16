from __future__ import annotations

from typing import Any, Dict, List

from .config import out_dir
from .io import load_json, save_json


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stage_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    probe_path = out_dir(cfg) / "probe.json"
    screen_path = out_dir(cfg) / "screen.json"
    if not probe_path.is_file():
        raise FileNotFoundError("run probe before report")
    probe = load_json(probe_path)
    screen = load_json(screen_path) if screen_path.is_file() else {"n_kept": 0, "n_rejected": 0}
    rows = probe["rows"]
    min_shift = float(cfg["probe"]["min_quota_shift"])
    n = len(rows)
    d_rA = [abs(r["avtp"]["delta_r"][0]) for r in rows]
    a_out = [r["avtp"]["events"]["a_out"] for r in rows]
    swap = [r["avtp"]["events"]["swap_ba"] for r in rows]
    full_hold = [int(bool(r["full_adv_ok"])) for r in rows]
    avtp_break = [int(r["avtp"]["comp_clean_ok"] and not r["avtp"]["comp_adv_ok"]) for r in rows]
    global_swap = [r["global"]["events"]["swap_ba"] for r in rows]
    causal = [r["causal"] for r in rows if r.get("causal")]
    iso_recover = [int(c["isolated_ok"]) for c in causal]
    restore_recover = [int(c["restore_a_ok"]) for c in causal]
    full_recover = [int(c["full_adv_ok"]) for c in causal]

    frac_quota = _mean([int(x >= min_shift) for x in d_rA])
    frac_a_out = _mean([int(x > 0) for x in a_out])
    frac_swap = _mean(swap)
    frac_full_hold = _mean(full_hold)
    frac_comp_fail = _mean(avtp_break)

    continue_ok = frac_quota >= 0.25 or frac_a_out >= 0.25
    lamp_like = frac_comp_fail > 0 and frac_full_hold < 0.5
    if n == 0:
        decision = "STOP"
        reason = "no probe rows"
    elif lamp_like and frac_a_out == 0:
        decision = "STOP"
        reason = "errors look like general cross-image interference, not token eviction"
    elif continue_ok and frac_full_hold >= 0.5:
        decision = "CONTINUE"
        reason = "B changes A's quota or survivor set while full-token mostly holds"
    elif continue_ok:
        decision = "CONTINUE_WEAK"
        reason = "selection moves, but full-token also often fails"
    else:
        decision = "STOP"
        reason = "B does not stably move A's quota or survivor set"

    summary = {
        "n_probe": n,
        "n_screen_kept": screen.get("n_kept"),
        "n_screen_rejected": screen.get("n_rejected"),
        "mean_abs_drA": _mean(d_rA),
        "frac_quota_shift": frac_quota,
        "frac_a_out": frac_a_out,
        "frac_swap_avtp": frac_swap,
        "frac_swap_global": _mean(global_swap),
        "frac_full_adv_hold": frac_full_hold,
        "frac_avtp_comp_fail": frac_comp_fail,
        "n_causal": len(causal),
        "frac_isolated_recover": _mean(iso_recover) if iso_recover else None,
        "frac_restore_a_recover": _mean(restore_recover) if restore_recover else None,
        "frac_full_recover_on_evict": _mean(full_recover) if full_recover else None,
        "decision": decision,
        "reason": reason,
        "dataset": "coco300_standin",
        "model": "Qwen2-VL-7B-Instruct",
    }
    save_json(out_dir(cfg) / "report.json", summary)
    md = _render(summary)
    (out_dir(cfg) / "P0_REPORT.md").write_text(md, encoding="utf-8")
    return summary


def _render(s: Dict[str, Any]) -> str:
    return f"""# V-CachePoll P0 report

- Dataset: `{s['dataset']}` (COCO stand-in)
- Model: `{s['model']}`
- Decision: **{s['decision']}**
- Reason: {s['reason']}

| Metric | Value |
|---|---|
| screen kept / rejected | {s['n_screen_kept']} / {s['n_screen_rejected']} |
| probe n | {s['n_probe']} |
| mean \\|Δr_A\\| | {s['mean_abs_drA']:.4f} |
| frac quota shift ≥ threshold | {s['frac_quota_shift']:.3f} |
| frac A-out (AVTP) | {s['frac_a_out']:.3f} |
| frac B-in/A-out swap (AVTP) | {s['frac_swap_avtp']:.3f} |
| frac B-in/A-out swap (global Top-K) | {s['frac_swap_global']:.3f} |
| frac full-token still correct on B_adv | {s['frac_full_adv_hold']:.3f} |
| frac AVTP compressed-only fail | {s['frac_avtp_comp_fail']:.3f} |
| causal n (A-out > 0) | {s['n_causal']} |
| isolated-quota recover | {s['frac_isolated_recover']} |
| restore-A recover | {s['frac_restore_a_recover']} |

CONTINUE means: change B (noise only) moves A's shared-budget survivors, and the full-token model usually still answers A correctly.

This is a mechanism probe, not a paper table.
"""


def _safe_mean(xs: List[float]) -> float:
    return _mean(xs)


def stage_p1_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    attack_path = out_dir(cfg) / "p1_attack.json"
    if not attack_path.is_file():
        raise FileNotFoundError("run p1 attack before report")
    blob = load_json(attack_path)
    rows = [r for r in blob.get("rows") or [] if r.get("eval")]
    probe_path = out_dir(cfg) / "probe.json"
    p0 = {r["pair_id"]: r for r in load_json(probe_path)["rows"]} if probe_path.is_file() else {}
    min_shift = float(cfg["probe"]["min_quota_shift"])
    n = len(rows)
    errors = [r for r in blob.get("rows") or [] if r.get("error")]

    d_rA, a_out, swap, full_hold, comp_fail, g_swap = [], [], [], [], [], []
    iso, restore, p0_a_out = [], [], []
    for r in rows:
        ev = r["eval"]["avtp"]["events"]
        hist = r.get("history") or []
        r0 = hist[0]["r"] if hist else r["eval"]["avtp"]["r"]
        r1 = r["eval"]["avtp"]["r"]
        d_rA.append(abs(r1[0] - r0[0]))
        a_out.append(ev["a_out"])
        swap.append(ev["swap_ba"])
        full_hold.append(int(bool(r["eval"]["full_ok"])))
        comp_fail.append(int(bool(r["eval"]["comp_only_fail"])))
        g_swap.append(r["eval"]["global"]["events"]["swap_ba"])
        iso.append(int(bool(r["eval"]["isolated_ok"])))
        restore.append(int(bool(r["eval"]["restore_a_ok"])))
        if r["pair_id"] in p0:
            p0_a_out.append(p0[r["pair_id"]]["avtp"]["events"]["a_out"])

    frac_quota = _mean([int(x >= min_shift) for x in d_rA])
    frac_a_out = _mean([int(x > 0) for x in a_out])
    frac_full = _mean(full_hold)
    frac_fail = _mean(comp_fail)
    mean_a_out = _mean([float(x) for x in a_out])
    mean_p0_a_out = _mean([float(x) for x in p0_a_out]) if p0_a_out else 0.0
    improved = (frac_quota >= 0.15) or (mean_a_out > max(mean_p0_a_out * 1.3, mean_p0_a_out + 0.5))

    if n == 0:
        decision, reason = "STOP", "no successful P1 rows"
    elif frac_full < 0.5 and frac_fail == 0:
        decision, reason = "STOP", "full-token often fails and compressed-only fail is absent (LAMP-like)"
    elif improved and frac_full >= 0.8:
        decision, reason = "CONTINUE", "quota/eviction beat P0 random while full-token holds"
    elif improved:
        decision, reason = "CONTINUE_WEAK", "selection moves more than random, but full-token is unstable"
    else:
        decision, reason = "STOP", "P1 is not clearly better than P0 random noise on B"

    summary = {
        "n_ok": n,
        "n_error": len(errors),
        "mean_abs_drA": _safe_mean(d_rA),
        "frac_quota_shift": frac_quota,
        "frac_a_out": frac_a_out,
        "mean_a_out": mean_a_out,
        "mean_p0_a_out": mean_p0_a_out,
        "frac_swap_avtp": _mean(swap),
        "frac_swap_global": _mean(g_swap),
        "frac_full_hold": frac_full,
        "frac_comp_only_fail": frac_fail,
        "frac_isolated_ok": _mean(iso),
        "frac_restore_a_ok": _mean(restore),
        "decision": decision,
        "reason": reason,
        "dataset": "coco300_standin",
        "model": "Qwen2-VL-7B-Instruct",
        "eps": float(cfg["attack"]["eps"]),
        "steps": int(cfg["attack"]["steps"]),
    }
    save_json(out_dir(cfg) / "p1_report.json", summary)
    md = _render_p1(summary)
    (out_dir(cfg) / "P1_REPORT.md").write_text(md, encoding="utf-8")
    return summary


def _render_p1(s: Dict[str, Any]) -> str:
    return f"""# V-CachePoll P1 report

- Dataset: `{s['dataset']}` (COCO stand-in)
- Model: `{s['model']}`
- Attack: quota + eviction PGD on B only, $\\epsilon={s['eps']:.4f}$, steps={s['steps']}
- Decision: **{s['decision']}**
- Reason: {s['reason']}

| Metric | Value |
|---|---|
| finished / errors | {s['n_ok']} / {s['n_error']} |
| mean \\|Δr_A\\| | {s['mean_abs_drA']:.4f} |
| frac quota shift ≥ 0.02 | {s['frac_quota_shift']:.3f} |
| frac A-out (AVTP) | {s['frac_a_out']:.3f} |
| mean A-out count | {s['mean_a_out']:.2f} |
| P0 random mean A-out | {s['mean_p0_a_out']:.2f} |
| frac B-in/A-out (AVTP) | {s['frac_swap_avtp']:.3f} |
| frac B-in/A-out (global Top-K) | {s['frac_swap_global']:.3f} |
| frac full-token still correct | {s['frac_full_hold']:.3f} |
| frac compressed-only fail | {s['frac_comp_only_fail']:.3f} |
| isolated-quota still ok | {s['frac_isolated_ok']:.3f} |
| restore-A still ok | {s['frac_restore_a_ok']:.3f} |

P1 is the first optimized attack on the 35 clean-correct pairs. It is not a paper table.
"""


def stage_p2_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from pathlib import Path

    attack_path = Path(cfg["attack"].get("result_path") or (out_dir(cfg) / "p2_attack.json"))
    if not attack_path.is_file():
        raise FileNotFoundError("run p2 attack before report")
    blob = load_json(attack_path)
    rows = [r for r in blob.get("rows") or [] if r.get("eval")]
    crit_path = out_dir(cfg) / "p2_crit.json"
    crit = load_json(crit_path) if crit_path.is_file() else {}
    p1_path = out_dir(cfg) / "p1_attack.json"
    p1 = {r["pair_id"]: r for r in (load_json(p1_path).get("rows") or []) if r.get("eval")} if p1_path.is_file() else {}
    n = len(rows)
    errors = [r for r in blob.get("rows") or [] if r.get("error")]
    full_hold, comp_fail, a_out, u_out, u_rate, swap, iso, restore = [], [], [], [], [], [], [], []
    for r in rows:
        ev = r["eval"]
        av = ev["avtp"]["events"]
        full_hold.append(int(bool(ev["full_ok"])))
        comp_fail.append(int(bool(ev["comp_only_fail"])))
        a_out.append(av["a_out"])
        u_out.append(int(ev.get("u_out") or 0))
        u_rate.append(float(ev.get("u_evict_rate") or 0.0))
        swap.append(av["swap_ba"])
        iso.append(int(bool(ev["isolated_ok"])))
        restore.append(int(bool(ev["restore_a_ok"])))
    frac_fail = _mean(comp_fail)
    frac_full = _mean(full_hold)
    n_loc = int(crit.get("n_localized") or 0)
    n_crit = int(crit.get("n") or 0)
    if n == 0:
        decision, reason = "STOP", "no successful P2 rows"
    elif frac_fail >= 0.20 and frac_full >= 0.80:
        decision, reason = "CONTINUE", "compressed-only task failure with full-token holding"
    elif frac_fail > 0 and frac_full >= 0.80:
        decision, reason = "CONTINUE_WEAK", "some compressed-only failures, but rare"
    elif frac_full < 0.5:
        decision, reason = "STOP", "full-token often fails (LAMP-like / too much semantic change)"
    else:
        decision, reason = "STOP", "still no compressed-only task failure on COCO stand-in"

    summary = {
        "n_ok": n,
        "n_error": len(errors),
        "n_screen_kept": n,
        "n_localized": n_loc,
        "frac_localized": (n_loc / n_crit) if n_crit else None,
        "frac_full_hold": frac_full,
        "frac_comp_only_fail": frac_fail,
        "frac_a_out": _mean([int(x > 0) for x in a_out]),
        "mean_a_out": _mean([float(x) for x in a_out]),
        "mean_u_out": _mean([float(x) for x in u_out]),
        "mean_u_evict_rate": _mean(u_rate),
        "frac_swap_avtp": _mean(swap),
        "frac_isolated_ok": _mean(iso),
        "frac_restore_a_ok": _mean(restore),
        "frac_restore_recovers_fail": _mean(
            [int(bool(r["eval"]["restore_a_ok"])) for r in rows if r["eval"]["comp_only_fail"]]
        )
        if any(r["eval"]["comp_only_fail"] for r in rows)
        else None,
        "n_p1_overlap": sum(1 for r in rows if r["pair_id"] in p1),
        "decision": decision,
        "reason": reason,
        "dataset": "coco300_standin",
        "model": "Qwen2-VL-7B-Instruct",
        "r_base": float(cfg["compressor"]["r_base"]),
        "eps": float(cfg["attack"]["eps"]),
        "steps": int(cfg["attack"]["steps"]),
    }
    save_json(out_dir(cfg) / "p2_report.json", summary)
    (out_dir(cfg) / "P2_REPORT.md").write_text(_render_p2(summary), encoding="utf-8")
    return summary


def _render_p2(s: Dict[str, Any]) -> str:
    return f"""# V-CachePoll P2 report

- Dataset: `{s['dataset']}` (COCO stand-in)
- Model: `{s['model']}`
- \(r_{{base}}={s['r_base']}\), \(\\epsilon={s['eps']:.4f}\), steps={s['steps']}
- Decision: **{s['decision']}**
- Reason: {s['reason']}

| Metric | Value |
|---|---|
| finished / errors | {s['n_ok']} / {s['n_error']} |
| localized critical tokens | {s['n_localized']} (frac={s['frac_localized']}) |
| frac A-out | {s['frac_a_out']:.3f} |
| mean A-out / U-out | {s['mean_a_out']:.2f} / {s['mean_u_out']:.2f} |
| mean U eviction rate | {s['mean_u_evict_rate']:.3f} |
| frac full-token still correct | {s['frac_full_hold']:.3f} |
| frac compressed-only fail | {s['frac_comp_only_fail']:.3f} |
| isolated still ok | {s['frac_isolated_ok']:.3f} |
| restore-A still ok | {s['frac_restore_a_ok']:.3f} |
| restore recovers fail | {s['frac_restore_recovers_fail']} |

P2 asks whether a tighter shared budget plus targeting high-value A tokens produces compression-only task failure.
"""


def _method_rows(path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    blob = load_json(path)
    return [r for r in blob.get("rows") or [] if r.get("eval")]


def _method_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    full = [int(bool(r["eval"]["full_ok"])) for r in rows]
    cof = [int(bool(r["eval"]["comp_only_fail"])) for r in rows]
    a_out = [r["eval"]["avtp"]["events"]["a_out"] for r in rows]
    u_rate = [float(r["eval"].get("u_evict_rate") or 0.0) for r in rows]
    u_out = [int(r["eval"].get("u_out") or 0) for r in rows]
    return {
        "n": len(rows),
        "frac_full_hold": _mean(full),
        "frac_comp_only_fail": _mean(cof),
        "mean_a_out": _mean([float(x) for x in a_out]),
        "mean_u_out": _mean([float(x) for x in u_out]),
        "mean_u_evict_rate": _mean(u_rate),
        "n_cof": int(sum(cof)),
    }


def stage_p3_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from pathlib import Path

    root = out_dir(cfg)
    names = ["random", "task", "caa", "rank", "vcache"]
    paths = {
        "random": root / "p3_random.json",
        "task": root / "p3_task.json",
        "caa": root / "p3_caa.json",
        "rank": root / "p3_rank.json",
        "vcache": root / "p2_attack.json",
    }
    methods = {k: _method_stats(_method_rows(p)) for k, p in paths.items()}
    vc = methods.get("vcache") or {}
    others = [methods[k] for k in ("random", "task", "caa", "rank") if methods[k].get("n")]
    vc_cof = float(vc.get("frac_comp_only_fail") or 0)
    vc_u = float(vc.get("mean_u_evict_rate") or 0)
    vc_full = float(vc.get("frac_full_hold") or 0)
    best_other_cof = max((float(x.get("frac_comp_only_fail") or 0) for x in others), default=0.0)
    best_other_u = max((float(x.get("mean_u_evict_rate") or 0) for x in others), default=0.0)
    if not others:
        decision, reason = "WAIT", "baselines not finished"
    elif vc_cof > best_other_cof + 0.05 and vc_full >= 0.8:
        decision, reason = "CONTINUE", "V-CachePoll compressed-only fail beats same-budget baselines; full-token holds"
    elif vc_u > best_other_u + 0.1 and vc_full >= 0.8:
        decision, reason = "CONTINUE_WEAK", "U-eviction is higher but compressed-only ASR margin is thin"
    else:
        decision, reason = "STOP", "V-CachePoll does not clearly beat same-protocol baselines"

    summary = {
        "methods": methods,
        "decision": decision,
        "reason": reason,
        "best_other_cof": best_other_cof,
        "best_other_u": best_other_u,
        "dataset": "coco300_standin",
        "model": "Qwen2-VL-7B-Instruct",
        "r_base": float(cfg["compressor"]["r_base"]),
        "eps": float(cfg["attack"]["eps"]),
        "steps": int(cfg["attack"]["steps"]),
    }
    save_json(root / "p3_report.json", summary)
    lines = [
        "# V-CachePoll P3 report",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Model: `{summary['model']}`",
        f"- \(r_{{base}}={summary['r_base']}\), \(\\epsilon={summary['eps']:.4f}\), steps={summary['steps']}",
        f"- Decision: **{decision}**",
        f"- Reason: {reason}",
        "",
        "| Method | n | full-token hold | compressed-only fail | mean A-out | mean U-out | U-evict rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        m = methods[name]
        if not m.get("n"):
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {m['n']} | {m['frac_full_hold']:.3f} | {m['frac_comp_only_fail']:.3f} | "
            f"{m['mean_a_out']:.2f} | {m['mean_u_out']:.2f} | {m['mean_u_evict_rate']:.3f} |"
        )
    lines += [
        "",
        "Same 32 P2 pairs, same \(\\epsilon\) and step budget. `vcache` is the P2 V-CachePoll run.",
        "Random uses 40 random restarts scored by exact compressor J. Rank-PGD is quota+evict PGD without exchange search.",
        "",
    ]
    (root / "P3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary

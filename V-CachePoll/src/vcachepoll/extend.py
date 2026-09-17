"""P4–P11 orchestration: CPU inventory, GPU gates, reports. No new attack story."""

from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .attack import (
    _empty,
    _load_kept,
    _now,
    _result_path,
    _strip,
    _write_progress,
    evaluate_final,
    pgd_one,
    prepare_sample,
    stage_p1_attack,
    stage_p1_smoke,
)
from .budget import DEFAULT_BUDGETS, budgets_for_cfg
from .config import load_cfg, out_dir
from .datasets import (
    build_multi_pairs,
    chartqa_ready,
    duplicate_first_aux,
    screen_eligible,
    textvqa_images_ready,
    write_multi_pairs,
)
from .io import load_json, save_json
from .report import _method_stats, _method_rows

ROOT = Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll")
EXTEND_OUT = ROOT / "out" / "extend"
STATUS = EXTEND_OUT / "STATUS.md"

MODEL_CATALOG = {
    "qwen2vl": {
        "path": "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct",
        "role": "current baseline victim",
    },
    "qwen3vl": {
        "path": "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct",
        "role": "AVTP original family; P10 M0–M5",
    },
    "internvl35": {
        "path": "/root/autodl-tmp/models/InternVL3.5-8B",
        "role": "AVTP original family; P10 M0–M5",
    },
    "llava_ov": {
        "path": "/root/autodl-tmp/models/LLaVA-OneVision-7B",
        "role": "AVTP original family; P10 M0–M5",
    },
}


def _stamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_status(stage: str, extra: str = "") -> None:
    EXTEND_OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        f"# V-CachePoll P4–P11 status\n\n- updated: `{_stamp()}`\n- stage: **{stage}**\n\n{extra.strip()}\n",
        encoding="utf-8",
    )
    print(f"== extend {stage} ==", flush=True)


def gpu_holder() -> Optional[Dict[str, Any]]:
    """Return the other process holding the GPU, if any."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    me = os.getpid()
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == me:
            continue
        cmd = ""
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            cmd = parts[-1] if parts else ""
        return {"pid": pid, "mem": parts[1] if len(parts) > 1 else "", "cmd": cmd.strip()}
    return None


def gpu_free_for_vcache() -> bool:
    holder = gpu_holder()
    if holder is None:
        return True
    cmd = holder.get("cmd") or ""
    if "V-CachePoll" in cmd:
        return True
    return False


def disk_ok(min_gb: float = 2.0) -> Dict[str, Any]:
    import shutil

    usage = shutil.disk_usage("/root/autodl-tmp")
    free_gb = usage.free / (1024 ** 3)
    return {"free_gb": round(free_gb, 2), "ok": free_gb >= min_gb}


def p10_inventory() -> Dict[str, Any]:
    rows = {}
    for name, spec in MODEL_CATALOG.items():
        p = Path(spec["path"])
        rows[name] = {
            "path": spec["path"],
            "present": p.is_dir() and any(p.iterdir()) if p.exists() else False,
            "role": spec["role"],
        }
    return rows


def p11_inventory(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_cfg(ROOT / "configs" / "p11.yaml")
    return {
        "textvqa_json": str((cfg.get("data") or {}).get("textvqa_json") or ROOT / "data" / "textvqa" / "TextVQA_0.5.1_val.json"),
        "textvqa_images": textvqa_images_ready(cfg),
        "chartqa": chartqa_ready(cfg),
        "coco_standin": Path(str(cfg["data"]["coco_root"])).is_dir(),
        "screen_rule": "A-only correct AND full multi-image correct AND clean compressed correct",
    }


def stage_p4_unit_ok() -> Dict[str, Any]:
    """P4 does not change the attack; logging helpers are covered by CPU tests."""
    return {"phase": "P4", "algorithm_frozen": True, "logging": "mechlog.mechanism_snapshot"}


def stage_p8_from_crit(crit_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compare U_score / U_LOO / drop-bot from saved P2 crit. Restore-U needs GPU eval."""
    path = crit_path or (ROOT / "out" / "p2_crit.json")
    if not path.is_file():
        return {"n": 0, "error": f"missing {path}"}
    rows = load_json(path).get("rows") or []
    n = len(rows)
    n_loo = sum(1 for r in rows if r.get("u_src") == "loo" or r.get("loo_hits"))
    n_drop_bot = sum(1 for r in rows if r.get("u_src") == "drop_bot")
    n_drop_top = sum(1 for r in rows if r.get("u_src") == "drop_top")
    n_bot_frac = sum(1 for r in rows if r.get("u_src") == "bot_frac")
    dest = EXTEND_OUT / "p8_u_sets.json"
    blob = {
        "n": n,
        "n_U_LOO": n_loo,
        "n_U_drop_bot": n_drop_bot,
        "n_U_drop_top": n_drop_top,
        "n_U_bot_frac": n_bot_frac,
        "note": "U_score is compressor Top-K; U_LOO is leave-one-out answer change; U_grad needs a GPU CE backward. Restore-U recovery is logged in new evaluate_final runs as restore_u_ok.",
        "gpu_restore_u": "queued",
    }
    save_json(dest, blob)
    return blob


def stage_p9_table(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Causal-control table from already finished P2/P3 artifacts. No GPU."""
    cfg = cfg or load_cfg(ROOT / "configs" / "p3.yaml")
    root = Path(cfg.get("output_dir") or (ROOT / "out"))
    paths = {
        "V-CachePoll": root / "p2_attack.json",
        "Task-PGD": root / "p3_task.json",
        "CAGE-B": root / "p3_cage.json",
        "CAA-B": root / "p3_caa.json",
        "Rank-PGD": root / "p3_rank.json",
        "Random": root / "p3_random.json",
        "LAMP-like": root / "p9_lamp.json",
    }
    table = {}
    for name, path in paths.items():
        rows = _method_rows(path)
        stats = _method_stats(rows)
        restore = None
        iso = None
        if rows:
            cof = [r for r in rows if r.get("eval", {}).get("comp_only_fail")]
            if cof:
                restore = sum(int(bool(r["eval"].get("restore_a_ok"))) for r in cof) / len(cof)
            iso = sum(int(bool(r["eval"].get("isolated_ok"))) / max(len(rows), 1) for r in rows)
            iso = sum(int(bool(r["eval"].get("isolated_ok"))) for r in rows) / len(rows)
        stats["isolated_ok"] = iso
        stats["restore_a_on_cof"] = restore
        table[name] = stats
    dest = EXTEND_OUT / "p9_table.json"
    save_json(dest, {"updated_at": _stamp(), "table": table, "lamp_run": paths["LAMP-like"].is_file()})
    lines = [
        "# P9 ordinary multi-image vs compression-resource attack",
        "",
        "| Method | n | Full Acc | Shared CASR | Isolated still-ok | Restore-A on COF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in table.items():
        if not m.get("n"):
            lines.append(f"| {name} | 0 | — | — | — | — |")
            continue
        iso = m.get("isolated_ok")
        rec = m.get("restore_a_on_cof")
        lines.append(
            f"| {name} | {m['n']} | {m['frac_full_hold']:.3f} | {m['frac_comp_only_fail']:.3f} "
            f"| {iso if iso is None else f'{iso:.3f}'} | {rec if rec is None else f'{rec:.3f}'} |"
        )
    lines += [
        "",
        "V-CachePoll should show shared CASR ≫ isolated failure (isolated still-ok high) and restore-A recovery.",
        "Ordinary Task-PGD / LAMP-like should not show that pattern.",
        "LAMP-like GPU run writes `out/p9_lamp.json` when the GPU is free.",
        "",
    ]
    (EXTEND_OUT / "P9_TABLE.md").write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(dest), "table": table}


def stage_p5_pairs(cfg: Dict[str, Any], n_aux: int) -> Dict[str, Any]:
    dest = out_dir(cfg) / f"pairs_m{n_aux}.json"
    pairs = write_multi_pairs(cfg, dest, n_aux=n_aux)
    dup = duplicate_first_aux(pairs[0], n_aux=max(int(n_aux), 2))
    save_json(out_dir(cfg) / f"pairs_m{n_aux}_dupB.json", {"n": 1, "pairs": [dup]})
    return {"n": len(pairs), "path": str(dest), "dup_id": dup["pair_id"]}


def stage_p6_transfer_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Describe experiment A/B. Experiment C (randomized r) stays off."""
    budgets = budgets_for_cfg(cfg)
    return {
        "experiment_a": "budget-specific attack at each r in budgets",
        "experiment_b": "train at r=0.2, evaluate saved delta at each r_test",
        "experiment_c": "SKIP — doc says do not add EOT-style randomized budget by default",
        "budgets": budgets,
        "train_r": 0.2,
    }


def p12_decision() -> Dict[str, Any]:
    return {
        "add_amp_loss": False,
        "reason": "Doc P12: wait until multi-image + multi-budget + multi-compressor finish. Do not add L_amp now.",
    }


def next_gpu_commands() -> List[str]:
    py = "/root/autodl-tmp/conda/envs/vattack/bin/python"
    root = str(ROOT)
    return [
        f"cd {root} && {py} scripts/run_extend.py --phase p4 --stage smoke --limit 2",
        f"cd {root} && {py} scripts/run_extend.py --phase p4 --stage pilot --limit 8",
        f"cd {root} && {py} scripts/run_extend.py --phase p4 --stage attack",
        f"cd {root} && {py} scripts/run_extend.py --phase p5 --stage smoke --n-aux 1 --limit 4",
        f"cd {root} && {py} scripts/run_extend.py --phase p6 --stage smoke --limit 1",
        f"cd {root} && {py} scripts/run_extend.py --phase p7 --stage smoke --limit 2",
        f"cd {root} && {py} scripts/run_extend.py --phase p9 --stage lamp --limit 8",
    ]


def frozen_p2_cfg_overlay() -> Dict[str, Any]:
    cfg = load_cfg(ROOT / "configs" / "p2.yaml")
    out = ROOT / "out" / "p4"
    atk = cfg["attack"]
    atk["log_prefix"] = "p4"
    atk["n_smoke"] = 2
    atk["smoke_steps"] = 6
    atk["progress_path"] = str(out / "p4_progress.json")
    atk["delta_dir"] = str(out / "p4_deltas")
    atk["smoke_path"] = str(out / "p4_smoke.json")
    atk["result_path"] = str(out / "p4_attack.json")
    atk["screen_path"] = str(ROOT / "out" / "p2_screen.json")
    cfg["output_dir"] = str(out)
    return cfg


def run_gpu_phase(phase: str, stage: str, limit: Optional[int] = None, n_aux: int = 1) -> Dict[str, Any]:
    if phase == "p10":
        return {"skipped": True, "reason": "new weights missing", "inventory": p10_inventory()}
    if phase == "p11":
        return {"skipped": True, "reason": "TextVQA images / ChartQA missing", "inventory": p11_inventory()}
    if phase == "p12":
        return p12_decision()
    if not gpu_free_for_vcache():
        holder = gpu_holder()
        return {
            "skipped": True,
            "reason": "GPU occupied",
            "holder": holder,
            "next": next_gpu_commands()[0],
        }
    disk = disk_ok(1.5)
    if not disk["ok"]:
        return {"skipped": True, "reason": "disk", "disk": disk}

    if phase == "p4":
        cfg = frozen_p2_cfg_overlay()
        if stage == "smoke":
            return stage_p1_smoke(cfg)
        if stage == "pilot":
            return stage_p1_attack(cfg, limit=limit or 8)
        if stage in {"attack", "full"}:
            return stage_p1_attack(cfg, limit=limit)
        raise ValueError(stage)

    if phase == "p5":
        cfg = load_cfg(ROOT / "configs" / "p5.yaml")
        cfg["attack"]["perturb"] = str(cfg.get("attack", {}).get("perturb") or "single")
        if stage == "pairs":
            return stage_p5_pairs(cfg, n_aux=n_aux)
        if stage == "smoke":
            cfg["attack"]["n_smoke"] = int(limit or 2)
            return stage_p1_smoke(cfg)
        if stage == "attack":
            return stage_p1_attack(cfg, limit=limit)
        raise ValueError(stage)

    if phase == "p6":
        cfg = load_cfg(ROOT / "configs" / "p6.yaml")
        if stage == "smoke":
            cfg["attack"]["n_smoke"] = int(limit or 1)
            return stage_p1_smoke(cfg)
        if stage == "attack":
            return stage_p1_attack(cfg, limit=limit)
        raise ValueError(stage)

    if phase == "p7":
        cfg = load_cfg(ROOT / "configs" / "p7.yaml")
        if stage == "smoke":
            cfg["attack"]["n_smoke"] = int(limit or 2)
            return stage_p1_smoke(cfg)
        if stage == "attack":
            return stage_p1_attack(cfg, limit=limit)
        raise ValueError(stage)

    if phase == "p9" and stage == "lamp":
        cfg = load_cfg(ROOT / "configs" / "p9.yaml")
        return stage_p1_attack(cfg, limit=limit)

    raise ValueError(f"unknown phase/stage {phase}/{stage}")

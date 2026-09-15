#!/usr/bin/env python3
"""P0.5 — DENY linear separability analysis (zero GPU).

Given routeC/out/deny_hidden/*.npz (L12/L16/L20/L24:last_user, dim 3584),
answer the mechanistic question:

    Is DENY a real geometric state, linearly separable from REFUSE and ANSWER
    on the visual latent, or is it just a text-generation accident?

Checks (for each layer):
  1. DENY vs REFUSE          (the wall test: is the safety wall a distinct state)
  2. DENY vs ANSWER          (is DENY just "non-answer")
  3. DENY vs (REFUSE∪ANSWER) (one-vs-rest)
  4. 3-way DENY/REFUSE/ANSWER multiclass
  5. Within-cell separation (same query+carrier): does DENY separate even when
     image content is held fixed (controls for carrier confound)
  6. Random-dir / shuffle control (baseline AUC ~0.5)

Reports mean AUC + 5-fold CV AUC, plus within-cell macro AUC. Writes
routeC/out/deny_separability.json and a markdown report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p0.datautil import load_json, save_json  # noqa: E402

OUT = ROOT / "routeC" / "out"
HIDDEN_DIR = OUT / "deny_hidden"
CR0 = ROOT / "outputs" / "causal_reach" / "cr0" / "candidates.json"
N1R = ROOT / "outputs" / "n1r_fast" / "candidates.json"
LAYERS = [12, 16, 20, 24]

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402


def _records(c: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(c.get("records", []))


def load_meta() -> Dict[str, Dict[str, Any]]:
    meta = {}
    for path in (CR0, N1R):
        for r in _records(load_json(path)):
            meta[r["record_id"]] = r
    return meta


def load_hidden() -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for p in sorted(HIDDEN_DIR.glob("*.npz")):
        rid = p.stem
        z = np.load(p)
        out[rid] = {k: z[k].astype(np.float32) for k in z.files}
    return out


def binary_auc(X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Mean AUC via StratifiedKFold; if a class <2 or too few, return n/a."""
    from collections import Counter
    cnt = Counter(y)
    if len(cnt) < 2 or min(cnt.values()) < 2:
        return {"auc_mean": float("nan"), "n_pos": cnt.get(1, 0), "n_neg": cnt.get(0, 0), "note": "insufficient"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    try:
        aucs = cross_val_score(clf, X, y, cv=min(cv, min(cnt.values())), scoring="roc_auc")
        return {"auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()), "n_pos": cnt[1], "n_neg": cnt[0]}
    except Exception as e:  # noqa: BLE001
        return {"auc_mean": float("nan"), "n_pos": cnt.get(1, 0), "n_neg": cnt.get(0, 0), "note": str(e)[:120]}


def main() -> None:
    meta = load_meta()
    hidden = load_hidden()
    print(f"loaded {len(hidden)} hidden records, {len(meta)} meta", flush=True)

    # build (rid -> mode) for the records we actually have hidden for
    mode_of = {}
    cell_of = {}
    for rid in hidden:
        m = meta[rid]
        mode_of[rid] = m.get("response_mode")
        cell_of[rid] = (m.get("query_id"), m.get("carrier_id"))

    results: Dict[str, Any] = {"schema": "deny_separability_v1", "layers": {}, "n_records": len(hidden)}

    for layer in LAYERS:
        key = f"L{layer}:last_user"
        X = np.vstack([hidden[rid][key] for rid in sorted(hidden)])
        rids = sorted(hidden)
        y_mode = np.array([mode_of[rid] for rid in rids])
        cells = np.array([f"{cell_of[rid][0]}:{cell_of[rid][1]}" for rid in rids])

        r = {
            "n": int(len(rids)),
            "mode_counts": {m: int((y_mode == m).sum()) for m in sorted(set(y_mode))},
            "binary": {},
            "multiclass": {},
            "within_cell": {},
            "random_baseline": {},
        }

        # 1) DENY vs REFUSE
        mask = np.isin(y_mode, ["DENY", "REFUSE"])
        r["binary"]["deny_vs_refuse"] = binary_auc(
            X[mask], (y_mode[mask] == "DENY").astype(int)
        )
        # 2) DENY vs ANSWER
        mask = np.isin(y_mode, ["DENY", "ANSWER"])
        r["binary"]["deny_vs_answer"] = binary_auc(
            X[mask], (y_mode[mask] == "DENY").astype(int)
        )
        # 3) DENY vs (REFUSE∪ANSWER)
        y_ovr = (y_mode == "DENY").astype(int)
        r["binary"]["deny_vs_rest"] = binary_auc(X, y_ovr)

        # 4) 3-way multiclass accuracy
        clf3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial"))
        try:
            accs = cross_val_score(clf3, X, y_mode, cv=5, scoring="accuracy")
            r["multiclass"]["3way_acc_mean"] = float(accs.mean())
            r["multiclass"]["3way_acc_std"] = float(accs.std())
        except Exception as e:  # noqa: BLE001
            r["multiclass"]["3way_acc_mean"] = float("nan")
            r["multiclass"]["note"] = str(e)[:120]

        # 5) Within-cell separation (only cells with >=1 DENY and >=1 non-DENY)
        cell_aucs = []
        cell_detail = {}
        for cell in sorted(set(cells)):
            cm = cells == cell
            sub_modes = y_mode[cm]
            if "DENY" not in set(sub_modes):
                continue
            others = [m for m in set(sub_modes) if m != "DENY"]
            if not others:
                continue
            # collapse others into one "non-DENY" class
            sub_y = (sub_modes == "DENY").astype(int)
            if len(set(sub_y)) < 2 or min(np.bincount(sub_y)) < 2:
                continue
            auc = binary_auc(X[cm], sub_y, cv=min(3, min(np.bincount(sub_y))))
            if np.isfinite(auc["auc_mean"]):
                cell_aucs.append(auc["auc_mean"])
                cell_detail[cell] = {"auc": auc["auc_mean"], "n_deny": int(sub_y.sum()), "n_other": int((1 - sub_y).sum())}
        r["within_cell"]["n_cells"] = len(cell_detail)
        r["within_cell"]["macro_auc"] = float(np.mean(cell_aucs)) if cell_aucs else float("nan")
        r["within_cell"]["per_cell"] = cell_detail

        # 6) Random-dir baseline: DENY vs rest with a random projection
        rng = np.random.default_rng(2026)
        rp = rng.normal(size=X.shape[1]).astype(np.float32)
        rp /= (np.linalg.norm(rp) + 1e-8)
        proj = X @ rp
        # AUC of a scalar is not meaningful; instead shuffle labels to get chance
        y_shuf = y_ovr.copy()
        rng.shuffle(y_shuf)
        r["random_baseline"]["shuffled_label_auc"] = binary_auc(X, y_shuf)["auc_mean"]

        results["layers"][str(layer)] = r
        print(f"L{layer}: deny_vs_refuse={r['binary']['deny_vs_refuse'].get('auc_mean')} "
              f"deny_vs_answer={r['binary']['deny_vs_answer'].get('auc_mean')} "
              f"deny_vs_rest={r['binary']['deny_vs_rest'].get('auc_mean')} "
              f"3way={r['multiclass'].get('3way_acc_mean')} "
              f"within_cell_macro={r['within_cell']['macro_auc']} "
              f"shuffled={r['random_baseline']['shuffled_label_auc']}", flush=True)

    save_json(OUT / "deny_separability.json", results)
    print("wrote routeC/out/deny_separability.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P0.7 — within-query DENY separability (the decisive confound control).

The confound audit showed: query h53-vs-h61 separates at AUC=1.0 within DENY,
and carrier c07-vs-c08 at 0.88-0.99. So the global DENY-vs-ANSWER AUC (0.72-0.84)
is partly query/carrier confound. The decisive test: does DENY separate from
ANSWER/REFUSE *within the same query* (carrier may still vary)?

If within-query AUC also collapses, DENY is NOT an independent geometric state —
it is a projection of query content (which concepts are visually ungroundable).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT / "src"))

from p0.datautil import load_json  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HIDDEN_DIR = ROOT / "routeC" / "out" / "deny_hidden"


def auc(X, y, pos, neg):
    m = (y == pos) | (y == neg)
    n_pos = int((y[m] == pos).sum())
    n_neg = int((y[m] == neg).sum())
    if m.sum() < 6 or n_pos < 2 or n_neg < 2:
        return float("nan"), n_pos, n_neg
    yy = (y[m] == pos).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = min(5, n_pos, n_neg)
    return float(cross_val_score(clf, X[m], yy, cv=cv, scoring="roc_auc").mean()), n_pos, n_neg


def main():
    meta = {}
    for path in [
        ROOT / "outputs/causal_reach/cr0/candidates.json",
        ROOT / "outputs/n1r_fast/candidates.json",
    ]:
        for r in load_json(path)["records"]:
            meta[r["record_id"]] = r

    files = sorted(HIDDEN_DIR.glob("*.npz"))
    rids = [f.stem for f in files]
    mode = np.array([meta[r].get("response_mode") for r in rids])
    query = np.array([meta[r]["query_id"] for r in rids])

    # queries with >=2 DENY and >=2 ANSWER (or REFUSE)
    for layer in [12, 16, 20, 24]:
        key = f"L{layer}:last_user"
        X = np.vstack([np.load(f)[key].astype(np.float32) for f in files])
        print(f"\n=== L{layer} within-query DENY separability ===")
        rows = []
        for q in sorted(set(query)):
            mq = query == q
            sub_mode = mode[mq]
            if (sub_mode == "DENY").sum() < 2:
                continue
            a_ans, na, nans = auc(X[mq], sub_mode, "DENY", "ANSWER")
            a_ref, nr, nref = auc(X[mq], sub_mode, "DENY", "REFUSE")
            rows.append((q, na, nans, a_ans, nr, nref, a_ref))
        # aggregate macro AUC (unweighted across queries that have the class pair)
        ans_vals = [r[3] for r in rows if np.isfinite(r[3])]
        ref_vals = [r[6] for r in rows if np.isfinite(r[6])]
        print(f"  {'query':>5} {'n_D':>3} {'n_A':>3} {'DENY-vs-A':>10} {'n_R':>3} {'DENY-vs-R':>10}")
        for q, na, nans, a_ans, nr, nref, a_ref in rows:
            print(f"  {q:>5} {na:>3} {nans:>3} {a_ans:>10.3f} {nref:>3} {a_ref:>10.3f}")
        print(f"  MACRO DENY-vs-ANSWER: {np.mean(ans_vals):.3f} (n={len(ans_vals)} queries)")
        print(f"  MACRO DENY-vs-REFUSE: {np.mean(ref_vals):.3f} (n={len(ref_vals)} queries)")


if __name__ == "__main__":
    main()

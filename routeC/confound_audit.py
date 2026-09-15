#!/usr/bin/env python3
"""P0.6 — confound audit: is the DENY separability driven by carrier/query confound?

Uses the same hidden states. For each layer, measures:
  - carrier separability (c07 vs c08) across ALL samples and within DENY only
  - query separability within DENY (h53 vs h61)
If the global DENY-vs-answer AUC collapses once carrier is controlled, the
global result is a confound, not a real DENY geometric state.
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
    """Binary AUC between class `pos` and class `neg` in the label array `y`."""
    m = (y == pos) | (y == neg)
    if m.sum() < 6 or (y[m] == pos).sum() < 2 or (y[m] == neg).sum() < 2:
        return float("nan")
    yy = (y[m] == pos).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = min(5, int((y[m] == pos).sum()), int((y[m] == neg).sum()))
    return float(cross_val_score(clf, X[m], yy, cv=cv, scoring="roc_auc").mean())


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
    carrier = np.array([meta[r]["carrier_id"] for r in rids])
    query = np.array([meta[r]["query_id"] for r in rids])

    print("carrier distribution (all):", {c: int((carrier == c).sum()) for c in sorted(set(carrier))})
    dm = mode == "DENY"
    print("carrier distribution (DENY):", {c: int((carrier[dm] == c).sum()) for c in sorted(set(carrier[dm]))})
    print("query distribution (DENY):", {q: int((query[dm] == q).sum()) for q in sorted(set(query[dm]))})
    print()

    for layer in [12, 16, 20, 24]:
        key = f"L{layer}:last_user"
        X = np.vstack([np.load(f)[key].astype(np.float32) for f in files])
        Xd = X[dm]
        cd = carrier[dm]
        qd = query[dm]
        print(f"--- L{layer} ---")
        print(f"  carrier c07-vs-c08 (all):   {auc(X, carrier, 'c07', 'c08'):.3f}")
        print(f"  carrier c07-vs-c08 (DENY):  {auc(Xd, cd, 'c07', 'c08'):.3f}")
        mq = np.isin(qd, ["h53", "h61"])
        print(f"  query h53-vs-h61 (DENY):    {auc(Xd[mq], qd[mq], 'h53', 'h61'):.3f}")
        # DENY vs ANSWER within a SINGLE carrier only (controls carrier fully)
        for c in ["c07", "c08"]:
            msc = (carrier == c) & ((mode == "DENY") | (mode == "ANSWER"))
            if msc.sum() >= 6:
                print(f"  DENY-vs-ANSWER within {c}: {auc(X[msc], mode[msc], 'DENY', 'ANSWER'):.3f}")


if __name__ == "__main__":
    main()

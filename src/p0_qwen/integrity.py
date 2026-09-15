"""Split isolation, provenance hashes, and cache validity for P0-Qwen Attack Gate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from p0.catalog import attack_test_pairs, probe_pairs
from p0.datautil import load_json, save_json
from p0.subspace import cov_matched_basis, mean_state, pca_basis_info, random_basis

PIN_LAYER = 24
PIN_REQUESTED_RANK = 32
ATTACK_FREE_GB = 10.0
LEAK_USED_NO_PROC_MIB = 4500
REQUIRED_PROVENANCE_KEYS = (
    "traces_rhc_sha256",
    "subspace_sha256",
    "u_matrix_sha256",
    "test_catalog_sha256",
    "model_revision",
    "attack_config_sha256",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes((text or "").encode("utf-8"))


def sha256_jsonable(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(blob)


def catalog_sha256() -> str:
    probe = [{"id": p["id"], "query": p["query"]} for p in probe_pairs()]
    attack = [{"id": p["id"], "query": p["query"]} for p in attack_test_pairs()]
    return sha256_jsonable({"probe": probe, "attack_test": attack})


def test_catalog_sha256() -> str:
    attack = [{"id": p["id"], "query": p["query"], "benign_id": p["benign_id"]} for p in attack_test_pairs()]
    return sha256_jsonable(attack)


def attack_config_sha256(cfg: Dict[str, Any]) -> str:
    return sha256_jsonable(
        {
            "seed": cfg.get("seed"),
            "attack": cfg.get("attack"),
            "image_size": (cfg.get("model") or {}).get("image_size"),
            "max_new_tokens": (cfg.get("model") or {}).get("max_new_tokens"),
            "pin_layer": PIN_LAYER,
            "pin_requested_rank": PIN_REQUESTED_RANK,
        }
    )


def model_revision(cfg: Dict[str, Any]) -> str:
    path = Path((cfg.get("model") or {}).get("local_path") or "")
    parts = {"path": str(path)}
    for name in ("config.json", "generation_config.json", "preprocessor_config.json"):
        p = path / name
        parts[name] = sha256_file(p)
    return sha256_jsonable(parts)


def u_matrix_sha256(U: Any) -> str:
    arr = np.asarray(U, dtype=np.float32)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def provenance_complete(prov: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(prov, dict):
        return False
    return all(prov.get(k) for k in REQUIRED_PROVENANCE_KEYS)


def probe_id_sets(probe: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    discover = set(probe.get("discover") or [])
    holdout = set(probe.get("holdout") or [])
    usable = set(probe.get("usable") or []) | discover | holdout
    return discover, holdout, usable


def catalog_id_sets() -> Tuple[Set[str], Set[str]]:
    probe_ids = {p["id"] for p in probe_pairs()}
    attack_ids = {p["id"] for p in attack_test_pairs()}
    return probe_ids, attack_ids


def assert_split_isolation(probe: Optional[Dict[str, Any]] = None, traces: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    probe_ids, attack_ids = catalog_id_sets()
    if probe_ids & attack_ids:
        raise AssertionError(f"catalog probe ∩ attack-test nonempty: {sorted(probe_ids & attack_ids)[:8]}")
    rec: Dict[str, Any] = {
        "n_probe": len(probe_ids),
        "n_attack_test": len(attack_ids),
        "probe_attack_overlap": 0,
    }
    if probe:
        discover, holdout, _usable = probe_id_sets(probe)
        if discover & holdout:
            raise AssertionError(f"discover ∩ holdout nonempty: {sorted(discover & holdout)[:8]}")
        if discover - probe_ids:
            raise AssertionError(f"discover ids outside h01–h90: {sorted(discover - probe_ids)[:8]}")
        if holdout - probe_ids:
            raise AssertionError(f"holdout ids outside h01–h90: {sorted(holdout - probe_ids)[:8]}")
        if discover & attack_ids:
            raise AssertionError("discover ∩ attack-test nonempty")
        if holdout & attack_ids:
            raise AssertionError("holdout ∩ attack-test nonempty")
        rec.update(
            {
                "n_discover": len(discover),
                "n_holdout": len(holdout),
                "discover_holdout_overlap": 0,
            }
        )
    if traces:
        for rec_t in traces.get("records") or []:
            qid = rec_t.get("query_id")
            if qid in attack_ids:
                raise AssertionError(f"trace query {qid} is on the attack-test split")
            if qid not in probe_ids:
                raise AssertionError(f"trace query {qid} is not in probe catalog")
        rec["n_trace_records"] = len(traces.get("records") or [])
    return rec


def _deltas(records: Sequence[Dict[str, Any]], layer: int, jb_key: str = "jb") -> np.ndarray:
    key = f"L{layer}:last_user"
    rows = []
    for p in records:
        src = np.asarray(p[jb_key]["hidden"][key], dtype=np.float32).reshape(-1)
        ref = np.asarray(p["clean_hidden"][key], dtype=np.float32).reshape(-1)
        rows.append(src - ref)
    return np.stack(rows, axis=0)


def align_subspace_effective_rank(sub_blob: Dict[str, Any], traces: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Truncate overcomplete U / U_rand to centered PCA effective rank. CPU only."""
    from p0_qwen.traces import fail_pairs, jb_pairs

    pairs = jb_pairs(traces, split="discover")
    if len(pairs) < 2:
        pairs = jb_pairs(traces, split=None)
    n_disc = len(pairs)
    changed = False
    sub_blob["n_pairs"] = n_disc
    sub_blob["n_discover_rhc"] = n_disc
    sub_blob["centered"] = True
    sub_blob["requested_rank"] = PIN_REQUESTED_RANK
    for layer_s, layer_rec in (sub_blob.get("layers") or {}).items():
        layer = int(layer_s)
        if not pairs:
            continue
        X = _deltas(pairs, layer)
        hrefs = [np.asarray(p["clean_hidden"][f"L{layer}:last_user"], dtype=np.float32).reshape(-1) for p in pairs]
        hjbs = [np.asarray(p["jb"]["hidden"][f"L{layer}:last_user"], dtype=np.float32).reshape(-1) for p in pairs]
        for r_s, spec in (layer_rec.get("ranks") or {}).items():
            requested = int(r_s)
            info = pca_basis_info(X, requested)
            U = info["U"]
            used = int(info["used_rank"])
            old_u = spec.get("U")
            old_cols = len(old_u[0]) if old_u else None
            old_rand = spec.get("U_rand")
            rand_cols = len(old_rand[0]) if old_rand else None
            already = (
                spec.get("requested_rank") == requested
                and spec.get("effective_rank") == info["effective_rank"]
                and spec.get("used_rank") == used
                and old_cols == used
                and rand_cols == used
                and spec.get("n_discover_rhc") == n_disc
            )
            if already:
                continue
            spec["U"] = U.tolist()
            spec["mu_ref"] = mean_state(hrefs, U).tolist()
            spec["mu_jb"] = mean_state(hjbs, U).tolist()
            spec["U_rand"] = random_basis(X.shape[1], used, seed=2026 + layer + requested).tolist()
            spec["U_cov"] = cov_matched_basis(X, used, seed=7 + layer + requested).tolist()
            spec["delta_norm_mean"] = float(np.linalg.norm(X, axis=1).mean())
            spec["requested_rank"] = requested
            spec["effective_rank"] = int(info["effective_rank"])
            spec["used_rank"] = used
            spec["n_discover_rhc"] = n_disc
            spec["centered"] = True
            spec["singular_values"] = info["singular_values"]
            spec["n_columns_before_align"] = old_cols
            changed = True
        fail_spec = layer_rec.get("fail")
        fails = fail_pairs(traces, split="discover") or fail_pairs(traces, split=None)
        if fail_spec and len(fails) >= 2:
            Xf = _deltas(fails, layer, jb_key="fail")
            r_fail_req = int(fail_spec.get("rank") or min(8, Xf.shape[0] - 1))
            finfo = pca_basis_info(Xf, r_fail_req)
            Uf = finfo["U"]
            hfail = [np.asarray(p["fail"]["hidden"][f"L{layer}:last_user"], dtype=np.float32).reshape(-1) for p in fails]
            if len(fail_spec.get("U", [[]])[0]) != Uf.shape[1] or fail_spec.get("effective_rank") != finfo["effective_rank"]:
                fail_spec["U"] = Uf.tolist()
                fail_spec["rank"] = int(finfo["used_rank"])
                fail_spec["requested_rank"] = r_fail_req
                fail_spec["effective_rank"] = int(finfo["effective_rank"])
                fail_spec["used_rank"] = int(finfo["used_rank"])
                fail_spec["mu_fail"] = mean_state(hfail, Uf).tolist()
                fail_spec["mu_ref"] = mean_state(hrefs, Uf).tolist()
                changed = True
    l24 = ((sub_blob.get("layers") or {}).get(str(PIN_LAYER)) or {}).get("ranks", {}).get(str(PIN_REQUESTED_RANK)) or {}
    if l24:
        sub_blob["effective_rank"] = l24.get("effective_rank")
        sub_blob["used_rank"] = l24.get("used_rank")
        sub_blob["u_matrix_sha256"] = u_matrix_sha256(l24.get("U"))
    return sub_blob, changed


def build_provenance(
    cfg: Dict[str, Any],
    out: Path,
    *,
    subspace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    traces_rhc = out / "traces_rhc.json"
    traces = out / "traces.json"
    sub_path = out / "subspace.json"
    l24 = None
    if subspace:
        l24 = ((subspace.get("layers") or {}).get(str(PIN_LAYER)) or {}).get("ranks", {}).get(str(PIN_REQUESTED_RANK))
    u_hash = None
    if l24 and l24.get("U") is not None:
        u_hash = u_matrix_sha256(l24["U"])
    elif subspace and subspace.get("u_matrix_sha256"):
        u_hash = subspace["u_matrix_sha256"]
    return {
        "traces_rhc_sha256": sha256_file(traces_rhc),
        "traces_sha256": sha256_file(traces),
        "subspace_sha256": sha256_file(sub_path),
        "u_matrix_sha256": u_hash,
        "test_catalog_sha256": test_catalog_sha256(),
        "catalog_sha256": catalog_sha256(),
        "model_revision": model_revision(cfg),
        "attack_config_sha256": attack_config_sha256(cfg),
        "model_path": (cfg.get("model") or {}).get("local_path"),
        "pin_layer": PIN_LAYER,
        "pin_requested_rank": PIN_REQUESTED_RANK,
        "n_discover_rhc": None if subspace is None else subspace.get("n_discover_rhc", subspace.get("n_pairs")),
        "effective_rank": None if subspace is None else subspace.get("effective_rank"),
        "requested_rank": PIN_REQUESTED_RANK,
        "used_rank": None if subspace is None else subspace.get("used_rank"),
        "centered": True,
    }


def hashes_match(blob: Optional[Dict[str, Any]], expected: Dict[str, Any], keys: Iterable[str] = REQUIRED_PROVENANCE_KEYS) -> bool:
    if not blob:
        return False
    prov = blob.get("provenance") or {}
    if not provenance_complete(prov):
        return False
    for k in keys:
        if not expected.get(k) or prov.get(k) != expected.get(k):
            return False
    return True


def patch_cache_ok(blob: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    if not blob or blob.get("error"):
        return False
    if blob.get("selection_was_pre_clean_holdout"):
        return False
    if not hashes_match(blob, expected, keys=("traces_rhc_sha256", "test_catalog_sha256", "model_revision")):
        return False
    hold = blob.get("summary_holdout") or {}
    ns = [int((hold.get(str(L)) or {}).get("n") or 0) for L in hold]
    if ns and max(ns) > 20:
        # pre-clean holdout was n=27
        return False
    return bool(blob.get("split") == "holdout" or blob.get("u_from_recleaned_discover_rhc"))


def causal_cache_ok(blob: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    if not blob or blob.get("error"):
        return False
    if blob.get("frozen") and blob.get("selection_was_pre_clean_holdout") and not blob.get("postclean_holdout_eval"):
        return False
    if not hashes_match(blob, expected, keys=("traces_rhc_sha256", "subspace_sha256", "u_matrix_sha256")):
        return False
    return bool(blob.get("postclean_holdout_eval"))


def attack_cache_ok(blob: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    if not blob or blob.get("error"):
        return False
    if not hashes_match(blob, expected):
        return False
    if not blob.get("eligible_ids"):
        return False
    return True


def archive_stale(src: Path, dest: Path) -> Optional[str]:
    if not src.exists():
        return None
    if dest.exists():
        alt = dest.with_name(dest.stem + "_stale" + dest.suffix)
        if alt.exists():
            src.unlink()
            return str(alt)
        shutil.move(str(src), str(alt))
        return str(alt)
    shutil.move(str(src), str(dest))
    return str(dest)


def frozen_l24_spec(sub_blob: Dict[str, Any]) -> Dict[str, Any]:
    spec = ((sub_blob.get("layers") or {}).get(str(PIN_LAYER)) or {}).get("ranks", {}).get(str(PIN_REQUESTED_RANK))
    if not spec or not spec.get("U"):
        raise KeyError("missing L24 requested-rank-32 U in subspace.json")
    return spec


def compact_patch_rows(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for r in blob.get("rows") or []:
        r2j = r.get("ref_to_jb") or {}
        j2r = r.get("jb_to_ref") or {}
        rows.append(
            {
                "query_id": r.get("query_id"),
                "carrier_id": r.get("carrier_id"),
                "split": r.get("split"),
                "layer": r.get("layer"),
                "tag": r.get("tag"),
                "control": r.get("control"),
                "clean_R": r.get("clean_R"),
                "jb_R": r.get("jb_R"),
                "ref_to_jb_R": r2j.get("R"),
                "ref_to_jb_label": r2j.get("label"),
                "jb_to_ref_R": None if not j2r else j2r.get("R"),
                "jb_to_ref_label": None if not j2r else j2r.get("label"),
            }
        )
    return rows

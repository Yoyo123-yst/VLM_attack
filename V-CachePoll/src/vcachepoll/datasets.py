"""Multi-image COCO pairs and TextVQA / ChartQA builders (CPU, no model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .coco_pairs import _hash_int, build_pairs, list_coco_images, load_json, load_sources, write_pairs
from .io import save_json


def aux_prompt(n_aux: int, question: str, two_image_prefix: str) -> str:
    q = str(question).strip()
    if int(n_aux) <= 1:
        return f"{two_image_prefix} {q}".strip()
    return f"Look at the first image only. Ignore the other images. {q}".strip()


def build_multi_pairs(
    coco_root: Path,
    vqa_json: Path,
    coco_csv: Path,
    n_pairs: int,
    seed: int,
    n_aux: int,
    question_prefix: str,
) -> List[Dict[str, Any]]:
    """A + M_aux unused val2017 images. n_aux=1 matches two-image COCO pairs."""
    n_aux = int(n_aux)
    if n_aux < 1:
        raise ValueError("n_aux must be >= 1")
    if n_aux == 1:
        pairs = build_pairs(
            coco_root=coco_root,
            vqa_json=vqa_json,
            coco_csv=coco_csv,
            n_pairs=n_pairs,
            seed=seed,
            question_prefix=question_prefix,
        )
        for p in pairs:
            p["b_paths"] = [p["b_path"]]
            p["n_aux"] = 1
        return pairs

    vqa = load_json(vqa_json)
    sources = load_sources(coco_csv)
    a_names = [row["image"] for row in vqa if row["image"] in sources]
    a_set = set(a_names)
    pool = [name for name in list_coco_images(coco_root) if name not in a_set]
    if len(pool) < n_aux:
        raise RuntimeError(f"need {n_aux} unused COCO images, have {len(pool)}")
    pairs: List[Dict[str, Any]] = []
    for i, row in enumerate(vqa):
        if len(pairs) >= n_pairs:
            break
        a_name = row["image"]
        if a_name not in sources:
            continue
        a_path = coco_root / a_name
        if not a_path.is_file():
            continue
        questions: Sequence[str] = row.get("vqa") or []
        if not questions:
            continue
        b_names = []
        used = {a_name}
        for j in range(n_aux):
            b_name = pool[_hash_int(f"{seed}:{a_name}:aux{j}") % len(pool)]
            guard = 0
            while b_name in used and guard < len(pool):
                b_name = pool[(_hash_int(f"{seed}:{a_name}:aux{j}:{guard}") + guard) % len(pool)]
                guard += 1
            if b_name in used:
                continue
            used.add(b_name)
            b_names.append(b_name)
        if len(b_names) != n_aux:
            continue
        b_paths = [str(coco_root / n) for n in b_names]
        if any(not Path(p).is_file() for p in b_paths):
            continue
        q = str(questions[0]).strip()
        pairs.append(
            {
                "pair_id": f"c{i:03d}_m{n_aux}",
                "a_file": a_name,
                "b_file": b_names[0],
                "b_files": b_names,
                "a_path": str(a_path),
                "b_path": b_paths[0],
                "b_paths": b_paths,
                "n_aux": n_aux,
                "source": sources[a_name],
                "question": q,
                "prompt": aux_prompt(n_aux, q, question_prefix),
            }
        )
    if not pairs:
        raise RuntimeError("built zero multi-image COCO pairs")
    return pairs


def duplicate_first_aux(pair: Dict[str, Any], n_aux: int) -> Dict[str, Any]:
    """P5 packing test: B2=B1=... copies of the first auxiliary image."""
    out = dict(pair)
    b0 = pair.get("b_path") or (pair.get("b_paths") or [None])[0]
    if not b0:
        raise ValueError("pair has no b_path")
    out["b_paths"] = [b0] * int(n_aux)
    out["b_path"] = b0
    out["n_aux"] = int(n_aux)
    out["pair_id"] = f"{pair.get('pair_id', 'dup')}_dup{n_aux}"
    out["prompt"] = aux_prompt(n_aux, pair.get("question") or "", "Look at the first image only. Ignore the second image.")
    return out


def textvqa_image_dir(cfg: Dict[str, Any]) -> Path:
    raw = (cfg.get("data") or {}).get("textvqa_image_dir")
    if raw:
        return Path(raw)
    return Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll/data/textvqa/images")


def textvqa_json_path(cfg: Dict[str, Any]) -> Path:
    raw = (cfg.get("data") or {}).get("textvqa_json")
    if raw:
        return Path(raw)
    return Path("/root/autodl-tmp/multimodal_attack_project/V-CachePoll/data/textvqa/TextVQA_0.5.1_val.json")


def textvqa_images_ready(cfg: Dict[str, Any]) -> bool:
    d = textvqa_image_dir(cfg)
    if not d.is_dir():
        return False
    return any(d.glob("*"))


def chartqa_ready(cfg: Dict[str, Any]) -> bool:
    raw = str((cfg.get("data") or {}).get("chartqa_dir") or "").strip()
    if not raw:
        return False
    d = Path(raw)
    if not d.is_dir():
        return False
    return any(d.iterdir())


def build_textvqa_pairs(
    cfg: Dict[str, Any],
    n_pairs: int,
    seed: int,
    n_aux: int = 1,
    coco_b_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """A from TextVQA; B from unused COCO if provided. Requires on-disk TextVQA images."""
    if not textvqa_images_ready(cfg):
        return []
    blob = load_json(textvqa_json_path(cfg))
    rows = blob.get("data") or []
    img_dir = textvqa_image_dir(cfg)
    coco_root = coco_b_root or Path(str((cfg.get("data") or {}).get("coco_root") or ""))
    pool: List[str] = []
    if coco_root.is_dir():
        pool = list_coco_images(coco_root)
    prefix = str((cfg.get("data") or {}).get("question_prefix") or "Look at the first image only. Ignore the second image.")
    pairs: List[Dict[str, Any]] = []
    for row in rows:
        if len(pairs) >= n_pairs:
            break
        image_id = str(row.get("image_id") or "")
        candidates = list(img_dir.glob(f"{image_id}*"))
        if not candidates:
            continue
        a_path = candidates[0]
        answers = row.get("answers") or []
        gold = str(answers[0] if answers else "")
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        b_paths: List[str] = []
        if pool and coco_root.is_dir():
            for j in range(max(int(n_aux), 1)):
                b_name = pool[_hash_int(f"{seed}:tvqa:{image_id}:{j}") % len(pool)]
                b_paths.append(str(coco_root / b_name))
        else:
            continue
        pairs.append(
            {
                "pair_id": f"tvqa_{row.get('question_id', image_id)}",
                "a_file": a_path.name,
                "b_file": Path(b_paths[0]).name,
                "a_path": str(a_path),
                "b_path": b_paths[0],
                "b_paths": b_paths,
                "n_aux": int(n_aux),
                "source": gold,
                "question": q,
                "prompt": aux_prompt(n_aux, q, prefix),
                "ans_a_only": gold,
                "dataset": "textvqa",
            }
        )
    return pairs


def screen_eligible(a_only_ok: bool, full_ok: bool, clean_comp_ok: bool) -> bool:
    """P11: only attack samples that are correct A-only, full multi-image, and clean compressed."""
    return bool(a_only_ok) and bool(full_ok) and bool(clean_comp_ok)


def write_multi_pairs(cfg: Dict[str, Any], dest: Path, n_aux: int) -> List[Dict[str, Any]]:
    data = cfg["data"]
    pairs = build_multi_pairs(
        coco_root=Path(data["coco_root"]),
        vqa_json=Path(data["vqa_json"]),
        coco_csv=Path(data["coco_csv"]),
        n_pairs=int(data.get("n_pairs") or 8),
        seed=int(cfg.get("seed") or 2026),
        n_aux=int(n_aux),
        question_prefix=str(data.get("question_prefix") or "Look at the first image only. Ignore the second image."),
    )
    save_json(
        dest,
        {
            "dataset": f"coco300_maux{n_aux}",
            "n": len(pairs),
            "n_aux": int(n_aux),
            "pairs": pairs,
        },
    )
    return pairs

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .io import load_json, save_json


_STOP = {
    "a",
    "an",
    "the",
    "that",
    "this",
    "with",
    "from",
    "into",
    "onto",
    "have",
    "has",
    "had",
    "been",
    "being",
    "which",
    "what",
    "when",
    "where",
    "there",
    "their",
    "image",
    "images",
    "first",
    "second",
    "picture",
    "shown",
    "visible",
    "appears",
    "object",
    "objects",
    "looks",
    "looking",
    "ignore",
    "only",
}


def load_sources(csv_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["image"]] = row["source"]
    return out


def source_keywords(source: str) -> List[str]:
    words = [w for w in re.split(r"[^a-z0-9]+", source.lower()) if w and w not in _STOP]
    return words


def content_words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) >= 4 and w not in _STOP}


def novel_words(answer: str, prompt: str) -> set[str]:
    return content_words(answer) - content_words(prompt)


def answers_agree(pred: str, ref: str, prompt: str, source: str = "") -> bool:
    """True if pred and ref share answer content beyond the question text."""
    if mentions_source(pred, source) and mentions_source(ref, source):
        return True
    wp = novel_words(pred, prompt)
    wr = novel_words(ref, prompt)
    if wp and wr:
        return len(wp & wr) >= 1
    return False


def mentions_source(answer: str, source: str) -> bool:
    text = (answer or "").lower()
    src = (source or "").lower().strip()
    if not src:
        return False
    if src in text:
        return True
    keys = source_keywords(source)
    if not keys:
        return False
    return keys[-1] in text


def _hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def list_coco_images(root: Path) -> List[str]:
    files = sorted(p.name for p in root.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no jpg files under {root}")
    return files


def build_pairs(
    coco_root: Path,
    vqa_json: Path,
    coco_csv: Path,
    n_pairs: int,
    seed: int,
    question_prefix: str,
) -> List[Dict[str, Any]]:
    vqa = load_json(vqa_json)
    sources = load_sources(coco_csv)
    a_names = [row["image"] for row in vqa if row["image"] in sources]
    a_set = set(a_names)
    pool = [name for name in list_coco_images(coco_root) if name not in a_set]
    if not pool:
        raise RuntimeError("no unused val2017 images for B")
    pairs = []
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
        b_name = pool[_hash_int(f"{seed}:{a_name}") % len(pool)]
        b_path = coco_root / b_name
        if not b_path.is_file() or b_name == a_name:
            continue
        q = str(questions[0]).strip()
        pairs.append(
            {
                "pair_id": f"c{i:03d}",
                "a_file": a_name,
                "b_file": b_name,
                "a_path": str(a_path),
                "b_path": str(b_path),
                "source": sources[a_name],
                "question": q,
                "prompt": f"{question_prefix} {q}".strip(),
            }
        )
    if not pairs:
        raise RuntimeError("built zero COCO A/B pairs")
    return pairs


def write_pairs(cfg: Dict[str, Any], dest: Path) -> List[Dict[str, Any]]:
    data = cfg["data"]
    pairs = build_pairs(
        coco_root=Path(data["coco_root"]),
        vqa_json=Path(data["vqa_json"]),
        coco_csv=Path(data["coco_csv"]),
        n_pairs=int(data["n_pairs"]),
        seed=int(cfg["seed"]),
        question_prefix=str(data["question_prefix"]),
    )
    save_json(
        dest,
        {
            "dataset": "coco300_standin",
            "n": len(pairs),
            "note": "A from coco300 VQA; B from unused COCO val2017. Not TextVQA.",
            "pairs": pairs,
        },
    )
    return pairs

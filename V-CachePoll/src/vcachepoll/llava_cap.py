"""LLaVA-1.5 local capability probe: can it pack two images for AVTP transfer?"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

LLAVA_SNAP = Path(
    "/root/autodl-tmp/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
)


def _snapshot() -> str:
    if LLAVA_SNAP.is_dir() and (LLAVA_SNAP / "config.json").is_file():
        return str(LLAVA_SNAP)
    root = LLAVA_SNAP.parent
    if not root.is_dir():
        raise FileNotFoundError(f"no LLaVA snapshots under {root}")
    kids = [p for p in root.iterdir() if p.is_dir()]
    if not kids:
        raise FileNotFoundError(f"empty LLaVA snapshot dir {root}")
    return str(kids[0])


def probe_llava() -> Dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    path = _snapshot()
    out: Dict[str, Any] = {"path": path, "multi_image": False, "error": None}
    try:
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    except Exception as exc:
        out["error"] = f"processor: {type(exc).__name__}: {exc}"
        return out
    a = Image.new("RGB", (336, 336), color=(20, 80, 20))
    b = Image.new("RGB", (336, 336), color=(80, 20, 20))
    prompt = "USER: <image>\n<image>\nWhat is in the first image? ASSISTANT:"
    try:
        packed = proc(images=[a, b], text=prompt, return_tensors="pt")
        shapes = {k: list(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in packed.items()}
        out["processor_shapes"] = shapes
        n_img_tok = None
        ids = packed.get("input_ids")
        if ids is not None:
            n_img_tok = int((ids == proc.tokenizer.convert_tokens_to_ids("<image>")).sum().item()) if "<image>" in proc.tokenizer.get_vocab() else int(ids.shape[-1])
        out["n_seq"] = int(ids.shape[-1]) if ids is not None else None
        out["n_image_tokens_guess"] = n_img_tok
        # LLaVA-1.5 typically collapses to one image feature grid.
        pv = packed.get("pixel_values")
        out["pixel_values_shape"] = list(pv.shape) if pv is not None else None
        out["multi_image"] = bool(pv is not None and pv.dim() >= 4 and pv.shape[0] >= 2)
        if not out["multi_image"] and pv is not None and pv.dim() == 5:
            out["multi_image"] = int(pv.shape[1]) >= 2
    except Exception as exc:
        out["error"] = f"pack: {type(exc).__name__}: {exc}"
        return out

    # Cheap generate only if two-image pack looks real and GPU is free enough.
    try:
        free = torch.cuda.mem_get_info()[0] / (1024 ** 3) if torch.cuda.is_available() else 0.0
        out["cuda_free_gb"] = free
        if out["multi_image"] and free > 18.0:
            from transformers import LlavaForConditionalGeneration

            model = LlavaForConditionalGeneration.from_pretrained(
                path, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model.eval()
            packed = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in packed.items()}
            gen = model.generate(**packed, max_new_tokens=8)
            out["gen_preview"] = proc.batch_decode(gen, skip_special_tokens=True)[0][:200]
            del model
            torch.cuda.empty_cache()
        else:
            out["skipped_generate"] = "not multi-image or not enough free VRAM"
    except Exception as exc:
        out["error"] = f"generate: {type(exc).__name__}: {exc}"
    return out

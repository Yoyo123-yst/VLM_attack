"""Differentiable Qwen2-VL patchify. Matches Qwen2VLImageProcessor._preprocess."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2


def grid_hw(grid_row: torch.Tensor, patch_size: int = PATCH_SIZE) -> Tuple[int, int]:
    _t, h, w = (int(x) for x in grid_row.reshape(-1)[:3].tolist())
    return int(h) * patch_size, int(w) * patch_size


def pil_to_hw(image: Image.Image, height: int, width: int) -> torch.Tensor:
    img = image.resize((width, height), Image.BICUBIC)
    x = torch.from_numpy(np.array(img).astype("float32") / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0)


def x01_to_pil(x01: torch.Tensor) -> Image.Image:
    arr = x01.detach().float().cpu().clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")


def patchify_x01(
    x01: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x01.ndim != 4 or x01.shape[0] != 1 or x01.shape[1] != 3:
        raise ValueError(f"expected [1,3,H,W], got {tuple(x01.shape)}")
    _, _, height, width = x01.shape
    if height % (patch_size * merge_size) or width % (patch_size * merge_size):
        raise ValueError(f"H={height} W={width} must be divisible by {patch_size * merge_size}")
    mean = CLIP_MEAN.to(device=x01.device, dtype=x01.dtype)
    std = CLIP_STD.to(device=x01.device, dtype=x01.dtype)
    x = (x01 - mean) / std
    patches = x.repeat(temporal_patch_size, 1, 1, 1)
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    grid = torch.tensor([[grid_t, grid_h, grid_w]], device=x01.device, dtype=torch.long)
    return flat, grid


def pixel_rows_from_grid(grid_thw: torch.Tensor) -> list[int]:
    rows = []
    for row in grid_thw.reshape(-1, 3):
        t, h, w = (int(x) for x in row.tolist())
        rows.append(int(t * h * w))
    return rows


def clip_delta(x0: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
    delta = delta.clamp(-eps, eps)
    return torch.clamp(x0 + delta, 0.0, 1.0) - x0


def tv_loss(x01: torch.Tensor) -> torch.Tensor:
    dh = (x01[:, :, 1:, :] - x01[:, :, :-1, :]).abs().mean()
    dw = (x01[:, :, :, 1:] - x01[:, :, :, :-1]).abs().mean()
    return dh + dw

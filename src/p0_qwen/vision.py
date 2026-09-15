"""Fixed-resolution differentiable Qwen2-VL image packing."""

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


def to_square_pil(image: Image.Image, size: int = 336) -> Image.Image:
    w, h = image.size
    scale = size / float(min(w, h))
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = image.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def pil_to_x01(image: Image.Image, size: int = 336) -> torch.Tensor:
    img = to_square_pil(image, size=size)
    x = torch.from_numpy(np.array(img).astype("float32") / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0)


def patchify_x01(
    x01: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map [1,3,H,W] in [0,1] to Qwen2-VL pixel_values + image_grid_thw.

    Matches Qwen2VLImageProcessor._preprocess reshape/transpose for a single image.
    """
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


def tv_loss(x01: torch.Tensor) -> torch.Tensor:
    dh = (x01[:, :, 1:, :] - x01[:, :, :-1, :]).abs().mean()
    dw = (x01[:, :, :, 1:] - x01[:, :, :, :-1]).abs().mean()
    return dh + dw


def eot_scale(x: torch.Tensor, scale_lo: float, scale_hi: float) -> torch.Tensor:
    if scale_lo == 1.0 and scale_hi == 1.0:
        return x
    s = float(torch.empty(1, device=x.device).uniform_(scale_lo, scale_hi))
    if abs(s - 1.0) < 1e-3:
        return x
    x2 = torch.nn.functional.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
    return torch.nn.functional.interpolate(x2, size=x.shape[-2:], mode="bilinear", align_corners=False)

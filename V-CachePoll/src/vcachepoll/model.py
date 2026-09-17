from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from .compressor import layer_variation_score, tokens_per_image, visual_owner
from .vision import grid_hw, patchify_x01, pil_to_hw, pixel_rows_from_grid


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def perturb_linf(image: Image.Image, eps: float, seed: int) -> Image.Image:
    """Deterministic integer-noise approximation of an l_inf ball on [0,1] pixels."""
    import numpy as np

    rng = np.random.RandomState(seed)
    arr = np.asarray(image).astype(np.int16)
    radius = max(int(round(eps * 255.0)), 1)
    noise = rng.randint(-radius, radius + 1, size=arr.shape, dtype=np.int16)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@dataclass
class PackedInput:
    tensors: Dict[str, torch.Tensor]
    n_per_image: List[int]
    vis_index: torch.Tensor  # positions in the sequence that are visual tokens


class QwenMultiImage:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        import os

        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        path = cfg["model"]["local_path"]
        size = int(cfg["model"]["image_size"])
        dtype_name = str(cfg["model"].get("dtype", "bfloat16"))
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
        attn = str(cfg["model"].get("attn_implementation", "sdpa"))
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        ip = self.processor.image_processor
        ip.max_pixels = size * size
        ip.min_pixels = size * size
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation=attn,
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"attn={attn} failed ({exc}); retrying eager", flush=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
                attn_implementation="eager",
                trust_remote_code=True,
            )
        self.model.eval()
        self.model.requires_grad_(False)
        self.max_new = int(cfg["model"]["max_new_tokens"])
        self.image_token_id = int(self.model.config.image_token_id)
        self.merge = int(cfg["compressor"].get("spatial_merge_size", 2))
        self.score_layers = [int(x) for x in cfg["compressor"]["score_layers"]]
        self.device = next(self.model.parameters()).device
        gc = getattr(self.model, "generation_config", None)
        if gc is not None:
            gc.do_sample = False
            gc.temperature = None
            gc.top_p = None
            gc.top_k = None

    def pack(self, images: Sequence[Image.Image], prompt: str) -> PackedInput:
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tensors = self.processor(text=[text], images=list(images), return_tensors="pt", padding=True)
        tensors = {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in tensors.items()
        }
        grid = tensors["image_grid_thw"]
        n_per = tokens_per_image(grid, merge=self.merge)
        vis = torch.nonzero(tensors["input_ids"][0] == self.image_token_id, as_tuple=False).flatten()
        if int(vis.numel()) != sum(n_per):
            raise RuntimeError(
                f"visual token mismatch: seq has {int(vis.numel())}, grid implies {sum(n_per)}"
            )
        return PackedInput(tensors=tensors, n_per_image=n_per, vis_index=vis)

    def owner(self, packed: PackedInput) -> torch.Tensor:
        return visual_owner(packed.n_per_image, device=packed.vis_index.device)

    def with_pixels(self, packed: PackedInput, pixel_values: torch.Tensor) -> PackedInput:
        tensors = dict(packed.tensors)
        tensors["pixel_values"] = pixel_values
        return PackedInput(tensors=tensors, n_per_image=list(packed.n_per_image), vis_index=packed.vis_index)

    def x01_for_grid(self, image: Image.Image, grid_row: torch.Tensor) -> torch.Tensor:
        height, width = grid_hw(grid_row)
        return pil_to_hw(image, height, width).to(device=self.device, dtype=torch.float32)

    def patchify(self, x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return patchify_x01(x01.to(dtype=torch.float32))

    def pixels_from_xs(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not xs:
            raise ValueError("pixels_from_xs needs at least one image tensor")
        return torch.cat([self.patchify(x)[0] for x in xs], dim=0)

    def pixels_from_x01(self, xA: torch.Tensor, xB: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        return self.pixels_from_xs([xA, xB, *rest])

    def split_processor_pixels(self, packed: PackedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        pv = packed.tensors["pixel_values"]
        rows = pixel_rows_from_grid(packed.tensors["image_grid_thw"])
        if sum(rows) != int(pv.shape[0]):
            raise RuntimeError(f"pixel row mismatch: grid {rows} vs pv {tuple(pv.shape)}")
        return pv[: rows[0]], pv[rows[0] :]

    def _forward_hidden(self, packed: PackedInput, pixel_values: Optional[torch.Tensor] = None):
        tensors = packed.tensors if pixel_values is None else {**packed.tensors, "pixel_values": pixel_values}
        out = self.model(
            **tensors,
            output_hidden_states=True,
            use_cache=False,
        )
        captured = {i: out.hidden_states[i][0] for i in range(len(out.hidden_states))}
        seq_score = layer_variation_score(captured, self.score_layers)
        last_h = out.hidden_states[-1][0, -1]
        scores = seq_score[packed.vis_index]
        return scores, last_h, out

    @torch.no_grad()
    def visual_scores(self, packed: PackedInput) -> torch.Tensor:
        scores, _last, out = self._forward_hidden(packed)
        del out
        return scores.detach()

    def visual_scores_grad(self, packed: PackedInput, pixel_values: torch.Tensor):
        """Importance scores + last-token hidden, graph intact for PGD."""
        scores, last_h, out = self._forward_hidden(packed, pixel_values=pixel_values)
        del out
        return scores, last_h

    def _full_embeds(self, packed: PackedInput) -> torch.Tensor:
        input_ids = packed.tensors["input_ids"]
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        pixel_values = packed.tensors["pixel_values"].type(self.model.visual.get_dtype())
        image_embeds = self.model.visual(pixel_values, grid_thw=packed.tensors["image_grid_thw"])
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _compact(self, packed: PackedInput, keep_visual: torch.Tensor):
        ids = packed.tensors["input_ids"][0]
        vis = packed.vis_index
        owner = self.owner(packed)
        keep = keep_visual.to(device=ids.device, dtype=torch.bool)
        seq_keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        seq_keep[vis] = False
        seq_keep[vis[keep]] = True
        new_ids = ids[seq_keep].unsqueeze(0)
        new_attn = torch.ones_like(new_ids)
        embeds = self._full_embeds(packed)[:, seq_keep]
        grid_rows = []
        for i in range(len(packed.n_per_image)):
            ki = int((keep[owner == i]).sum().item())
            ki = max(ki, 1)
            patches = ki * self.merge * self.merge
            grid_rows.append([1, self.merge, patches // self.merge])
        grid = torch.tensor(grid_rows, dtype=torch.long, device=ids.device)
        return new_ids, new_attn, embeds, grid

    @torch.no_grad()
    def generate(self, packed: PackedInput, keep_visual: Optional[torch.Tensor] = None) -> str:
        """Official generate. Unselected visual tokens are dropped from the prefix."""
        if keep_visual is None:
            gen = self.model.generate(
                **packed.tensors,
                max_new_tokens=self.max_new,
                do_sample=False,
                use_cache=True,
            )
            prefix = int(packed.tensors["input_ids"].shape[1])
        else:
            new_ids, new_attn, embeds, grid = self._compact(packed, keep_visual)
            gen = self.model.generate(
                input_ids=new_ids,
                inputs_embeds=embeds,
                attention_mask=new_attn,
                image_grid_thw=grid,
                max_new_tokens=self.max_new,
                do_sample=False,
                use_cache=True,
            )
            prefix = int(new_ids.shape[1])
        return self.processor.tokenizer.decode(gen[0, prefix:], skip_special_tokens=True).strip()


def load_pair_images(pair: Dict[str, Any], b_image: Optional[Image.Image] = None) -> Tuple[Image.Image, Image.Image]:
    a = open_rgb(pair["a_path"])
    b = b_image if b_image is not None else open_rgb(pair["b_path"])
    return a, b


def load_sample_images(pair: Dict[str, Any], b_image: Optional[Image.Image] = None) -> List[Image.Image]:
    a = open_rgb(pair["a_path"])
    paths = list(pair.get("b_paths") or [pair["b_path"]])
    if b_image is not None:
        rest = [b_image] + [open_rgb(p) for p in paths[1:]]
    else:
        rest = [open_rgb(p) for p in paths]
    return [a, *rest]

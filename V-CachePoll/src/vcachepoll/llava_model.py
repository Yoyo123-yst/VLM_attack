"""LLaVA-1.5 two-image wrapper for AVTP transfer probes."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .compressor import layer_variation_score, visual_owner
from .llava_cap import _snapshot
from .model import PackedInput, open_rgb
from .vision import CLIP_MEAN, CLIP_STD, pil_to_hw


class LlavaMultiImage:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch as _t
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        path = str(cfg.get("llava", {}).get("local_path") or _snapshot())
        dtype = _t.float16
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = "default"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new = int(cfg.get("model", {}).get("max_new_tokens", 32))
        self.score_layers = list(cfg.get("compressor", {}).get("score_layers", [1, 14, 19]))
        self.image_token_id = int(self.model.config.image_token_index)
        self.image_size = 336

    def pixels_from_x01(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        mean = CLIP_MEAN.to(device=xA.device, dtype=xA.dtype)
        std = CLIP_STD.to(device=xA.device, dtype=xA.dtype)
        return torch.cat([(xA - mean) / std, (xB - mean) / std], dim=0)

    def x01_from_pil(self, image: Image.Image) -> torch.Tensor:
        return pil_to_hw(image, self.image_size, self.image_size).to(device=self.device, dtype=torch.float32)

    def pack(self, images: List[Image.Image], prompt: str) -> PackedInput:
        question = prompt
        # Drop the Qwen-style prefix if present; LLaVA uses USER/ASSISTANT.
        if "Ignore the second image." in question:
            question = question.split("Ignore the second image.", 1)[-1].strip()
        text = f"USER: <image>\n<image>\nLook at the first image only. Ignore the second image. {question} ASSISTANT:"
        tensors = self.processor(images=images, text=text, return_tensors="pt")
        tensors = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in tensors.items()}
        ids = tensors["input_ids"][0]
        vis = torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        if vis.numel() < 2:
            raise RuntimeError(f"LLaVA packed {int(vis.numel())} image tokens")
        n_total = int(vis.numel())
        n_a = n_total // 2
        n_per = [n_a, n_total - n_a]
        return PackedInput(tensors=tensors, n_per_image=n_per, vis_index=vis)

    def owner(self, packed: PackedInput) -> torch.Tensor:
        return visual_owner(packed.n_per_image, device=packed.vis_index.device)

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
        scores, last_h, out = self._forward_hidden(packed, pixel_values=pixel_values)
        del out
        return scores, last_h

    def _compact(self, packed: PackedInput, keep_visual: torch.Tensor):
        ids = packed.tensors["input_ids"][0]
        vis = packed.vis_index
        keep = keep_visual.to(device=ids.device, dtype=torch.bool)
        seq_keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        seq_keep[vis] = False
        seq_keep[vis[keep]] = True
        inputs_embeds = self.model.get_input_embeddings()(packed.tensors["input_ids"])
        pixel_values = packed.tensors["pixel_values"]
        image_outputs = self.model.vision_tower(pixel_values, output_hidden_states=True)
        selected = image_outputs.hidden_states[self.model.config.vision_feature_layer]
        if self.model.config.vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        image_features = self.model.multi_modal_projector(selected)
        image_features = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.dtype)
        mask = (packed.tensors["input_ids"] == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        embeds = inputs_embeds.masked_scatter(mask, image_features)
        new_ids = ids[seq_keep].unsqueeze(0)
        new_attn = torch.ones_like(new_ids)
        new_embeds = embeds[:, seq_keep]
        return new_ids, new_attn, new_embeds

    @torch.no_grad()
    def generate(self, packed: PackedInput, keep_visual: Optional[torch.Tensor] = None) -> str:
        if keep_visual is None:
            gen = self.model.generate(
                **packed.tensors,
                max_new_tokens=self.max_new,
                do_sample=False,
                use_cache=True,
            )
            prefix = int(packed.tensors["input_ids"].shape[1])
        else:
            new_ids, new_attn, embeds = self._compact(packed, keep_visual)
            gen = self.model.generate(
                inputs_embeds=embeds,
                attention_mask=new_attn,
                max_new_tokens=self.max_new,
                do_sample=False,
                use_cache=True,
            )
            prefix = int(embeds.shape[1]) if int(gen.shape[1]) > int(embeds.shape[1]) else 0
        return self.processor.tokenizer.decode(gen[0, prefix:], skip_special_tokens=True).strip()

"""LLaVA-1.5 wrapper: 4-bit LLM, fp16 vision, residual hooks, first-token grads."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def configure_processor(processor) -> None:
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def llama_layers(model) -> torch.nn.ModuleList:
    lm = model.language_model
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError("cannot find LLaMA layers")


class LlavaP0:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        path = cfg["model"]["local_path"]
        quant = None
        if cfg["model"].get("load_in_4bit", True):
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=list(cfg["model"].get("skip_modules", [])),
            )
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        configure_processor(self.processor)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            path,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=cfg["model"].get("attn_implementation", "eager"),
        )
        self.model.eval()
        self.max_new = int(cfg["model"]["max_new_tokens"])
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.num_layers = int(self.model.config.text_config.num_hidden_layers)
        self.image_token_id = int(self.model.config.image_token_index)
        self.device = next(self.model.parameters()).device
        self.layers = llama_layers(self.model)
        from .prefixes import PREFIXES

        self.prefixes = PREFIXES
        self.prefix_name = "A"

    def set_prefix(self, name: str) -> None:
        if name not in self.prefixes:
            raise KeyError(f"unknown prefix {name}")
        self.prefix_name = name

    def prompt_text(self, question: str) -> str:
        wrapped = self.prefixes[self.prefix_name] + question
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": wrapped}, {"type": "image"}],
            }
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def encode(self, image: Image.Image, question: str) -> Dict[str, torch.Tensor]:
        text = self.prompt_text(question)
        packed = self.processor(text=text, images=image, return_tensors="pt")
        out = {}
        for k, v in packed.items():
            if hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        if "pixel_values" in out:
            out["pixel_values"] = out["pixel_values"].to(dtype=torch.float16)
        return out

    def last_user_index(self, input_ids: torch.Tensor) -> int:
        return int(input_ids.shape[-1] - 1)

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: Optional[int] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> str:
        inputs = self.encode(image, question)
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=torch.float16)
        n_in = int(inputs["input_ids"].shape[-1])
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new,
            do_sample=False,
            use_cache=True,
        )
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        return text.strip()

    def _hidden_at_layer(
        self,
        inputs: Dict[str, torch.Tensor],
        layer: int,
        token_index: int,
        patch: Optional[Tuple[str, torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Return residual at (layer, token). Optional patch = (mode, source_h, U).

        mode:
          full  — replace the whole residual vector
          sub   — replace only the U U^T component
        """
        captured: Dict[str, torch.Tensor] = {}

        def hook(_m, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            seq = hidden.shape[1]
            pos = token_index if seq > 1 else 0
            if seq > 1 and token_index >= seq:
                pos = seq - 1
            if patch is not None:
                mode, src, U = patch
                hidden = hidden.clone()
                src_vec = src.to(device=hidden.device, dtype=hidden.dtype).view(-1)
                if mode == "full":
                    hidden[:, pos, :] = src_vec
                elif mode == "sub":
                    assert U is not None
                    u = U.to(device=hidden.device, dtype=hidden.dtype)
                    cur = hidden[:, pos, :]
                    src_b = src_vec.unsqueeze(0)
                    hidden[:, pos, :] = cur + (src_b - cur) @ u @ u.T
                elif mode == "add":
                    hidden[:, pos, :] = hidden[:, pos, :] + src_vec
            captured["h"] = hidden[:, pos, :]
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        hdl = self.layers[layer].register_forward_hook(hook)
        try:
            self.model(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            hdl.remove()
        return captured["h"]

    @torch.no_grad()
    def collect_hidden(
        self,
        image: Image.Image,
        question: str,
        layers: Sequence[int],
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        inputs = self.encode(image, question)
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=torch.float16)
        tok = self.last_user_index(inputs["input_ids"])
        out: Dict[str, torch.Tensor] = {}
        for layer in layers:
            h = self._hidden_at_layer(inputs, layer, tok, patch=None)
            out[f"L{layer}:last_user"] = h.detach().float().cpu().reshape(-1)
        return out

    @torch.no_grad()
    def generate_with_patch(
        self,
        image: Image.Image,
        question: str,
        layer: int,
        source_h: torch.Tensor,
        mode: str = "full",
        U: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> str:
        inputs = self.encode(image, question)
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(device=self.device, dtype=torch.float16)
        tok = self.last_user_index(inputs["input_ids"])
        n_in = int(inputs["input_ids"].shape[-1])

        def hook(_m, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] == 1:
                return output
            hidden = hidden.clone()
            pos = tok if tok < hidden.shape[1] else hidden.shape[1] - 1
            src_vec = source_h.to(device=hidden.device, dtype=hidden.dtype).view(-1)
            if mode == "full":
                hidden[:, pos, :] = src_vec
            elif mode == "sub":
                assert U is not None
                u = U.to(device=hidden.device, dtype=hidden.dtype)
                cur = hidden[:, pos, :]
                hidden[:, pos, :] = cur + (src_vec.unsqueeze(0) - cur) @ u @ u.T
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        hdl = self.layers[layer].register_forward_hook(hook)
        try:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new,
                do_sample=False,
                use_cache=True,
            )
        finally:
            hdl.remove()
        return self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()

    def first_token_logits(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "use_cache": False,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = self.model(**kwargs)
        return out.logits[0, -1]

    def hidden_last_user(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        layer: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        captured: Dict[str, torch.Tensor] = {}

        def hook(_m, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[:, -1, :]
            return output

        hdl = self.layers[layer].register_forward_hook(hook)
        try:
            kwargs = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "use_cache": False,
            }
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            self.model(**kwargs)
        finally:
            hdl.remove()
        return captured["h"][0]


def image_to_clip_pixels(image: Image.Image, size: int = 336) -> torch.Tensor:
    """Resize-shortest-edge + center crop, return [1,3,H,W] in [0,1]."""
    w, h = image.size
    scale = size / float(min(w, h))
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = image.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = torch.from_numpy(
        __import__("numpy").array(img).astype("float32") / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    return x


def normalize_clip(x01: torch.Tensor, device, dtype) -> torch.Tensor:
    mean = CLIP_MEAN.to(device=device, dtype=dtype)
    std = CLIP_STD.to(device=device, dtype=dtype)
    return (x01.to(device=device, dtype=dtype) - mean) / std


def tv_loss(x01: torch.Tensor) -> torch.Tensor:
    dh = (x01[:, :, 1:, :] - x01[:, :, :-1, :]).abs().mean()
    dw = (x01[:, :, :, 1:] - x01[:, :, :, :-1]).abs().mean()
    return dh + dw


def token_id_list(processor, phrases: List[str]) -> List[int]:
    ids = set()
    tok = processor.tokenizer
    for p in phrases:
        for variant in (p, " " + p, p.capitalize(), " " + p.capitalize()):
            pieces = tok.encode(variant, add_special_tokens=False)
            ids.update(int(x) for x in pieces)
    return sorted(i for i in ids if i >= 0)


REFUSE_WORDS = ["Sorry", "I", "As", "Unfortunately", "No", "Cannot", "Can't"]
COMPLY_WORDS = ["Sure", "Here", "Yes", "Okay", "OK", "Step", "1", "First"]


def margin_from_logits(logits: torch.Tensor, refuse_ids: List[int], comply_ids: List[int]) -> torch.Tensor:
    """Positive means more refusal than compliance (we minimize this to jailbreak)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    r = torch.logsumexp(logp[refuse_ids], dim=0)
    c = torch.logsumexp(logp[comply_ids], dim=0)
    return r - c

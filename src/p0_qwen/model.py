"""Qwen2-VL-7B wrapper: 4-bit LLM, fp16 vision, residual hooks, differentiable pixels."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from p0.model import COMPLY_WORDS, REFUSE_WORDS, margin_from_logits, token_id_list  # noqa: F401
from p0.prefixes import PREFIXES

from .vision import patchify_x01, pil_to_x01, to_square_pil


def qwen_layers(model) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("cannot find Qwen2-VL language layers")


class QwenP0:
    def __init__(self, cfg: Dict[str, Any], setting: str = "native") -> None:
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        path = cfg["model"]["local_path"]
        skip = list(cfg["model"].get("skip_modules", ["visual"]))
        free_gb = 0.0
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        # 4-bit Qwen+visual needs ~5.8GB. If the card is already occupied,
        # keep fp16 vision on GPU and the language model on CPU (bnb 4-bit
        # does not quantize CPU modules). Force GPU when resuming a 336px run.
        force_gpu = os.environ.get("P0_QWEN_FORCE_GPU", "").strip() in {"1", "true", "True"}
        self.cpu_llm = (free_gb < 6.5) and not force_gpu
        if force_gpu and free_gb < 6.5:
            print(f"P0_QWEN_FORCE_GPU=1 with cuda free={free_gb:.2f}GB; trying 4-bit GPU anyway", flush=True)
        quant = None
        device_map: Any = "auto"
        if self.cpu_llm:
            device_map = {
                "visual": 0,
                "model.embed_tokens": "cpu",
                "model.norm": "cpu",
                "model.rotary_emb": "cpu",
                "lm_head": "cpu",
            }
            for i in range(28):
                device_map[f"model.layers.{i}"] = "cpu"
            print(f"cuda free={free_gb:.2f}GB; visual GPU / language CPU fp16", flush=True)
        else:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip,
            )
            print(f"cuda free={free_gb:.2f}GB; 4-bit LLM + fp16 visual on GPU", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=True
        )
        keep_336 = os.environ.get("P0_QWEN_KEEP_336", "").strip() in {"1", "true", "True"}
        if self.cpu_llm and not keep_336:
            # 336px → 144 visual tokens; CPU 7B prefill is not feasible.
            # 140 is divisible by 28 and yields 25 visual tokens.
            size = 140
            print("cpu_llm: image_size 140 (25 visual tokens) for feasible prefill", flush=True)
        else:
            size = int(cfg["model"].get("image_size", 336))
            if self.cpu_llm:
                print("cpu_llm: keeping image_size 336 for attack protocol", flush=True)
        max_pixels = size * size
        min_pixels = size * size
        ip = self.processor.image_processor
        ip.max_pixels = max_pixels
        ip.min_pixels = min_pixels
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            path,
            quantization_config=quant,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=cfg["model"].get("attn_implementation", "eager"),
        )
        self.model.eval()
        self.model.requires_grad_(False)
        vis = getattr(self.model, "visual", None)
        self.max_new = int(cfg["model"]["max_new_tokens"])
        if self.cpu_llm:
            self.max_new = min(self.max_new, 24)
        self.hidden_size = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers)
        if vis is not None:
            self.visual_device = next(vis.parameters()).device
        else:
            self.visual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_device = torch.device("cpu") if self.cpu_llm else self.visual_device
        self.device = self.visual_device
        print(
            f"visual={self.visual_device} llm={self.llm_device} cpu_llm={self.cpu_llm} "
            f"map={getattr(self.model, 'hf_device_map', None)}",
            flush=True,
        )
        gc = getattr(self.model, "generation_config", None)
        if gc is not None:
            gc.do_sample = False
            gc.temperature = None
            gc.top_p = None
            gc.top_k = None
        self.layers = qwen_layers(self.model)
        self.image_size = size
        self.setting = setting
        self.prefixes = PREFIXES
        if self.cpu_llm:
            from accelerate.hooks import remove_hook_from_module, remove_hook_from_submodules

            remove_hook_from_module(self.model)
            remove_hook_from_submodules(self.model.model)
            if hasattr(self.model, "lm_head"):
                remove_hook_from_submodules(self.model.lm_head)
            self.model.model.to("cpu")
            self.model.lm_head.to("cpu")
            n_threads = max(8, (os.cpu_count() or 16) // 2)
            torch.set_num_threads(n_threads)
            print(
                f"stripped accelerate hooks from language model; LLM pinned to CPU; torch threads={n_threads}",
                flush=True,
            )

    def set_setting(self, setting: str) -> None:
        if setting not in {"native", "prefix"}:
            raise KeyError(f"unknown setting {setting}")
        self.setting = setting

    def prompt_messages(self, question: str) -> List[Dict[str, Any]]:
        user = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
        if self.setting == "native":
            return [user]
        sys_text = self.prefixes["A"].strip()
        return [{"role": "system", "content": sys_text}, user]

    def prompt_text(self, question: str) -> str:
        return self.processor.apply_chat_template(
            self.prompt_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )

    def image_to_x01(self, image: Image.Image) -> torch.Tensor:
        return pil_to_x01(image, size=self.image_size).to(self.device)

    def patchify(self, x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x01.to(device=self.device, dtype=torch.float16)
        pv, grid = patchify_x01(x)
        return pv, grid

    def encode(self, image: Image.Image, question: str) -> Dict[str, torch.Tensor]:
        pil = to_square_pil(image, size=self.image_size)
        text = self.prompt_text(question)
        packed = self.processor(text=[text], images=[pil], return_tensors="pt")
        x01 = pil_to_x01(pil, size=self.image_size).to(self.device)
        pv, grid = self.patchify(x01)
        out: Dict[str, torch.Tensor] = {
            "input_ids": packed["input_ids"].to(self.llm_device),
            "pixel_values": pv,
            "image_grid_thw": grid,
        }
        if "attention_mask" in packed:
            out["attention_mask"] = packed["attention_mask"].to(self.llm_device)
        return out

    def last_user_index(self, input_ids: torch.Tensor) -> int:
        return int(input_ids.shape[-1] - 1)

    def _apply_pixels(
        self,
        inputs: Dict[str, torch.Tensor],
        x01: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = dict(inputs)
        if x01 is not None:
            pv, grid = self.patchify(x01)
            out["pixel_values"] = pv
            out["image_grid_thw"] = grid
        elif pixel_values is not None:
            out["pixel_values"] = pixel_values.to(device=self.device, dtype=torch.float16)
            if image_grid_thw is not None:
                out["image_grid_thw"] = image_grid_thw.to(device=self.device)
        return out

    def _greedy(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Greedy decode without GenerationMixin device unification.

        Needed when vision is on CUDA and the language model is pinned to CPU.
        """
        input_ids = inputs["input_ids"]
        model_kwargs: Dict[str, Any] = {k: v for k, v in inputs.items() if k != "input_ids"}
        model_kwargs["use_cache"] = True
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos_set = {int(self.processor.tokenizer.eos_token_id)}
        elif isinstance(eos, (list, tuple)):
            eos_set = {int(x) for x in eos}
        else:
            eos_set = {int(eos)}
        n_in = int(input_ids.shape[-1])
        model_kwargs["cache_position"] = cache_position
        for step in range(max_new_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            outputs = self.model(**model_inputs, return_dict=True)
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True).to(input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )
            if "cache_position" not in model_kwargs:
                model_kwargs["cache_position"] = cache_position[-1:] + 1
            cache_position = model_kwargs["cache_position"]
            if int(next_token.item()) in eos_set:
                break
            if step == 0:
                print(f"greedy first token ok; remaining={max_new_tokens-1}", flush=True)
        _ = n_in
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: Optional[int] = None,
        x01: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> str:
        inputs = self._apply_pixels(self.encode(image, question), x01, pixel_values, image_grid_thw)
        n_in = int(inputs["input_ids"].shape[-1])
        max_new = max_new_tokens or self.max_new
        if self.cpu_llm or not do_sample:
            out = self._greedy(inputs, max_new) if self.cpu_llm else self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
            )
        else:
            gen_kw = dict(
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                use_cache=True,
            )
            if seed is not None:
                gen_kw["generator"] = torch.Generator(device=self.device).manual_seed(int(seed))
            out = self.model.generate(**inputs, **gen_kw)
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text.strip()

    def _hidden_at_layer(
        self,
        inputs: Dict[str, torch.Tensor],
        layer: int,
        token_index: int,
        patch: Optional[Tuple[str, torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
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
        x01: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        inputs = self._apply_pixels(self.encode(image, question), x01, pixel_values, image_grid_thw)
        tok = self.last_user_index(inputs["input_ids"])
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_i: int):
            def hook(_m, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                seq = hidden.shape[1]
                pos = tok if seq > 1 else 0
                if seq > 1 and tok >= seq:
                    pos = seq - 1
                captured[layer_i] = hidden[:, pos, :].detach().float().cpu().reshape(-1)
                return output

            return hook

        for layer in layers:
            handles.append(self.layers[layer].register_forward_hook(make_hook(int(layer))))
        try:
            self.model(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            for hdl in handles:
                hdl.remove()
        return {f"L{layer}:last_user": captured[int(layer)] for layer in layers}

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
        x01: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> str:
        inputs = self._apply_pixels(self.encode(image, question), x01, pixel_values, image_grid_thw)
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
            max_new = max_new_tokens or self.max_new
            if self.cpu_llm:
                out = self._greedy(inputs, max_new)
            else:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    use_cache=True,
                )
        finally:
            hdl.remove()
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text

    def first_token_logits(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "use_cache": False,
            "output_hidden_states": False,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out = self.model(**kwargs)
        logits = out.logits[0, -1]
        del out
        return logits

    def hidden_last_user(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        layer: int,
        attention_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        captured: Dict[str, torch.Tensor] = {}

        def hook(_m, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[:, -1, :]
            return output

        hdl = self.layers[layer].register_forward_hook(hook)
        try:
            kwargs: Dict[str, Any] = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "use_cache": False,
                "output_hidden_states": False,
            }
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            self.model(**kwargs)
        finally:
            hdl.remove()
        return captured["h"][0]

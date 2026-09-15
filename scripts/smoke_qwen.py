#!/usr/bin/env python3
"""Smoke-load local Qwen2-VL-7B-Instruct in 4-bit."""

import os
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

PATH = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"
IMG = "/root/autodl-tmp/V-Attack-main/datasets/coco300/val2017/000000000139.jpg"


def main() -> None:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual"],
    )
    print("loading Qwen2-VL-7B-Instruct 4-bit...", flush=True)
    processor = AutoProcessor.from_pretrained(PATH, local_files_only=True, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        PATH,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    image = Image.open(IMG).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one short sentence."}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32)
    n_in = int(inputs["input_ids"].shape[-1])
    gen = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        layers = getattr(model.language_model, "layers", None)
    rec = {
        "generate": gen.strip(),
        "n_layers": None if layers is None else len(layers),
        "vram_mb": float(torch.cuda.memory_allocated() / 1024 / 1024),
        "path": PATH,
    }
    print(rec, flush=True)


if __name__ == "__main__":
    main()

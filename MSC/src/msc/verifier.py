"""Official greedy decode + majority re-decode. Raw text never leaves this module."""

from __future__ import annotations

from typing import Any, Dict

from cgvsf.verifier_adapter import official_decode as _official_decode

from .states import safety_state


def official_decode(wrapper, item, x01, image, max_new_tokens: int = 96) -> Dict[str, Any]:
    pub = _official_decode(wrapper, item, x01, image, max_new_tokens=max_new_tokens)
    pub["safety_state"] = safety_state(pub)
    return pub


def majority_redecode(
    wrapper,
    item,
    x01,
    image,
    *,
    n: int,
    max_new_tokens: int = 96,
    majority_k: int | None = None,
) -> Dict[str, Any]:
    labels = []
    n_rhc = 0
    for _ in range(int(n)):
        tr = official_decode(wrapper, item, x01, image, max_new_tokens=max_new_tokens)
        labels.append(str(tr.get("safety_state")))
        n_rhc += int(bool(tr.get("core_rhc")))
    k = int(majority_k) if majority_k is not None else (int(n) // 2 + 1)
    return {
        "n": int(n),
        "n_rhc": int(n_rhc),
        "majority": int(n_rhc) >= k,
        "majority_k": k,
        "modes": labels,
    }

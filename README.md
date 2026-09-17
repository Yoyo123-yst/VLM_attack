# VLM attack method exploration

White-box visual jailbreak probes on local LVLMs. Each subdirectory is a separate method with its own freeze, gates, and reports. Sealed queries `h83–h130` are not read. Raw harmful text is not stored.

| Folder | Method | Status |
|---|---|---|
| [MSC](MSC/) | Mode-Switched Control: official-greedy mode → class-level phrase-set PGD | P0-1 `switched` **STOP** (skip on INVALID). P1 `switched_keep` **GO** (8/8 hard-8) |
| [V-CachePoll](V-CachePoll/) | Visual token squatting / cache pollution | P0–P3 CONTINUE |
| [CG-VSF](CG-VSF/) | Counterexample-guided certificate cutting | P0-A/P0-B **STOP** |
| [TraceFlip](TraceFlip/) | Trajectory repair vs GateFlip baseline | TraceFlip claim fails; GateFlip is the engineering baseline |

---

# CausalBottleneck-MJ

Preliminary verification of **CausalBottleneck-MJ**: whether a local residual safety subspace in a white-box LVLM is simultaneously causal, visually controllable, and selective.

Early tree also implements **P0: Visual Causal-Controllability Test**.

## Hardware adaptation (read this first)

The written P0 protocol asked for Qwen2-VL-7B. This machine cannot run that setup:

| Constraint | Available | Consequence |
|---|---|---|
| White-box checkpoint | LLaVA-1.5-7B-hf only (local HF cache) | P0 uses LLaVA, not Qwen2-VL |
| GPU | RTX 3080 Ti, 12 GB | 4-bit NF4 + first-token losses |
| Disk | ~14 GB free | no new 7B/8B download |

If P0 fails on LLaVA, that is a model-specific result, not a silent substitution. The later dual-model stage still wants Qwen2-VL + LLaVA when both weights fit.

## P0 hypothesis

There exists a residual subspace \(U\) such that

```
visual δ  →  Uᵀ h(I+δ, q)  →  safety-behavior change
```

and \(U\) is causal (bidirectional patching), controllable (bounded visual PGD), and selective (benign drop ≤ 5%).

## How to run

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda/envs/vattack

export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /root/autodl-tmp/multimodal_attack_project
python scripts/run_p0.py --stage smoke
python scripts/run_p0.py --stage mini
```

`--stage mini` is the first real verification (enough samples to accept/reject the hypothesis, small enough for 12 GB). `--stage full` is the protocol-size run after mini passes.

## Pipeline stages

1. `smoke` — load, generate, residual hook, image gradient
2. `probe` — keep only queries the clean model actually refuses
3. `traces` — collect REF / JB / FAIL / BENIGN
4. `patch` — coarse bidirectional residual patching
5. `subspace` — PCA on \(\Delta h = h^{JB}-h^{REF}\)
6. `causal` — low-rank \(U\) patch vs random / full residual
7. `attack` — universal visual attacks + baselines
8. `report` — tables, bootstrap CIs, go/no-go gates
9. `mini` / `full` — run 2–8 in order, skip finished stages

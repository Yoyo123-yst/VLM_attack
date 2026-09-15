# P0.5 Mechanism Robustness Report

LLaVA-1.5-7B. Same queries and carrier images as P0. Only the safety prefix changes in A.

Utility sign: \(\Delta U = U_{\text{patched}} - U_{\text{clean}}\). Positive = utility rose.

Qwen2-VL-7B-Instruct is on disk at `/root/autodl-tmp/models/Qwen2-VL-7B-Instruct` for later replication. This audit is still LLaVA.

## P0.5-A: template robustness (full residual, holdout)

| Setting | layer | dR REF→JB | dR JB→REF | ΔU benign |
|---|---:|---:|---:|---:|
| A→A (P0 reuse) | 20 | 0.636 | −1.000 | −0.074 |
| A→A | 24 | 0.727 | −0.977 | −0.074 |
| A→A | 28 | 0.818 | −0.909 | −0.074 |
| A→B | 20 | 0.568 | −1.000 | −0.074 |
| A→B | 24 | 0.659 | −0.977 | −0.074 |
| A→B | 28 | 0.727 | −0.886 | −0.074 |
| B→B | 20 | 1.000 | −0.800 | n/a |
| B→B | 24 | 1.000 | −0.800 | n/a |
| B→B | 28 | 1.000 | −1.000 | n/a |
| A+B→C, \(U\) r=8 | 20 | 0.477 | −0.455 | n/a |
| Random A→B | 20/24/28 | 0.000 | 0.000 | n/a |

Prefix B traces: 33/40 clean refusals, 26 JB (same JB count as A).

**A→B passes.** Full residual patching still has large bidirectional effects under a different wording of the same policy. Random directions stay at zero. The residual mediator is not locked to Prefix A’s token string.

Benign \(\Delta U=-0.074\) is a 7.4-point drop, slightly past the 5-point gate. Record it; do not hide it. Harmful patching is still selective enough that this is not “break the whole model,” but the 5% line is missed.

Claim allowed now:

> Under two semantically equivalent explicit safety prefixes, LLaVA-1.5 mid-to-late residual is a bidirectional safety-behavior mediator.

Still not allowed: “universal LVLM safety bottleneck” or cross-architecture sharing.

## P0.5-B: rank curve (L20 holdout)

| r | dR REF→JB | dR JB→REF |
|---:|---:|---:|
| 1 | 0.182 | 0.000 |
| 2 | 0.182 | −0.091 |
| 4 | 0.273 | −0.091 |
| 8 | 0.273 | −0.273 |
| 16 | 0.273 | −0.545 |
| 32 | 0.273 | −0.545 |
| full residual | 0.636 | −1.000 |
| random dir r=8 | 0.000 | 0.000 |
| cov-matched r=8 | 0.000 | 0.000 |

Knee: bidirectional necessity appears at **r=8** and saturates by **r=16**. Sufficiency plateaus at 0.27 from r=4, well below full residual 0.64.

Keep “local low-rank component” for r=8–16. Do **not** call the whole mediator a bottleneck: most of the REF→JB effect still lives outside rank 32.

## P0.5-C: per-sample visual entrance (new queries, n=23 REF, ε=16/255, 40 steps)

| Method | ASR | Refusal | Mean state shift to JB |
|---|---:|---:|---:|
| refusal-margin PGD | 0.522 | 0.391 | −2.45 |
| \(U\)-guided PGD | **0.739** | **0.174** | **+10.39** |
| random-subspace PGD | 0.000 | 1.000 | −13.14 |

\(\mathrm{corr}(\Delta U^\top h,\; R)=0.61\) for the U attack. Random never left REF, so its correlation is undefined.

\(U\) beats both the ordinary visual PGD and a random subspace on a held-out query pool. That is attack value for **instance-specific** visual optimization, not for a shared \(\delta_v\).

## Stage gate

```
主线：CausalBottleneck-MJ
P0.5-A：PASS（跨模板 full residual 成立）
P0.5-B：低维成分存在，但不是整层瓶颈
P0.5-C：U 对逐样本视觉优化有预测力
P1 双模型：仍暂停
universal visual：仍阻塞
下一阶段：P0-T 文本入口（固定、跨 query 复用的 s 能否推动同一个 U）
```

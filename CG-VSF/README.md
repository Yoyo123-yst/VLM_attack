# CG-VSF

Counterexample-Guided Visual Safety Falsification. P0 screens two hypotheses before any full method is built:

1. **P0-A**: accumulating real decode failure certificates beats last-certificate-only and refusal-margin, without a fixed `"Sure"` target.
2. **P0-B**: local visual control energy predicts 24-step certificate elimination better than gradient norm / margin.

TraceFlip stays frozen. GateFlip stays a strong engineering baseline, not the paper claim. Sealed queries `h83–h130` are not read.

## Status

**P0-A verdict: STOP. P0-B verdict: STOP.** 整晚分析见 [`out/EXPERIMENT_ANALYSIS.md`](out/EXPERIMENT_ANALYSIS.md)。

Hard 8-cell `core_rhc`：

| method | RHC |
|---|---|
| targeted_prefix_earlystop | 8/8 |
| refusal_margin | 7/8 |
| gateflip_fair | 6/8 |
| accumulated_certificate | 4/8 |
| last_certificate | 3/8 |
| accumulated_exact_only | 1/8 |

Token-level CE is not the attack. QP stays out of the attacker. Live status: `out/STATUS.md`.

Success is always `core_rhc`. Raw model text is never stored.

## Commands

```bash
python CG-VSF/tests/test_cgvsf_cpu.py
python CG-VSF/scripts/night_watch.py          # waits for GPU, then P0-A → P0-B
python CG-VSF/scripts/check_g0.py --freeze CG-VSF/CGVSF_P0_FROZEN.json
python CG-VSF/scripts/run_p0a.py --config CG-VSF/configs/p0a.yaml --resume
python CG-VSF/scripts/render_report.py --stage p0a
python CG-VSF/scripts/run_p0b.py --config CG-VSF/configs/p0b.yaml
python CG-VSF/scripts/advance_stage.py --require-preregistered-gate
```

P0-A uses the 8 TraceFlip-failed cells, 5 rounds × 24 backward, decode every round. GO requires **7/8** `core_rhc` and at least +1 vs `last_certificate`. **6/8 is never GO.**

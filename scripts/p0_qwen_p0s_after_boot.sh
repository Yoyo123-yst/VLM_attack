#!/bin/bash
# P0-S: collect RELATED_SAFE if needed, then development factor patching.
# Does not start Attack Gate or touch h91-h130 / h83-h90 confirm.
set -u
ROOT=/root/autodl-tmp/multimodal_attack_project
OUT="$ROOT/outputs/p0_qwen/full/native"
LOG="$OUT/p0s_run.log"
PY=/root/autodl-tmp/conda/envs/vattack/bin/python
LOCK="$OUT/p0s_run.lock"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "$(date -Is) already running pid=$(cat "$LOCK")"
  exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "======== $(date -Is) p0s start ========"
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "=== cpu tests ==="
"$PY" -u tests/test_p0_cpu.py || exit $?

echo "=== cpu-only p0s ==="
"$PY" -u scripts/run_p0_qwen_safety_axes.py --cpu-only --scale full --setting native || exit $?

echo "=== nvidia-smi ==="
nvidia-smi || true
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l | tr -d ' ')
echo "$(date -Is) used=${used}MiB free=${free}MiB compute_apps=${apps}"
if [ "${apps:-0}" = "0" ] && [ "${used:-0}" -ge 4500 ]; then
  echo "ERROR: leak without compute apps; not starting P0-S GPU"
  exit 2
fi
if [ -z "${free:-}" ] || [ "$free" -lt 10000 ]; then
  echo "ERROR: need >=10000 MiB free"
  exit 2
fi

echo "=== collect RELATED_SAFE on discover (blocked confirm+attack) then factor patch holdout ==="
"$PY" -u scripts/run_p0_qwen_safety_axes.py --collect-related-safe --gpu-patch --scale full --setting native
rc=$?
echo "p0s gpu exit=$rc"

if python3 - <<'PY'
import json
from pathlib import Path
c = json.loads(Path("/root/autodl-tmp/multimodal_attack_project/outputs/p0_qwen/full/native/causal.json").read_text())
a = json.loads(Path("/root/autodl-tmp/multimodal_attack_project/outputs/p0_qwen/full/native/attack.json").read_text())
assert c.get("attack_blocked") is True
assert not (c.get("best") or {}).get("pass_lowrank")
assert a.get("error") == "l24_cleaned_holdout_gate_failed"
print("official Attack Gate still closed")
PY
then
  echo "gate untouched: ok"
else
  echo "ERROR: P0-S mutated official gate artifacts"
  exit 5
fi

echo "======== $(date -Is) p0s done ========"
exit "$rc"

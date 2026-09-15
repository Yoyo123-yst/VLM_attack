#!/bin/bash
# FAIL diagnosis GPU job with VRAM/log monitoring. Does not start Attack Gate.
set -u
ROOT=/root/autodl-tmp/multimodal_attack_project
OUT="$ROOT/outputs/p0_qwen/full/native"
LOG="$OUT/fail_diag_run.log"
PY=/root/autodl-tmp/conda/envs/vattack/bin/python
LOCK="$OUT/fail_diag_run.lock"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "$(date -Is) already running pid=$(cat "$LOCK")"
  exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "======== $(date -Is) fail-diag start ========"
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "=== cpu tests ==="
"$PY" -u tests/test_p0_cpu.py || exit $?

echo "=== cpu-only fail diag ==="
"$PY" -u scripts/run_p0_qwen_fail_diag.py --cpu-only --scale full --setting native || exit $?

echo "=== nvidia-smi ==="
nvidia-smi || true
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l | tr -d ' ')
echo "$(date -Is) used=${used}MiB free=${free}MiB compute_apps=${apps}"
if [ "${apps:-0}" = "0" ] && [ "${used:-0}" -ge 4500 ]; then
  echo "ERROR: leak without compute apps; not starting GPU fail-diag"
  exit 2
fi
if [ -z "${free:-}" ] || [ "$free" -lt 10000 ]; then
  echo "ERROR: need >=10000 MiB free for diagnostic patching"
  exit 2
fi

echo "=== gpu posthoc holdout patch (not Attack Gate) ==="
"$PY" -u scripts/run_p0_qwen_fail_diag.py --gpu-patch --scale full --setting native
rc=$?
echo "gpu fail-diag exit=$rc"

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
  echo "ERROR: fail-diag mutated official gate artifacts"
  exit 5
fi

echo "======== $(date -Is) fail-diag done ========"
exit "$rc"

#!/bin/bash
# N0: freeze P0, build answer-conditioned pairs. Does not start N1/N2/N4 or old Attack Gate.
set -u
ROOT=/root/autodl-tmp/multimodal_attack_project
OUT="$ROOT/outputs/n0"
NATIVE="$ROOT/outputs/p0_qwen/full/native"
LOG="$OUT/n0_run.log"
PY=/root/autodl-tmp/conda/envs/vattack/bin/python
LOCK="$OUT/n0_run.lock"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "$(date -Is) already running pid=$(cat "$LOCK")"
  exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "======== $(date -Is) n0 start ========"
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "=== cpu tests ==="
"$PY" -u tests/test_n0_cpu.py || exit $?
"$PY" -u tests/test_p0_cpu.py || exit $?

echo "=== cpu-only n0 ==="
set +e
"$PY" -u scripts/run_n0_dataset.py --cpu-only
cpu_rc=$?
set -e
if [ "$cpu_rc" = "0" ]; then
  echo "N0 passed on existing traces; no GPU collect"
  echo "======== $(date -Is) n0 done cpu ========"
  exit 0
fi
if [ "$cpu_rc" != "4" ]; then
  echo "ERROR: n0 cpu-only unexpected exit=$cpu_rc"
  exit "$cpu_rc"
fi

echo "=== nvidia-smi ==="
nvidia-smi || true
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l | tr -d ' ')
echo "$(date -Is) used=${used}MiB free=${free}MiB compute_apps=${apps}"
if [ "${apps:-0}" = "0" ] && [ "${used:-0}" -ge 4500 ]; then
  echo "ERROR: leak without compute apps; not starting N0 GPU"
  exit 2
fi
if [ -z "${free:-}" ] || [ "$free" -lt 10000 ]; then
  echo "ERROR: need >=10000 MiB free"
  exit 2
fi

echo "=== collect matched RELATED_SAFE on same image/query (blocked h83-h90 and h91-h130) ==="
"$PY" -u scripts/run_n0_dataset.py --collect
rc=$?
echo "n0 gpu exit=$rc"

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
  echo "ERROR: N0 mutated official gate artifacts"
  exit 5
fi

echo "======== $(date -Is) n0 done ========"
exit "$rc"

#!/bin/bash
# After instance/container reboot: integrity, VRAM gate, patch/causal if needed,
# smoke-one, then full attack, then report.
set -u
ROOT=/root/autodl-tmp/multimodal_attack_project
OUT="$ROOT/outputs/p0_qwen/full/native"
LOG="$OUT/attack_run.log"
PY=/root/autodl-tmp/conda/envs/vattack/bin/python
LOCK="$OUT/attack_run.lock"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "$(date -Is) already running pid=$(cat "$LOCK")"
  exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "======== $(date -Is) boot attack start ========"
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
unset P0_QWEN_FORCE_GPU
unset P0_QWEN_KEEP_336
cd "$ROOT"

echo "=== cpu tests ==="
"$PY" -u tests/test_p0_cpu.py
cpu_rc=$?
if [ "$cpu_rc" -ne 0 ]; then
  echo "ERROR: cpu tests failed"
  exit "$cpu_rc"
fi

echo "=== integrity (CPU) ==="
"$PY" -u scripts/run_p0_qwen.py --stage integrity --scale full --setting native
int_rc=$?
if [ "$int_rc" -ne 0 ]; then
  echo "ERROR: integrity failed"
  exit "$int_rc"
fi

echo "=== nvidia-smi ==="
nvidia-smi || true
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l | tr -d ' ')
echo "$(date -Is) used=${used}MiB free=${free}MiB compute_apps=${apps}"
if [ -z "${used:-}" ] || [ -z "${free:-}" ]; then
  echo "ERROR: nvidia-smi did not return memory numbers"
  exit 2
fi
if [ "${apps:-0}" = "0" ] && [ "$used" -ge 4500 ]; then
  echo "ERROR: ${used}MiB used with no compute apps (driver/platform reservation). Do not start 5h attack."
  exit 2
fi

echo "waiting for GPU (>=10000 MiB free)..."
ok=0
for i in $(seq 1 90); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "$(date -Is) try=$i used=${used}MiB free=${free}MiB"
  if [ -n "${free:-}" ] && [ "$free" -ge 10000 ]; then
    ok=1
    break
  fi
  sleep 5
done
nvidia-smi || true
if [ "$ok" != 1 ]; then
  echo "ERROR: GPU still leaked after wait; not starting 140px/CPU fallback"
  exit 2
fi

need_patch=1
if python3 - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/multimodal_attack_project/outputs/p0_qwen/full/native/integrity.json")
if not p.exists():
    raise SystemExit(1)
rec = json.loads(p.read_text())
raise SystemExit(0 if rec.get("patch_cache_ok") else 1)
PY
then
  need_patch=0
fi
if [ "$need_patch" = "1" ]; then
  echo "=== patch (holdout RHC) ==="
  "$PY" -u scripts/run_p0_qwen.py --stage patch --scale full --setting native
  patch_rc=$?
  if [ "$patch_rc" -ne 0 ]; then
    echo "ERROR: patch failed"
    exit "$patch_rc"
  fi
fi

need_causal=1
if python3 - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/multimodal_attack_project/outputs/p0_qwen/full/native/integrity.json")
c = Path("/root/autodl-tmp/multimodal_attack_project/outputs/p0_qwen/full/native/causal.json")
if not p.exists() or not c.exists():
    raise SystemExit(1)
rec = json.loads(p.read_text())
causal = json.loads(c.read_text())
ok = bool(rec.get("causal_cache_ok")) and causal.get("postclean_holdout_eval") and (causal.get("best") or {}).get("pass_lowrank")
raise SystemExit(0 if ok else 1)
PY
then
  need_causal=0
fi
if [ "$need_causal" = "1" ]; then
  echo "=== causal (L24 new U, holdout) ==="
  "$PY" -u scripts/run_p0_qwen.py --stage causal --scale full --setting native
  causal_rc=$?
  if [ "$causal_rc" -ne 0 ]; then
    echo "ERROR: causal failed"
    exit "$causal_rc"
  fi
fi

if python3 - <<'PY'
import json
from pathlib import Path
c = Path("/root/autodl-tmp/multimodal_attack_project/outputs/p0_qwen/full/native/causal.json")
causal = json.loads(c.read_text())
if causal.get("attack_blocked") or not (causal.get("best") or {}).get("pass_lowrank"):
    raise SystemExit(4)
raise SystemExit(0)
PY
then
  echo "L24 cleaned-holdout gate pass"
else
  echo "ERROR: L24 cleaned-holdout gate failed; writing report and stopping attack"
  "$PY" -u scripts/run_p0_qwen.py --stage report --scale full --setting native || true
  exit 4
fi

echo "=== smoke-one ==="
"$PY" -u scripts/run_p0_qwen.py --stage attack --scale full --smoke-one --setting native
smoke_rc=$?
echo "smoke-one exit=$smoke_rc"
if [ "$smoke_rc" -ne 0 ]; then
  echo "ERROR: smoke-one failed; skip full attack"
  exit "$smoke_rc"
fi
if grep -q '"error"' "$OUT/attack_smoke.json" 2>/dev/null; then
  echo "ERROR: attack_smoke.json still has error; skip full attack"
  cat "$OUT/attack_smoke.json"
  exit 3
fi

echo "=== full attack ==="
"$PY" -u scripts/run_p0_qwen.py --stage attack --scale full --setting native
atk_rc=$?
echo "full attack exit=$atk_rc"

echo "=== report ==="
"$PY" -u scripts/run_p0_qwen.py --stage report --scale full --setting native
echo "======== $(date -Is) boot attack done ========"
exit "$atk_rc"

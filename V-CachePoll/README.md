# V-CachePoll

Visual token squatting / cache-pollution probes on multi-image pruning.

P0 uses the local **COCO val2017 + coco300 VQA** stand-in. TextVQA is not on disk yet.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda/envs/vattack
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /root/autodl-tmp/multimodal_attack_project/V-CachePoll
python tests/test_p0_cpu.py
python scripts/run_p0.py --stage pairs
python scripts/run_p0.py --stage smoke
python scripts/run_p0.py --stage screen
python scripts/run_p0.py --stage probe
python scripts/run_p1.py --stage smoke
python scripts/run_p1.py --stage attack
python scripts/run_p1.py --stage report
```

`pairs` is CPU-only. `smoke` loads Qwen2-VL-7B on 2 samples. Do not start with `p0` until smoke works.

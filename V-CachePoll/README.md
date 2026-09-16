# V-CachePoll

Visual token squatting / cache-pollution probes on multi-image pruning.

当前进展见 **[PROGRESS.md](PROGRESS.md)**。P0–P3 均判定 CONTINUE。P0 使用本地 **COCO val2017 + coco300 VQA** 替身，TextVQA 尚未上盘。

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
python scripts/run_p2.py --stage screen
python scripts/run_p2.py --stage crit
python scripts/run_p2.py --stage smoke
python scripts/run_p2.py --stage attack
python scripts/run_p2.py --stage report
```

`pairs` is CPU-only. `smoke` loads Qwen2-VL-7B on 2 samples. Do not start with `p0` until smoke works.

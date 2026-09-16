# Paused only while V-CachePoll holds the GPU.

CG-VSF `scripts/night_watch.py` waits for `V-CachePoll/out/GPU_LOCK` / `NIGHT_STATUS.md` to go idle, then starts P0-A on a clean JSONL (old GARBAGE-tagger rows archived). Do not launch a second `run_p0a.py` by hand.

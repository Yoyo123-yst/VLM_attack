"""Toy AMS (no VLM): finite-horizon walk, token-budget-matched baselines.

Process: reflecting walk on {0,...,H} plus absorbing FAIL.
From H-1 a decoy leak goes to FAIL, so height x/H ranks a low-q state high.
True committor q(t,x) is exact DP. Refusal analog = height. Random = noise.
Metric = unique lineages, not clones. Never imports a VLM.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .particles import Particle

ScoreFn = Callable[[int, int, np.random.Generator], float]
FAIL = -1


def committor_table(H: int, T: int, p_up: float, p_decoy: float = 0.0) -> np.ndarray:
    """q[t, x] = P(hit H). Absorb at H (1) and FAIL (0). Reflect at 0."""
    q = np.zeros((T + 1, H + 1), dtype=np.float64)
    q[:, H] = 1.0
    for t in range(T - 1, -1, -1):
        q[t, H] = 1.0
        for x in range(H):
            if x == H - 1 and p_decoy > 0.0:
                p_rest = 1.0 - p_decoy
                down = max(0, x - 1)
                q[t, x] = (p_rest * p_up) * q[t + 1, H] + (p_rest * (1.0 - p_up)) * q[t + 1, down]
            else:
                up = x + 1
                down = max(0, x - 1)
                q[t, x] = p_up * q[t + 1, up] + (1.0 - p_up) * q[t + 1, down]
    return q


def step_state(
    x: int,
    H: int,
    p_up: float,
    rng: np.random.Generator,
    p_decoy: float = 0.0,
) -> int:
    if x == H or x == FAIL:
        return x
    if x == H - 1 and p_decoy > 0.0 and rng.random() < p_decoy:
        return FAIL
    if rng.random() < p_up:
        return min(H, x + 1)
    return max(0, x - 1)


def score_committor(q: np.ndarray) -> ScoreFn:
    def fn(t: int, x: int, rng: np.random.Generator) -> float:
        if x == FAIL:
            return 0.0
        t = min(max(int(t), 0), q.shape[0] - 1)
        x = min(max(int(x), 0), q.shape[1] - 1)
        return float(q[t, x])

    return fn


def score_refusal(H: int) -> ScoreFn:
    """Time-agnostic height. Analog of first-token refusal_margin."""

    def fn(t: int, x: int, rng: np.random.Generator) -> float:
        if x == FAIL:
            return 0.0
        return float(x) / float(H)

    return fn


def score_random() -> ScoreFn:
    def fn(t: int, x: int, rng: np.random.Generator) -> float:
        return float(rng.random())

    return fn


def _step_one(
    p: Particle,
    H: int,
    T: int,
    p_up: float,
    rng: np.random.Generator,
    score_fn: ScoreFn,
    p_decoy: float = 0.0,
) -> int:
    if (not p.alive) or p.hit or p.t >= T or p.state == FAIL:
        if p.state == FAIL:
            p.alive = False
            p.killed = True
            p.score = 0.0
        return 0
    p.state = step_state(int(p.state), H, p_up, rng, p_decoy=p_decoy)
    p.t += 1
    if p.state == H:
        p.hit = True
        p.score = 1.0
        return 1
    if p.state == FAIL:
        p.alive = False
        p.killed = True
        p.score = 0.0
        return 1
    p.score = score_fn(p.t, int(p.state), rng)
    if p.t >= T and not p.hit:
        p.alive = False
        p.killed = True
    return 1


def ams_run(
    score_fn: ScoreFn,
    *,
    H: int = 10,
    T: int = 28,
    p_up: float = 0.42,
    p_decoy: float = 0.0,
    n_particles: int = 16,
    n_levels: int = 6,
    kill_frac: float = 0.25,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Quantile AMS. Levels are empirical score quantiles, not a fixed [0,1] grid.

    Async: each particle is advanced until it crosses the current quantile
    (or hits / dies). Ranking, not absolute calibration, is what is tested.
    """
    particles: list[Particle] = []
    next_id = 0
    tokens = 0
    warmup = max(2, T // (n_levels + 1))
    for i in range(n_particles):
        p = Particle(
            particle_id=next_id,
            lineage_id=i,
            parent_id=None,
            t=0,
            state=0,
            score=score_fn(0, 0, rng),
        )
        next_id += 1
        for _ in range(warmup):
            tokens += _step_one(p, H, T, p_up, rng, score_fn, p_decoy=p_decoy)
        particles.append(p)

    def resample(keep: list[Particle]) -> list[Particle]:
        nonlocal next_id
        if not keep:
            return []
        out = list(keep)
        while len(out) < n_particles:
            src = keep[int(rng.integers(0, len(keep)))]
            child = src.spawn(next_id)
            next_id += 1
            out.append(child)
        return out[:n_particles]

    for _lvl in range(n_levels):
        active = [p for p in particles if p.alive and not p.hit]
        if not active:
            break
        scores = np.array([p.score for p in active], dtype=np.float64)
        thresh = float(np.quantile(scores, 1.0 - min(0.9, max(0.1, kill_frac))))
        for p in active:
            guard = 0
            while p.alive and (not p.hit) and p.t < T and p.score < thresh - 1e-15:
                tokens += _step_one(p, H, T, p_up, rng, score_fn, p_decoy=p_decoy)
                guard += 1
                if guard > T + 2:
                    break
        survivors = [p for p in particles if p.hit or (p.alive and p.score >= thresh - 1e-15)]
        if not survivors:
            survivors = [p for p in particles if p.hit] or sorted(
                particles, key=lambda z: z.score, reverse=True
            )[: max(1, n_particles // 4)]
        particles = resample(survivors)

    for p in particles:
        while p.alive and (not p.hit) and p.t < T:
            tokens += _step_one(p, H, T, p_up, rng, score_fn, p_decoy=p_decoy)

    hits = [p for p in particles if p.hit]
    unique = {p.lineage_id for p in hits}
    return {
        "n_particles": n_particles,
        "n_hit_particles": len(hits),
        "n_unique_lineages": len(unique),
        "n_clone_hits": max(0, len(hits) - len(unique)),
        "tokens": tokens,
        "hit_rate_particles": len(hits) / max(n_particles, 1),
    }


def naive_run(
    *,
    H: int = 10,
    T: int = 28,
    p_up: float = 0.42,
    p_decoy: float = 0.0,
    n_traj: int | None = 16,
    token_budget: int | None = None,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    tokens = 0
    n_hit = 0
    n_done = 0
    if token_budget is None and n_traj is None:
        raise ValueError("need n_traj or token_budget")
    while True:
        if token_budget is None and n_done >= int(n_traj):
            break
        if token_budget is not None and tokens >= int(token_budget) and n_done >= 1:
            break
        x = 0
        hit = False
        for _t in range(T):
            x = step_state(x, H, p_up, rng, p_decoy=p_decoy)
            tokens += 1
            if x == H:
                hit = True
                break
            if x == FAIL:
                break
            if token_budget is not None and tokens >= int(token_budget) and n_done >= 0:
                break
        n_done += 1
        if hit:
            n_hit += 1
        if token_budget is None and n_done >= int(n_traj):
            break
        if n_done > 10_000:
            break
    return {
        "n_particles": n_done,
        "n_hit_particles": n_hit,
        "n_unique_lineages": n_hit,
        "n_clone_hits": 0,
        "tokens": tokens,
        "hit_rate_particles": n_hit / max(n_done, 1),
    }


def bon_run(
    *,
    H: int = 10,
    T: int = 28,
    p_up: float = 0.42,
    p_decoy: float = 0.0,
    token_budget: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Independent full paths under the same token budget. No splitting."""
    out = naive_run(
        H=H, T=T, p_up=p_up, p_decoy=p_decoy, n_traj=None, token_budget=token_budget, rng=rng
    )
    out["method"] = "bon"
    return out


def budget_matched_once(
    rng: np.random.Generator,
    *,
    H: int = 8,
    T: int = 40,
    p_up: float = 0.48,
    p_decoy: float = 0.75,
    n_particles: int = 24,
    n_levels: int = 5,
) -> dict[str, dict[str, float | int]]:
    q = committor_table(H, T, p_up, p_decoy)
    kw = dict(
        H=H,
        T=T,
        p_up=p_up,
        p_decoy=p_decoy,
        n_particles=n_particles,
        n_levels=n_levels,
    )
    full = ams_run(score_committor(q), rng=rng, **kw)
    refusal = ams_run(score_refusal(H), rng=rng, **kw)
    rnd = ams_run(score_random(), rng=rng, **kw)
    budget = int(full["tokens"])
    naive = naive_run(
        H=H, T=T, p_up=p_up, p_decoy=p_decoy, n_traj=None, token_budget=budget, rng=rng
    )
    bon = bon_run(H=H, T=T, p_up=p_up, p_decoy=p_decoy, token_budget=budget, rng=rng)
    return {
        "naive": naive,
        "bon": bon,
        "ams_full": full,
        "ams_refusal": refusal,
        "ams_random": rnd,
        "target_tokens_naive": {"tokens": budget},
    }


def summarize_replicates(rows: Sequence[dict[str, dict[str, float | int]]]) -> dict[str, Any]:
    methods = ("naive", "bon", "ams_full", "ams_refusal", "ams_random")
    out: dict[str, Any] = {"n_rep": len(rows), "methods": {}}
    for m in methods:
        uniq = np.array([float(r[m]["n_unique_lineages"]) for r in rows])
        hits = np.array([float(r[m]["n_hit_particles"]) for r in rows])
        toks = np.array([float(r[m]["tokens"]) for r in rows])
        clones = np.array([float(r[m]["n_clone_hits"]) for r in rows])
        out["methods"][m] = {
            "unique_mean": float(uniq.mean()),
            "unique_std": float(uniq.std(ddof=1) if len(uniq) > 1 else 0.0),
            "hit_particles_mean": float(hits.mean()),
            "tokens_mean": float(toks.mean()),
            "clone_hits_mean": float(clones.mean()),
        }
    u_full = out["methods"]["ams_full"]["unique_mean"]
    u_ref = out["methods"]["ams_refusal"]["unique_mean"]
    u_rnd = out["methods"]["ams_random"]["unique_mean"]
    u_naive = out["methods"]["naive"]["unique_mean"]
    beats_random = u_full > u_rnd + 1e-9
    beats_naive = u_full > u_naive + 1e-9
    beats_refusal = u_full > u_ref + 1e-9
    only_random = beats_random and (not beats_refusal)
    gate = "GO" if (beats_random and beats_naive) else "STOP"
    out["gate"] = gate
    out["beats_random"] = bool(beats_random)
    out["beats_naive"] = bool(beats_naive)
    out["beats_refusal"] = bool(beats_refusal)
    out["only_beats_random_not_refusal"] = bool(only_random)
    out["note"] = (
        "Metric is unique lineages, not cloned particles. "
        "Toy GO if AMS-full beats naive and random. "
        "VLM Stage 2 STOP if committor beats random but not refusal-score."
    )
    return out


def run_toy(
    n_rep: int = 48,
    seed: int = 20260,
    **kwargs: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = [budget_matched_once(rng, **kwargs) for _ in range(n_rep)]
    summary = summarize_replicates(rows)
    summary["seed"] = seed
    summary["kwargs"] = {
        "H": kwargs.get("H", 8),
        "T": kwargs.get("T", 40),
        "p_up": kwargs.get("p_up", 0.48),
        "p_decoy": kwargs.get("p_decoy", 0.75),
        "n_particles": kwargs.get("n_particles", 24),
        "n_levels": kwargs.get("n_levels", 5),
    }
    return summary

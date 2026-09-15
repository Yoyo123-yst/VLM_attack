"""Particle / lineage records for AMS. Toy and later VLM share this schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Particle:
    particle_id: int
    lineage_id: int
    parent_id: int | None
    t: int
    state: int
    score: float
    alive: bool = True
    hit: bool = False
    killed: bool = False
    log_weight: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def spawn(self, new_id: int) -> "Particle":
        child = Particle(
            particle_id=new_id,
            lineage_id=self.lineage_id,
            parent_id=self.particle_id,
            t=self.t,
            state=self.state,
            score=self.score,
            alive=self.alive,
            hit=self.hit,
            killed=self.killed,
            log_weight=self.log_weight,
            meta=dict(self.meta),
        )
        return child


    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("meta", None)
        return d

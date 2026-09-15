"""Attack-cost ledger. Compare methods on generated tokens and wall time."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BudgetLedger:
    victim_generations: int = 0
    generated_tokens: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    judge_calls: int = 0
    wall_clock_seconds: float = 0.0
    peak_gpu_memory: float = 0.0

    def add_generation(self, tokens: int, wall_s: float = 0.0) -> None:
        self.victim_generations += 1
        self.generated_tokens += max(int(tokens), 0)
        self.wall_clock_seconds += float(wall_s)

    def add_forward(self, n: int = 1) -> None:
        self.forward_passes += int(n)

    def add_backward(self, n: int = 1) -> None:
        self.backward_passes += int(n)

    def add_judge(self, n: int = 1) -> None:
        self.judge_calls += int(n)

    def merge(self, other: "BudgetLedger") -> "BudgetLedger":
        return BudgetLedger(
            victim_generations=self.victim_generations + other.victim_generations,
            generated_tokens=self.generated_tokens + other.generated_tokens,
            forward_passes=self.forward_passes + other.forward_passes,
            backward_passes=self.backward_passes + other.backward_passes,
            judge_calls=self.judge_calls + other.judge_calls,
            wall_clock_seconds=self.wall_clock_seconds + other.wall_clock_seconds,
            peak_gpu_memory=max(self.peak_gpu_memory, other.peak_gpu_memory),
        )

    def as_dict(self) -> dict:
        return asdict(self)

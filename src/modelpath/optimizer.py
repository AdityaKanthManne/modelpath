"""Quality-constrained model selection using measured benchmark records.

This selects from supplied measurements, not predictions of future performance.
The caller is responsible for representative held-out evaluation and hardware.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBenchmark:
    name: str
    task: str
    accuracy: float
    p50_ms: float
    peak_memory_mb: float

    def __post_init__(self):
        if not self.name or not self.task or not 0 <= self.accuracy <= 1:
            raise ValueError("Invalid benchmark identity or accuracy")
        if self.p50_ms <= 0 or self.peak_memory_mb <= 0:
            raise ValueError("Latency and memory must be positive")


def choose_model(benchmarks: list[ModelBenchmark], *, task: str,
                 min_accuracy: float = 0.9, memory_budget_mb: float | None = None) -> ModelBenchmark:
    """Minimize measured P50 latency subject to quality and memory constraints."""
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be in [0, 1]")
    if memory_budget_mb is not None and memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be positive")
    candidates = [b for b in benchmarks if b.task == task and b.accuracy >= min_accuracy
                  and (memory_budget_mb is None or b.peak_memory_mb <= memory_budget_mb)]
    if not candidates:
        raise ValueError("No benchmarked model satisfies the constraints; do not silently degrade quality")
    return min(candidates, key=lambda b: (b.p50_ms, b.peak_memory_mb, b.name))

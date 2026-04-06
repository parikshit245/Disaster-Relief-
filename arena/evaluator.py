"""
arena/evaluator.py — Base Strategy + Evaluator
------------------------------------------------
AlgorithmResult   : dataclass holding every measured output.
AlgorithmStrategy : ABC every algorithm wrapper must implement.
Evaluator         : runs N strategies (in parallel threads), measures
                    time + memory, picks a winner by weighted composite score.
"""
from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, List


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlgorithmResult:
    """Single algorithm's measured output for one pipeline stage."""
    algorithm_name: str
    output: Any                          # ranked list / paths / allocations / assignments
    exec_time_ms: float                  # wall-clock milliseconds
    memory_kb: float                     # peak tracemalloc delta in KB
    quality_score: float                 # domain-specific metric (higher = better unless inverted)
    quality_label: str = ""              # human-readable label for quality (e.g. "Total Value")
    lower_quality_is_better: bool = False  # True for routing (distance)
    composite_score: float = 0.0         # filled in by Evaluator.pick_winner()
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract strategy
# ─────────────────────────────────────────────────────────────────────────────

class AlgorithmStrategy(ABC):
    """
    Every algorithm wrapper must subclass this.

    Subclasses implement _execute() which does the real work.
    The public run() method wraps _execute() with timing + memory measurement
    so no strategy needs to instrument itself.
    """

    name: str = "unnamed"

    def run(self, *args, **kwargs) -> AlgorithmResult:
        """Measure time + memory, call _execute(), return AlgorithmResult."""
        tracemalloc.start()
        t0 = time.perf_counter()

        result = self._execute(*args, **kwargs)

        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.exec_time_ms = (t1 - t0) * 1000
        result.memory_kb    = peak / 1024
        result.algorithm_name = self.name
        return result

    @abstractmethod
    def _execute(self, *args, **kwargs) -> AlgorithmResult:
        """Override: run the algorithm, return an AlgorithmResult with output + quality_score."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs multiple AlgorithmStrategy instances (in parallel threads),
    normalizes their metrics, and selects a winner by weighted composite score.

    Parameters
    ----------
    strategies     : list of AlgorithmStrategy subclass instances
    weight_time    : how much to favour faster algorithms  [0–1]
    weight_memory  : how much to favour lower memory usage [0–1]
    weight_quality : how much to favour better solution quality [0–1]
                     (weights are re-normalized to sum=1 internally)
    """

    def __init__(
        self,
        strategies: List[AlgorithmStrategy],
        weight_time: float    = 0.3,
        weight_memory: float  = 0.2,
        weight_quality: float = 0.5,
    ):
        self.strategies = strategies
        # Normalize weights to sum = 1
        total = weight_time + weight_memory + weight_quality + 1e-9
        self.w_time    = weight_time    / total
        self.w_memory  = weight_memory  / total
        self.w_quality = weight_quality / total

    # ── parallel execution ────────────────────────────────────────────────

    def run_all(self, *args, **kwargs) -> List[AlgorithmResult]:
        """
        Execute every strategy concurrently.
        Returns list of AlgorithmResult in the same order as self.strategies.
        """
        results: List[AlgorithmResult] = [None] * len(self.strategies)

        with ThreadPoolExecutor(max_workers=len(self.strategies)) as pool:
            future_to_idx = {
                pool.submit(s.run, *args, **kwargs): i
                for i, s in enumerate(self.strategies)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    # Build a placeholder result so the UI does not crash
                    results[idx] = AlgorithmResult(
                        algorithm_name=self.strategies[idx].name,
                        output=None,
                        exec_time_ms=0.0,
                        memory_kb=0.0,
                        quality_score=0.0,
                        metadata={"error": str(exc)},
                    )
        return results

    # ── winner selection ──────────────────────────────────────────────────

    def pick_winner(self, results: List[AlgorithmResult]) -> AlgorithmResult:
        """
        Min-max normalize Speed, Memory, Quality for each result,
        then compute composite = w_t*speed + w_m*memory + w_q*quality.
        Returns the result with the highest composite score (also mutates
        each result's composite_score field for display).
        """
        eps = 1e-9

        times    = [r.exec_time_ms  for r in results]
        mems     = [r.memory_kb     for r in results]
        quals    = [r.quality_score for r in results]

        min_t, max_t = min(times),  max(times)
        min_m, max_m = min(mems),   max(mems)
        min_q, max_q = min(quals),  max(quals)

        # Invert quality for stages where lower is better (e.g. routing distance)
        invert_quality = any(r.lower_quality_is_better for r in results)

        for r in results:
            norm_speed  = 1.0 - (r.exec_time_ms - min_t) / (max_t - min_t + eps)
            norm_memory = 1.0 - (r.memory_kb    - min_m) / (max_m - min_m + eps)
            if invert_quality:
                norm_quality = 1.0 - (r.quality_score - min_q) / (max_q - min_q + eps)
            else:
                norm_quality = (r.quality_score - min_q) / (max_q - min_q + eps)

            r.composite_score = (
                self.w_time    * norm_speed  +
                self.w_memory  * norm_memory +
                self.w_quality * norm_quality
            )

        return max(results, key=lambda r: r.composite_score)

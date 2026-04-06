"""
arena/algorithms/allocation.py — Resource Allocation Strategies
---------------------------------------------------------------
FractionalKnapsack : greedy by value/weight ratio — O(n log n), divisible items
DP01Knapsack       : exact 0/1 DP — O(n x W), integer/indivisible items

Both wrap the existing resource_allocator.py logic.
Quality metric: total value of allocated resources (higher = better).
"""
from __future__ import annotations

import os
import sys
from typing import Any, List

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODULES = os.path.join(_ROOT, "python modules")
for p in (_ROOT, _MODULES):
    if p not in sys.path:
        sys.path.insert(0, p)

from resource_allocator import resource_allocator as _ResourceAllocator

from arena.evaluator import AlgorithmResult, AlgorithmStrategy


# ── Strategy 1: Fractional Knapsack ──────────────────────────────────────────

class FractionalKnapsack(AlgorithmStrategy):
    """
    Greedy: sort resources by value/weight ratio, take as much as possible.
    Divisible items — may take fractional quantities.
    O(n log n) time, O(n) space.
    """
    name = "Fractional Knapsack"

    def _execute(self, resources: List[dict], capacity: float, **kwargs) -> AlgorithmResult:
        ra = _ResourceAllocator(resources, capacity, use_dp=False)
        total_val = ra.total_value()
        total_wt  = ra.total_weight()

        return AlgorithmResult(
            algorithm_name=self.name,
            output=ra.allocated_resources,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_val,
            quality_label="Total Allocated Value",
            lower_quality_is_better=False,
            metadata={
                "total_weight_kg": round(total_wt, 2),
                "total_value":     round(total_val, 2),
                "items_selected":  len(ra.allocated_resources),
                "capacity_kg":     capacity,
            },
        )


# ── Strategy 2: 0/1 Knapsack DP ──────────────────────────────────────────────

class DP01Knapsack(AlgorithmStrategy):
    """
    Exact DP: each resource type taken as a whole (all-or-nothing).
    O(n * W) time where W = int(capacity * 10), O(n * W) space.
    """
    name = "0/1 Knapsack DP"

    def _execute(self, resources: List[dict], capacity: float, **kwargs) -> AlgorithmResult:
        ra = _ResourceAllocator(resources, capacity, use_dp=True)
        total_val = ra.total_value()
        total_wt  = ra.total_weight()

        return AlgorithmResult(
            algorithm_name=self.name,
            output=ra.allocated_resources,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_val,
            quality_label="Total Allocated Value",
            lower_quality_is_better=False,
            metadata={
                "total_weight_kg": round(total_wt, 2),
                "total_value":     round(total_val, 2),
                "items_selected":  len(ra.allocated_resources),
                "capacity_kg":     capacity,
            },
        )

"""
arena/algorithms/prioritization.py — Request Prioritization Strategies
-----------------------------------------------------------------------
GreedyMaxHeap : uses a heapq max-heap to extract requests in priority order
SimpleSort    : uses Python's built-in TimSort on the same scoring function

Both algorithms apply the identical scoring formula from priority_agent.py,
so outputs are equivalent. The race demonstrates heap vs sort tradeoffs.
"""
from __future__ import annotations

import heapq
import os
import sys
from typing import Any, List

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODULES = os.path.join(_ROOT, "python modules")
for p in (_ROOT, _MODULES):
    if p not in sys.path:
        sys.path.insert(0, p)

from priority_agent import priority_agent as _PriorityAgent, ranked_request

from arena.evaluator import AlgorithmResult, AlgorithmStrategy


# ── Shared scoring helper ─────────────────────────────────────────────────────

_SEV_WEIGHTS = {"low": 1, "medium": 2, "high": 3, "critical": 4}

def _compute_score(req: dict) -> float:
    sev   = _SEV_WEIGHTS.get(req.get("severity", "low").lower(), 0)
    ppl   = min(req.get("people_affected", 0) / 100, 10)
    dead  = max(10 - req.get("deadline_hours", 24), 0)
    dist  = max(10 - req.get("distance_km", 0) / 10, 0)
    return round((sev * 4) + (ppl * 3) + (dead * 2) + (dist * 1), 2)


# ── Strategy 1: Greedy Max-Heap ───────────────────────────────────────────────

class GreedyMaxHeap(AlgorithmStrategy):
    """
    Score all requests then push (score, req) onto a max-heap
    (implemented as a min-heap on negated scores).
    Extract in order to produce ranked list.
    O(n log n) time, O(n) space.
    """
    name = "Greedy Max-Heap"

    def _execute(self, requests: List[dict], **kwargs) -> AlgorithmResult:
        heap: list = []
        for req in requests:
            score = _compute_score(req)
            # negate score for max-heap behaviour using Python's min-heap
            heapq.heappush(heap, (-score, req["id"], req, score))

        ranked: List[dict] = []
        total_score = 0.0
        while heap:
            neg_s, _, req, score = heapq.heappop(heap)
            ranked.append({**req, "score": score})
            total_score += score

        return AlgorithmResult(
            algorithm_name=self.name,
            output=ranked,
            exec_time_ms=0.0,    # filled by run()
            memory_kb=0.0,       # filled by run()
            quality_score=total_score,
            quality_label="Total Priority Score",
            lower_quality_is_better=False,
            metadata={"n_requests": len(requests)},
        )


# ── Strategy 2: Simple Sort (TimSort) ─────────────────────────────────────────

class SimpleSort(AlgorithmStrategy):
    """
    Score all requests into a list then call list.sort() (Python TimSort).
    O(n log n) time, O(n) space — same asymptotic complexity as heap
    but different constant factors and cache behaviour.
    """
    name = "Simple Sort (TimSort)"

    def _execute(self, requests: List[dict], **kwargs) -> AlgorithmResult:
        scored = [({**req, "score": _compute_score(req)}) for req in requests]
        scored.sort(key=lambda r: r["score"], reverse=True)

        total_score = sum(r["score"] for r in scored)

        return AlgorithmResult(
            algorithm_name=self.name,
            output=scored,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_score,
            quality_label="Total Priority Score",
            lower_quality_is_better=False,
            metadata={"n_requests": len(requests)},
        )

"""
arena/algorithms/assignment.py — Team Assignment Strategies
-----------------------------------------------------------
BacktrackingAssigner : exhaustive BT with bound pruning — wraps team_assigner.py
BranchBoundAssigner  : exact B&B with LP upper bound   — wraps optimizer.py

Quality metric: total assignment score (higher = better).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODULES = os.path.join(_ROOT, "python modules")
for p in (_ROOT, _MODULES):
    if p not in sys.path:
        sys.path.insert(0, p)

from team_assigner import Team, TeamAssigner, _score, _is_compatible
from optimizer     import BranchAndBound

from arena.evaluator import AlgorithmResult, AlgorithmStrategy


# ── Shared value-matrix builder ───────────────────────────────────────────────

def _build_value_matrix(
    teams: List[Team], requests: List[dict]
) -> Tuple[List[List[float]], List[str], List[str]]:
    """Returns (value_matrix, team_ids, req_ids)."""
    vm = [
        [_score(t, r) if _is_compatible(t, r) else 0.0 for t in teams]
        for r in requests
    ]
    return vm, [t.id for t in teams], [r["id"] for r in requests]


# ── Strategy 1: Backtracking ──────────────────────────────────────────────────

class BacktrackingAssigner(AlgorithmStrategy):
    """
    Exhaustive backtracking with greedy upper-bound pruning.
    Wraps the existing TeamAssigner class.
    Complexity: O(n!) worst case, dramatically pruned in practice.
    """
    name = "Backtracking"

    def _execute(
        self,
        teams: List[dict],
        requests: List[dict],
        budget: float = 1e9,
        **kwargs,
    ) -> AlgorithmResult:
        team_objs  = [Team(**t) for t in teams if t.get("available", True)]
        assigner   = TeamAssigner(team_objs, requests)
        assignments = assigner.assign()

        total_score = sum(a.score for a in assignments)
        assign_map  = {a.request_id: a.team_id for a in assignments}

        return AlgorithmResult(
            algorithm_name=self.name,
            output=assign_map,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_score,
            quality_label="Total Assignment Score",
            lower_quality_is_better=False,
            metadata={
                "n_assigned":  len(assignments),
                "n_requests":  len(requests),
                "total_score": round(total_score, 2),
            },
        )


# ── Strategy 2: Branch & Bound ────────────────────────────────────────────────

class BranchBoundAssigner(AlgorithmStrategy):
    """
    Globally optimal assignment under a budget hard constraint.
    Uses LP relaxation as the upper bound for aggressive pruning.
    Wraps the existing BranchAndBound class from optimizer.py.
    """
    name = "Branch & Bound"

    def _execute(
        self,
        teams: List[dict],
        requests: List[dict],
        budget: float = 1000.0,
        **kwargs,
    ) -> AlgorithmResult:
        team_objs = [Team(**t) for t in teams if t.get("available", True)]
        vm, team_ids, req_ids = _build_value_matrix(team_objs, requests)
        costs = [t.deploy_cost for t in team_objs]

        bnb = BranchAndBound(vm, costs, budget, team_ids, req_ids)
        best_val, assign_map = bnb.optimize()

        return AlgorithmResult(
            algorithm_name=self.name,
            output=assign_map,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=best_val,
            quality_label="Total Assignment Score",
            lower_quality_is_better=False,
            metadata={
                "n_assigned":  len(assign_map),
                "n_requests":  len(requests),
                "total_score": round(best_val, 2),
                "budget":      budget,
            },
        )

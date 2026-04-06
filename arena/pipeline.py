"""
arena/pipeline.py — ArenaPipeline & BenchmarkReport
-----------------------------------------------------
ArenaPipeline orchestrates the 4-stage race:
  Stage 1 ── Prioritization     (GreedyMaxHeap vs SimpleSort)
  Stage 2 ── Allocation         (FractionalKnapsack vs DP01Knapsack)
  Stage 3 ── Route Planning     (Dijkstra vs BellmanFord vs A*)
  Stage 4 ── Team Assignment    (Backtracking vs Branch & Bound)

Each stage runs all strategies IN PARALLEL, measures time + memory,
picks a winner by weighted composite score, and feeds that winner's
output as input to the next stage.

The DataStore singleton is NEVER mutated — a deep copy is used.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULES = os.path.join(_ROOT, "python modules")
for p in (_ROOT, _MODULES):
    if p not in sys.path:
        sys.path.insert(0, p)

from arena.evaluator import AlgorithmResult, Evaluator
from arena.algorithms.prioritization import GreedyMaxHeap, SimpleSort
from arena.algorithms.allocation     import FractionalKnapsack, DP01Knapsack
from arena.algorithms.routing        import DijkstraRouter, BellmanFordRouter, AStarRouter
from arena.algorithms.assignment     import BacktrackingAssigner, BranchBoundAssigner

from team_assigner import Team, _score, _is_compatible


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DispatchRow:
    """One row of the final assembled dispatch plan."""
    rank:             int
    request_id:       str
    location:         str
    severity:         str
    people:           int
    priority_score:   float
    team_id:          Optional[str]
    team_name:        Optional[str]
    route:            List[str]
    distance_km:      float
    resources:        List[Any]
    resource_value:   float
    assignment_score: float


@dataclass
class BenchmarkReport:
    """Full output of one ArenaPipeline.run() call."""
    # Per-stage results
    prioritization_results: List[AlgorithmResult] = field(default_factory=list)
    prioritization_winner:  Optional[AlgorithmResult] = None

    allocation_results:     List[AlgorithmResult] = field(default_factory=list)
    allocation_winner:      Optional[AlgorithmResult] = None

    routing_results:        List[AlgorithmResult] = field(default_factory=list)
    routing_winner:         Optional[AlgorithmResult] = None

    assignment_results:     List[AlgorithmResult] = field(default_factory=list)
    assignment_winner:      Optional[AlgorithmResult] = None

    # Final assembled dispatch plan
    dispatch_rows:          List[DispatchRow] = field(default_factory=list)
    unassigned:             List[str]         = field(default_factory=list)

    # Aggregate metrics
    total_assignment_score: float = 0.0
    total_resource_value:   float = 0.0
    total_pipeline_ms:      float = 0.0

    # Config echoed back for display
    capacity_kg:  float = 500.0
    budget_units: float = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ArenaPipeline:
    """
    Multi-algorithm benchmarking pipeline.

    Parameters
    ----------
    store        : DataStore singleton — deep-copied internally, never mutated.
    weight_time  : scoring weight for execution speed     (default 0.3)
    weight_memory: scoring weight for memory efficiency    (default 0.2)
    weight_quality: scoring weight for solution quality    (default 0.5)
    capacity_kg  : carrier weight limit for resource allocation
    budget_units : deployment budget for team assignment
    """

    def __init__(
        self,
        store: Any,
        weight_time:    float = 0.3,
        weight_memory:  float = 0.2,
        weight_quality: float = 0.5,
        capacity_kg:    float = 500.0,
        budget_units:   float = 1000.0,
    ):
        self.store          = store
        self.w_time         = weight_time
        self.w_memory       = weight_memory
        self.w_quality      = weight_quality
        self.capacity_kg    = capacity_kg
        self.budget_units   = budget_units

    def _evaluator(self, strategies):
        return Evaluator(
            strategies,
            weight_time    = self.w_time,
            weight_memory  = self.w_memory,
            weight_quality = self.w_quality,
        )

    # ── public entry point ────────────────────────────────────────────────

    def run(self) -> BenchmarkReport:
        t_pipeline_start = time.perf_counter()

        # Deep-copy so we never touch the singleton's state
        snap = copy.deepcopy(self.store)

        report = BenchmarkReport(
            capacity_kg  = self.capacity_kg,
            budget_units = self.budget_units,
        )

        if not snap.requests:
            return report

        # ── Stage 1: Prioritization ───────────────────────────────────────
        ev1 = self._evaluator([GreedyMaxHeap(), SimpleSort()])
        p_results = ev1.run_all(requests=snap.requests)
        p_winner  = ev1.pick_winner(p_results)

        report.prioritization_results = p_results
        report.prioritization_winner  = p_winner

        ranked_requests: List[dict] = p_winner.output or []

        # ── Stage 2: Resource Allocation ─────────────────────────────────
        ev2 = self._evaluator([FractionalKnapsack(), DP01Knapsack()])
        a_results = ev2.run_all(resources=snap.resources, capacity=self.capacity_kg)
        a_winner  = ev2.pick_winner(a_results)

        report.allocation_results = a_results
        report.allocation_winner  = a_winner

        allocated_resources = a_winner.output or []
        resource_value      = sum(
            getattr(r, "value", 0) * getattr(r, "quantity", 1)
            for r in allocated_resources
        )

        # ── Stage 3: Route Planning ───────────────────────────────────────
        # We need an assignment map for routing — use a quick greedy pass
        # (the real assignment race comes next; routing needs src→dst pairs)
        _quick_assign = self._quick_greedy_assign(snap.teams, ranked_requests)

        if snap.graph_edges:
            ev3 = self._evaluator([DijkstraRouter(), BellmanFordRouter(), AStarRouter()])
            r_results = ev3.run_all(
                graph_edges=snap.graph_edges,
                assignments=_quick_assign,
                teams=snap.teams,
                requests=ranked_requests,
            )
            r_winner  = ev3.pick_winner(r_results)
        else:
            # No graph — create stub results
            stub = AlgorithmResult(
                algorithm_name="No Graph",
                output={},
                exec_time_ms=0.0,
                memory_kb=0.0,
                quality_score=0.0,
                quality_label="Total Route Distance (km)",
                lower_quality_is_better=True,
            )
            r_results, r_winner = [stub], stub

        report.routing_results = r_results
        report.routing_winner  = r_winner
        routes_map: Dict = r_winner.output or {}

        # ── Stage 4: Team Assignment ──────────────────────────────────────
        ev4 = self._evaluator([
            BacktrackingAssigner(),
            BranchBoundAssigner(),
        ])
        as_results = ev4.run_all(
            teams    = snap.teams,
            requests = ranked_requests,
            budget   = self.budget_units,
        )
        as_winner  = ev4.pick_winner(as_results)

        report.assignment_results = as_results
        report.assignment_winner  = as_winner

        final_assign: Dict[str, str] = as_winner.output or {}

        # ── Stage 5: Assemble Dispatch Plan ─────────────────────────────
        teams_by_id = {t["id"]: t for t in snap.teams}

        for rank_idx, req in enumerate(ranked_requests, start=1):
            req_id   = req["id"]
            team_id  = final_assign.get(req_id)
            team     = teams_by_id.get(team_id, {}) if team_id else {}

            # Resolve route from routing winner's output
            src = team.get("base_location", "HQ") if team else "HQ"
            dst = req.get("location", "")
            route_key = f"{src}->{dst}"
            route_info = routes_map.get(route_key, (0.0, []))
            dist_km = route_info[0] if isinstance(route_info, (tuple, list)) else 0.0
            path    = route_info[1] if isinstance(route_info, (tuple, list)) and len(route_info) > 1 else []

            # Assignment score
            a_score = 0.0
            if team_id and team:
                from team_assigner import SPEC_COMPAT
                team_obj = Team(**team)
                if _is_compatible(team_obj, req):
                    a_score = _score(team_obj, req)

            row = DispatchRow(
                rank             = rank_idx,
                request_id       = req_id,
                location         = req.get("location", ""),
                severity         = req.get("severity", ""),
                people           = req.get("people_affected", 0),
                priority_score   = req.get("score", 0.0),
                team_id          = team_id,
                team_name        = team.get("name") if team else None,
                route            = path,
                distance_km      = dist_km,
                resources        = allocated_resources,
                resource_value   = resource_value,
                assignment_score = a_score,
            )
            report.dispatch_rows.append(row)

            if not team_id:
                report.unassigned.append(req_id)
            else:
                report.total_assignment_score += a_score
                report.total_resource_value   += resource_value

        report.total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000
        return report

    # ── internal helpers ──────────────────────────────────────────────────

    def _quick_greedy_assign(
        self, teams: List[dict], ranked_requests: List[dict]
    ) -> Dict[str, str]:
        """
        Fast greedy assignment used to generate src→dst pairs for the
        routing race (before the real assignment race runs).
        """
        available = {t["id"] for t in teams if t.get("available", True)}
        assign: Dict[str, str] = {}

        for req in ranked_requests:
            best_score = -1.0
            best_team  = None
            for t_dict in teams:
                if t_dict["id"] not in available:
                    continue
                team_obj = Team(**t_dict)
                if _is_compatible(team_obj, req):
                    s = _score(team_obj, req)
                    if s > best_score:
                        best_score = s
                        best_team  = t_dict["id"]
            if best_team:
                assign[req["id"]] = best_team
                available.discard(best_team)

        return assign

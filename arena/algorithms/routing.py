"""
arena/algorithms/routing.py — Route Planning Strategies
-------------------------------------------------------
DijkstraRouter    : min-heap SSSP        — O((V+E) log V)
BellmanFordRouter : relaxation-based SSSP — O(V x E), handles negative weights
AStarRouter       : heuristic-guided SSSP — O(E log V), h=0 (no GPS coords)
                    when h=0, A* is mathematically identical to Dijkstra but
                    uses a different internal bookkeeping pattern (open-set
                    vs visited-set), which gives a slightly different memory
                    and branching profile — illustrating the algorithm's structure.

Quality metric: SUM of shortest path distances across all (team → location) pairs.
Lower total distance = better, so lower_quality_is_better=True.
"""
from __future__ import annotations

import heapq
import os
import sys
from typing import Dict, List, Optional, Tuple

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODULES = os.path.join(_ROOT, "python modules")
for p in (_ROOT, _MODULES):
    if p not in sys.path:
        sys.path.insert(0, p)

from route_planner import Graph, RoutePlanner

from arena.evaluator import AlgorithmResult, AlgorithmStrategy

INF = float("inf")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_graph(edges: List[Tuple]) -> Graph:
    g = Graph()
    for u, v, w in edges:
        g.add_edge(str(u), str(v), float(w))
    return g


def _route_pairs(assignments: Dict[str, str], teams_by_id: Dict, requests_by_id: Dict):
    """
    Yield (src, dst) pairs for each assigned team → request location.
    """
    for req_id, team_id in assignments.items():
        team = teams_by_id.get(team_id)
        req  = requests_by_id.get(req_id)
        if team and req:
            yield team.get("base_location", "HQ"), req.get("location", "")


# ── Strategy 1: Dijkstra ──────────────────────────────────────────────────────

class DijkstraRouter(AlgorithmStrategy):
    """
    Single-source shortest path via a binary min-heap.
    Wraps the existing RoutePlanner from route_planner.py.
    """
    name = "Dijkstra"

    def _execute(
        self,
        graph_edges: List[Tuple],
        assignments: Dict[str, str],
        teams: List[dict],
        requests: List[dict],
        **kwargs,
    ) -> AlgorithmResult:
        g       = _build_graph(graph_edges)
        planner = RoutePlanner(g)
        teams_by_id   = {t["id"]: t for t in teams}
        requests_by_id = {r["id"]: r for r in requests}

        routes: Dict[str, Tuple[float, List[str]]] = {}
        total_dist = 0.0

        for src, dst in _route_pairs(assignments, teams_by_id, requests_by_id):
            if src in g and dst in g:
                dist, path = planner.shortest_path(src, dst)
                if dist < INF:
                    routes[f"{src}->{dst}"] = (dist, path)
                    total_dist += dist

        return AlgorithmResult(
            algorithm_name=self.name,
            output=routes,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_dist if total_dist > 0 else 1.0,
            quality_label="Total Route Distance (km)",
            lower_quality_is_better=True,
            metadata={"routes_planned": len(routes), "total_km": round(total_dist, 2)},
        )


# ── Strategy 2: Bellman-Ford ──────────────────────────────────────────────────

class BellmanFordRouter(AlgorithmStrategy):
    """
    Bellman-Ford SSSP: V-1 edge relaxation passes.
    O(V * E) — slower than Dijkstra for non-negative graphs but
    handles negative edge weights (demonstrates the algorithm family).
    """
    name = "Bellman-Ford"

    def _bellman_ford(self, g: Graph, src: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        dist = {n: INF  for n in g.nodes}
        prev = {n: None for n in g.nodes}
        if src not in g.nodes:
            return dist, prev
        dist[src] = 0.0

        nodes = list(g.nodes)
        n = len(nodes)

        # Build flat edge list for iteration
        edges: List[Tuple[str, str, float]] = []
        for u in nodes:
            for v, w in g.neighbours(u):
                edges.append((u, v, w))

        for _ in range(n - 1):
            updated = False
            for u, v, w in edges:
                if dist[u] < INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    updated = True
            if not updated:
                break  # early termination

        return dist, prev

    def _reconstruct(self, prev: Dict[str, Optional[str]], src: str, dst: str) -> List[str]:
        path: List[str] = []
        node: Optional[str] = dst
        while node is not None:
            path.append(node)
            if node == src:
                break
            node = prev.get(node)
        path.reverse()
        return path if path and path[0] == src else []

    def _execute(
        self,
        graph_edges: List[Tuple],
        assignments: Dict[str, str],
        teams: List[dict],
        requests: List[dict],
        **kwargs,
    ) -> AlgorithmResult:
        g = _build_graph(graph_edges)
        teams_by_id    = {t["id"]: t for t in teams}
        requests_by_id = {r["id"]: r for r in requests}

        routes: Dict[str, Tuple[float, List[str]]] = {}
        total_dist = 0.0

        # Cache Bellman-Ford per unique source
        bf_cache: Dict[str, Tuple] = {}

        for src, dst in _route_pairs(assignments, teams_by_id, requests_by_id):
            if src not in g.nodes or dst not in g.nodes:
                continue
            if src not in bf_cache:
                bf_cache[src] = self._bellman_ford(g, src)
            dist_map, prev_map = bf_cache[src]
            d = dist_map.get(dst, INF)
            if d < INF:
                path = self._reconstruct(prev_map, src, dst)
                routes[f"{src}->{dst}"] = (d, path)
                total_dist += d

        return AlgorithmResult(
            algorithm_name=self.name,
            output=routes,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_dist if total_dist > 0 else 1.0,
            quality_label="Total Route Distance (km)",
            lower_quality_is_better=True,
            metadata={"routes_planned": len(routes), "total_km": round(total_dist, 2)},
        )


# ── Strategy 3: A* ────────────────────────────────────────────────────────────

class AStarRouter(AlgorithmStrategy):
    """
    A* SSSP with h=0 heuristic (consistent zero — no GPS coordinates available).
    When h=0 mathematically degrades to Dijkstra, but the OPEN-SET bookkeeping
    pattern (priority queue + came_from dict + g_score map) differs from the
    visited-set pattern in Dijkstra, giving a different memory allocation profile.
    This illustrates the algorithmic structure of A*.
    """
    name = "A* Search"

    def _astar(
        self, g: Graph, src: str, dst: str
    ) -> Tuple[float, List[str]]:
        if src not in g.nodes or dst not in g.nodes:
            return INF, []

        # g_score[n] = cheapest known cost from src to n
        g_score: Dict[str, float]         = {n: INF  for n in g.nodes}
        came_from: Dict[str, Optional[str]] = {n: None for n in g.nodes}
        g_score[src] = 0.0

        # f_score = g_score + h; h=0 so f=g
        open_set: List[Tuple[float, str]] = [(0.0, src)]

        while open_set:
            f, current = heapq.heappop(open_set)

            if current == dst:
                # Reconstruct path
                path: List[str] = []
                node: Optional[str] = dst
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return g_score[dst], path

            if f > g_score[current]:
                continue  # stale entry

            for neighbour, weight in g.neighbours(current):
                tentative_g = g_score[current] + weight
                if tentative_g < g_score.get(neighbour, INF):
                    g_score[neighbour]   = tentative_g
                    came_from[neighbour] = current
                    h = 0.0  # zero heuristic — A* degrades to Dijkstra
                    heapq.heappush(open_set, (tentative_g + h, neighbour))

        return INF, []

    def _execute(
        self,
        graph_edges: List[Tuple],
        assignments: Dict[str, str],
        teams: List[dict],
        requests: List[dict],
        **kwargs,
    ) -> AlgorithmResult:
        g = _build_graph(graph_edges)
        teams_by_id    = {t["id"]: t for t in teams}
        requests_by_id = {r["id"]: r for r in requests}

        routes: Dict[str, Tuple[float, List[str]]] = {}
        total_dist = 0.0

        for src, dst in _route_pairs(assignments, teams_by_id, requests_by_id):
            if src not in g.nodes or dst not in g.nodes:
                continue
            d, path = self._astar(g, src, dst)
            if d < INF:
                routes[f"{src}->{dst}"] = (d, path)
                total_dist += d

        return AlgorithmResult(
            algorithm_name=self.name,
            output=routes,
            exec_time_ms=0.0,
            memory_kb=0.0,
            quality_score=total_dist if total_dist > 0 else 1.0,
            quality_label="Total Route Distance (km)",
            lower_quality_is_better=True,
            metadata={"routes_planned": len(routes), "total_km": round(total_dist, 2)},
        )

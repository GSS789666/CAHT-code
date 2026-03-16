"""
CAHT V2 — OR-Tools routing solver baseline.
"""
import numpy as np
import time
from config import W_ENERGY, W_MAKESPAN, W_TW, ORTOOLS_TIME
from alns_solver import compute_objective, nearest_greedy

def _dist(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


def solve_ortools(instance, time_limit=None):
    """
    Solve using OR-Tools VRP solver.
    Returns (assignment, sequence, obj, stats).
    Falls back to nearest_greedy if OR-Tools fails.
    """
    if time_limit is None:
        time_limit = ORTOOLS_TIME

    robots = instance["robots"]
    tasks  = instance["tasks"]
    N, M   = len(robots), len(tasks)

    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return _fallback_solve(instance, time_limit)

    num_nodes = N + M

    def node_pos(idx):
        if idx < N:
            return robots[idx]["pos"]
        else:
            return tasks[idx - N]["pick"]

    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            pi = node_pos(i)
            pj = node_pos(j)
            d = _dist(pi, pj)
            if j >= N:
                t = tasks[j - N]
                d += _dist(t["pick"], t["drop"])
            dist_matrix[i][j] = max(1, int(d * 10))

    starts = list(range(N))
    ends = list(range(N))

    manager = pywrapcp.RoutingIndexManager(num_nodes, N, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        f = manager.IndexToNode(from_idx)
        t_node = manager.IndexToNode(to_idx)
        return int(dist_matrix[f][t_node])

    transit_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Capacity constraint
    def demand_cb(idx):
        node = manager.IndexToNode(idx)
        if node < N:
            return 0
        return max(1, int(tasks[node - N]["weight"] * 10))

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    caps = [int(robots[i]["cap"] * 10) for i in range(N)]
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, caps, True, "Capacity")

    # Allow dropping nodes if infeasible
    penalty = 100000
    for j in range(M):
        node_index = manager.NodeToIndex(N + j)
        routing.AddDisjunction([node_index], penalty)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit

    t0 = time.time()
    solution = routing.SolveWithParameters(search_params)
    elapsed = time.time() - t0

    assignment_list = [-1] * M
    sequence_dict = {i: [] for i in range(N)}

    if solution:
        for i in range(N):
            idx = routing.Start(i)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node >= N:
                    j = node - N
                    assignment_list[j] = i
                    sequence_dict[i].append(j)
                idx = solution.Value(routing.NextVar(idx))

    assigned_count = sum(1 for a in assignment_list if a >= 0)
    if assigned_count == 0:
        return _fallback_solve(instance, elapsed)

    obj, energy, makespan, tw_pen, cvr, tw_sat = compute_objective(
        assignment_list, sequence_dict, robots, tasks)

    stats = {
        "obj": obj, "energy": energy, "makespan": makespan,
        "tw_penalty": tw_pen, "cvr_count": cvr,
        "cvr_pct": cvr / M * 100 if M > 0 else 0,
        "tw_sat": tw_sat / M * 100 if M > 0 else 100,
        "time_ms": elapsed * 1000,
    }
    return assignment_list, sequence_dict, obj, stats


def _fallback_solve(instance, elapsed_or_limit):
    """Fallback to nearest greedy if OR-Tools fails."""
    robots = instance["robots"]
    tasks = instance["tasks"]
    M = len(tasks)

    t0 = time.time()
    assignment, sequence = nearest_greedy(robots, tasks)
    solve_time = time.time() - t0

    obj, energy, makespan, tw_pen, cvr, tw_sat = compute_objective(
        assignment, sequence, robots, tasks)

    elapsed = elapsed_or_limit if isinstance(elapsed_or_limit, float) else solve_time
    stats = {
        "obj": obj, "energy": energy, "makespan": makespan,
        "tw_penalty": tw_pen, "cvr_count": cvr,
        "cvr_pct": cvr / M * 100 if M > 0 else 0,
        "tw_sat": tw_sat / M * 100 if M > 0 else 100,
        "time_ms": elapsed * 1000 if elapsed < 100 else elapsed,
    }
    return assignment, sequence, obj, stats

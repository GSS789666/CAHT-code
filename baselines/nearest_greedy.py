"""
CAHT V2 — Nearest-priority greedy baseline.
"""
from alns_solver import nearest_greedy, compute_objective
import time


def solve_nearest_greedy(instance):
    """Wrapper returning (assignment, sequence, obj, stats)."""
    robots = instance["robots"]
    tasks  = instance["tasks"]
    M = len(tasks)

    t0 = time.perf_counter()
    assignment, sequence = nearest_greedy(robots, tasks)
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    obj, energy, makespan, tw_pen, cvr, tw_sat = compute_objective(
        assignment, sequence, robots, tasks)

    stats = {
        "obj": obj, "energy": energy, "makespan": makespan,
        "tw_penalty": tw_pen, "cvr_count": cvr,
        "cvr_pct": cvr / M * 100 if M > 0 else 0,
        "tw_sat": tw_sat / M * 100 if M > 0 else 100,
        "time_ms": elapsed,
    }
    return assignment, sequence, obj, stats

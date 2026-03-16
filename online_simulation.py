"""
CAHT V2 — Online rolling-horizon simulation.
Poisson arrivals, periodic re-allocation.
"""
import numpy as np
import time, os
from config import (
    SIM_DURATION, SIM_LAMBDA, SIM_REALLOC, SEED,
    SCALES, W_ENERGY, W_MAKESPAN, W_TW,
)
from data_generator import generate_instance
from alns_solver import nearest_greedy, solve_alns, compute_objective


def _dist(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


def run_online_simulation(method_name, solve_fn, N=10, seed=SEED):
    """
    Simulate online task arrivals and periodic re-allocation.
    solve_fn(instance) → (assignment, sequence, obj, stats)
    Returns metrics dict.
    """
    rng = np.random.RandomState(seed)

    # generate base robots
    base_inst = generate_instance(N, 1, rng)
    robots = base_inst["robots"]

    # Poisson task arrivals
    tasks_queue = []
    t = 0.0
    task_id = 0
    while t < SIM_DURATION:
        dt = rng.exponential(1.0 / SIM_LAMBDA)
        t += dt
        if t >= SIM_DURATION:
            break
        new_inst = generate_instance(N, 1, rng)
        task = new_inst["tasks"][0]
        task["id"] = task_id
        task["arrival_time"] = t
        tasks_queue.append(task)
        task_id += 1

    # simulate
    completed = 0
    tw_satisfied = 0
    total_wait = 0.0
    solve_times = []

    pending_tasks = []
    robot_free_time = [0.0] * N
    robot_pos = [r["pos"].copy() for r in robots]

    for realloc_time in np.arange(0, SIM_DURATION, SIM_REALLOC):
        # collect newly arrived tasks
        new_tasks = [t for t in tasks_queue
                     if realloc_time - SIM_REALLOC < t.get("arrival_time", 0) <= realloc_time]
        pending_tasks.extend(new_tasks)

        if not pending_tasks:
            continue

        # build instance for solver
        M = min(len(pending_tasks), 50)
        # update robot positions based on free time
        current_robots = []
        for i, r in enumerate(robots):
            r_copy = dict(r)
            r_copy["pos"] = robot_pos[i].copy()
            current_robots.append(r_copy)

        inst = {
            "robots": current_robots,
            "tasks": pending_tasks[:M],
            "N": N,
            "M": M,
        }

        t0 = time.perf_counter()
        try:
            a, s, obj, stats = solve_fn(inst)
            solve_times.append((time.perf_counter() - t0) * 1000)
        except Exception:
            solve_times.append(0)
            continue

        # process assignments
        assigned_ids = set()
        for j in range(M):
            rid = a[j] if isinstance(a, list) and j < len(a) else -1
            if rid >= 0 and rid < N:
                task = pending_tasks[j]
                travel_d = _dist(current_robots[rid]["pos"], task["pick"]) + _dist(task["pick"], task["drop"])
                travel_time = travel_d / current_robots[rid]["vel"]
                finish_time = max(robot_free_time[rid], realloc_time) + travel_time
                robot_free_time[rid] = finish_time
                robot_pos[rid] = task["drop"].copy()

                completed += 1
                wait = realloc_time - task.get("arrival_time", 0)
                total_wait += max(0, wait)

                if finish_time <= task["tw_late"]:
                    tw_satisfied += 1

                assigned_ids.add(j)

        # remove assigned
        pending_tasks = [t for idx, t in enumerate(pending_tasks) if idx not in assigned_ids]

    total_tasks = len(tasks_queue)
    throughput = completed / (SIM_DURATION / 60) if SIM_DURATION > 0 else 0

    return {
        "method": method_name,
        "completed": completed,
        "total": total_tasks,
        "tw_sat": tw_satisfied / max(completed, 1) * 100,
        "avg_wait": total_wait / max(completed, 1),
        "throughput": throughput,
        "solve_time_ms": np.mean(solve_times) if solve_times else 0,
    }


def run_all_online(caht_model=None, pomo_model=None):
    """Run online simulation for all methods."""
    import torch
    from baselines.nearest_greedy import solve_nearest_greedy
    from baselines.pomo import pomo_assignment_to_sequence

    results = []

    # Nearest Greedy
    def ng_fn(inst):
        return solve_nearest_greedy(inst)
    results.append(run_online_simulation("Nearest Greedy", ng_fn))

    # ALNS (1s time limit for online)
    def alns_fn(inst):
        a, s, obj, st = solve_alns(inst, time_limit=1)
        return a, s, obj, st
    results.append(run_online_simulation("ALNS(1s)", alns_fn))

    # POMO
    if pomo_model is not None:
        pomo_model.eval()
        def pomo_fn(inst):
            with torch.no_grad():
                assignment, _ = pomo_model([inst], greedy=True)
                seqs = pomo_assignment_to_sequence(assignment, [inst])
                a = assignment[0].cpu().tolist()
                s = seqs[0]
                obj, *_ = compute_objective(a, s, inst["robots"], inst["tasks"])
                return a, s, obj, {"obj": obj, "time": 0}
        results.append(run_online_simulation("POMO", pomo_fn))

    # CAHT
    if caht_model is not None:
        caht_model.eval()
        def caht_fn(inst):
            with torch.no_grad():
                assignment, sequences = caht_model([inst], greedy=True)
                a = assignment[0].cpu().tolist()
                s = sequences[0]
                obj, *_ = compute_objective(a, s, inst["robots"], inst["tasks"])
                return a, s, obj, {"obj": obj, "time": 0}
        results.append(run_online_simulation("CAHT", caht_fn))

    return results

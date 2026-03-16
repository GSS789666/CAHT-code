"""
CAHT V2 — Adaptive Large Neighbourhood Search (ALNS) solver.
Used both as label generator and as evaluation baseline.
"""
import numpy as np
import time, copy
from config import (
    ALNS_ITERS, ALNS_TIME, ALNS_T_INIT, ALNS_COOLING,
    ALNS_Q_LO, ALNS_Q_HI, W_ENERGY, W_MAKESPAN, W_TW, SEED,
)

# ─────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────

def _dist(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


def compute_objective(assignment, sequence, robots, tasks):
    """
    Compute J = w_e*TotalEnergy + w_m*Makespan + w_tw*TW_Penalty.
    assignment: list of length M, assignment[j] = robot_id for task j
    sequence: dict {robot_id: [task_ids in order]}
    Returns (obj, energy, makespan, tw_penalty, cvr_count, tw_sat_count)
    """
    N = len(robots)
    M = len(tasks)
    total_energy = 0.0
    completion_times = []
    tw_penalty = 0.0

    energy_used = [0.0] * N
    task_arrival = {}  # j → arrival time at task

    for i in range(N):
        r = robots[i]
        task_order = sequence.get(i, [])
        if not task_order:
            continue
        pos = r["pos"].copy()
        cur_time = 0.0
        for j in task_order:
            t = tasks[j]
            # travel to pick
            d1 = _dist(pos, t["pick"])
            travel1 = d1 / r["vel"]
            cur_time += travel1
            total_energy += d1 * r["energy_rate"]
            energy_used[i] += d1 * r["energy_rate"]

            # travel pick → drop
            d2 = _dist(t["pick"], t["drop"])
            travel2 = d2 / r["vel"]
            cur_time += travel2
            total_energy += d2 * r["energy_rate"]
            energy_used[i] += d2 * r["energy_rate"]

            # time window check
            task_arrival[j] = cur_time
            if cur_time > t["tw_late"]:
                tw_penalty += cur_time - t["tw_late"]

            pos = t["drop"].copy()

        completion_times.append(cur_time)

    # constraint violation count (includes capacity, energy, AND time window)
    cvr_count = 0
    tw_sat_count = 0
    for j in range(M):
        rid = assignment[j]
        if rid < 0:
            cvr_count += 1
            continue
        t = tasks[j]
        r = robots[rid]
        violated = False
        # per-task capacity check
        if t["weight"] > r["cap"]:
            violated = True
        # energy check: did robot run out of battery?
        if energy_used[rid] > r["battery"]:
            violated = True
        # time window violation
        arrival = task_arrival.get(j, float("inf"))
        if arrival > t["tw_late"]:
            violated = True
        else:
            tw_sat_count += 1
        if violated:
            cvr_count += 1

    makespan = max(completion_times) if completion_times else 0.0
    obj = W_ENERGY * total_energy + W_MAKESPAN * makespan + W_TW * tw_penalty
    return obj, total_energy, makespan, tw_penalty, cvr_count, tw_sat_count


# ─────────────────────────────────────────────────────
# Nearest-Greedy initial solution (also used standalone)
# ─────────────────────────────────────────────────────

def nearest_greedy(robots, tasks):
    """Greedy: assign tasks by priority desc to nearest feasible robot."""
    N, M = len(robots), len(tasks)
    assignment = [-1] * M
    sequence = {i: [] for i in range(N)}
    bat_rem = [r["battery"] for r in robots]
    pos = [r["pos"].copy() for r in robots]

    order = sorted(range(M), key=lambda j: -tasks[j]["priority"])
    for j in order:
        t = tasks[j]
        best_r, best_d = -1, float("inf")
        for i in range(N):
            # per-task capacity: robot must be able to carry this single item
            if robots[i]["cap"] < t["weight"]:
                continue
            d = _dist(pos[i], t["pick"]) + _dist(t["pick"], t["drop"])
            energy = d * robots[i]["energy_rate"]
            if energy > bat_rem[i]:
                continue
            d_pick = _dist(pos[i], t["pick"])
            if d_pick < best_d:
                best_d = d_pick
                best_r = i
        if best_r >= 0:
            assignment[j] = best_r
            sequence[best_r].append(j)
            d = _dist(pos[best_r], t["pick"]) + _dist(t["pick"], t["drop"])
            bat_rem[best_r] -= d * robots[best_r]["energy_rate"]
            pos[best_r] = t["drop"].copy()

    return assignment, sequence


# ─────────────────────────────────────────────────────
# Destroy operators
# ─────────────────────────────────────────────────────

def _random_removal(assignment, sequence, M, rng, q):
    assigned = [j for j in range(M) if assignment[j] >= 0]
    if not assigned:
        return [], assignment, sequence
    n_remove = max(1, int(q * len(assigned)))
    removed = list(rng.choice(assigned, min(n_remove, len(assigned)), replace=False))
    a = assignment[:]
    s = {k: [x for x in v if x not in removed] for k, v in sequence.items()}
    for j in removed:
        a[j] = -1
    return removed, a, s


def _worst_removal(assignment, sequence, robots, tasks, M, rng, q):
    """Remove tasks that contribute most to objective."""
    assigned = [j for j in range(M) if assignment[j] >= 0]
    if not assigned:
        return [], assignment, sequence
    costs = []
    for j in assigned:
        rid = assignment[j]
        t = tasks[j]
        r = robots[rid]
        d = _dist(r["pos"], t["pick"]) + _dist(t["pick"], t["drop"])
        costs.append((d * r["energy_rate"], j))
    costs.sort(reverse=True)
    n_remove = max(1, int(q * len(assigned)))
    removed = [c[1] for c in costs[:n_remove]]
    a = assignment[:]
    s = {k: [x for x in v if x not in removed] for k, v in sequence.items()}
    for j in removed:
        a[j] = -1
    return removed, a, s


def _shaw_removal(assignment, sequence, robots, tasks, M, rng, q):
    """Remove similar tasks (close position + same robot)."""
    assigned = [j for j in range(M) if assignment[j] >= 0]
    if not assigned:
        return [], assignment, sequence
    seed_j = rng.choice(assigned)
    dists = []
    for j in assigned:
        if j == seed_j:
            dists.append((0.0, j))
            continue
        pos_sim = _dist(tasks[seed_j]["pick"], tasks[j]["pick"])
        robot_sim = 0.0 if assignment[j] == assignment[seed_j] else 50.0
        dists.append((pos_sim + robot_sim, j))
    dists.sort()
    n_remove = max(1, int(q * len(assigned)))
    removed = [d[1] for d in dists[:n_remove]]
    a = assignment[:]
    s = {k: [x for x in v if x not in removed] for k, v in sequence.items()}
    for j in removed:
        a[j] = -1
    return removed, a, s


# ─────────────────────────────────────────────────────
# Repair operators
# ─────────────────────────────────────────────────────

def _insertion_cost(rid, j, pos_after, robots, tasks, bat_rem):
    """Cost of inserting task j to robot rid at end."""
    r = robots[rid]
    t = tasks[j]
    # per-task capacity check
    if r["cap"] < t["weight"]:
        return float("inf")
    d = _dist(pos_after[rid], t["pick"]) + _dist(t["pick"], t["drop"])
    energy = d * r["energy_rate"]
    if energy > bat_rem[rid]:
        return float("inf")
    return d * r["energy_rate"]


def _greedy_insertion(removed, assignment, sequence, robots, tasks, N):
    a = assignment[:]
    s = {k: list(v) for k, v in sequence.items()}
    bat_rem = [robots[i]["battery"] for i in range(N)]
    pos_after = {}
    for i in range(N):
        if s[i]:
            pos_after[i] = tasks[s[i][-1]]["drop"].copy()
        else:
            pos_after[i] = robots[i]["pos"].copy()
    # estimate battery used
    for i in range(N):
        p = robots[i]["pos"].copy()
        for j in s[i]:
            d = _dist(p, tasks[j]["pick"]) + _dist(tasks[j]["pick"], tasks[j]["drop"])
            bat_rem[i] -= d * robots[i]["energy_rate"]
            p = tasks[j]["drop"].copy()

    for j in removed:
        best_r, best_c = -1, float("inf")
        for i in range(N):
            c = _insertion_cost(i, j, pos_after, robots, tasks, bat_rem)
            if c < best_c:
                best_c = c
                best_r = i
        if best_r >= 0:
            a[j] = best_r
            s[best_r].append(j)
            d = _dist(pos_after[best_r], tasks[j]["pick"]) + _dist(tasks[j]["pick"], tasks[j]["drop"])
            bat_rem[best_r] -= d * robots[best_r]["energy_rate"]
            pos_after[best_r] = tasks[j]["drop"].copy()
    return a, s


def _regret2_insertion(removed, assignment, sequence, robots, tasks, N):
    a = assignment[:]
    s = {k: list(v) for k, v in sequence.items()}
    bat_rem = [robots[i]["battery"] for i in range(N)]
    pos_after = {}
    for i in range(N):
        if s[i]:
            pos_after[i] = tasks[s[i][-1]]["drop"].copy()
        else:
            pos_after[i] = robots[i]["pos"].copy()
    for i in range(N):
        p = robots[i]["pos"].copy()
        for j in s[i]:
            d = _dist(p, tasks[j]["pick"]) + _dist(tasks[j]["pick"], tasks[j]["drop"])
            bat_rem[i] -= d * robots[i]["energy_rate"]
            p = tasks[j]["drop"].copy()

    remaining = list(removed)
    while remaining:
        best_j, best_r, best_regret = -1, -1, -float("inf")
        for j in remaining:
            costs = []
            for i in range(N):
                c = _insertion_cost(i, j, pos_after, robots, tasks, bat_rem)
                costs.append((c, i))
            costs.sort()
            if costs[0][0] == float("inf"):
                best_j = j
                best_r = -1
                break
            regret = (costs[1][0] if len(costs) > 1 else costs[0][0]) - costs[0][0]
            if regret > best_regret:
                best_regret = regret
                best_j = j
                best_r = costs[0][1]
        remaining.remove(best_j)
        if best_r >= 0:
            a[best_j] = best_r
            s[best_r].append(best_j)
            d = _dist(pos_after[best_r], tasks[best_j]["pick"]) + _dist(tasks[best_j]["pick"], tasks[best_j]["drop"])
            bat_rem[best_r] -= d * robots[best_r]["energy_rate"]
            pos_after[best_r] = tasks[best_j]["drop"].copy()
    return a, s


# ─────────────────────────────────────────────────────
# ALNS main
# ─────────────────────────────────────────────────────

def solve_alns(instance, time_limit=None, max_iters=None, seed=42):
    """
    Run ALNS on a single instance.
    Returns (assignment, sequence, obj, stats).
    """
    if time_limit is None:
        time_limit = ALNS_TIME
    if max_iters is None:
        max_iters = ALNS_ITERS

    rng = np.random.RandomState(seed)
    robots = instance["robots"]
    tasks = instance["tasks"]
    N, M = len(robots), len(tasks)

    # initial solution
    cur_a, cur_s = nearest_greedy(robots, tasks)
    cur_obj, *_ = compute_objective(cur_a, cur_s, robots, tasks)

    best_a, best_s, best_obj = cur_a[:], {k: list(v) for k, v in cur_s.items()}, cur_obj

    # operator weights
    n_destroy = 3
    n_repair = 2
    d_weights = np.ones(n_destroy)
    r_weights = np.ones(n_repair)
    d_counts = np.zeros(n_destroy)
    r_counts = np.zeros(n_destroy)
    d_scores = np.zeros(n_destroy)
    r_scores = np.zeros(n_repair)

    T = ALNS_T_INIT
    start = time.time()

    for it in range(max_iters):
        if time.time() - start > time_limit:
            break

        q = rng.uniform(ALNS_Q_LO, ALNS_Q_HI)

        # select destroy operator (roulette)
        d_probs = d_weights / d_weights.sum()
        d_op = rng.choice(n_destroy, p=d_probs)

        if d_op == 0:
            removed, new_a, new_s = _random_removal(cur_a, cur_s, M, rng, q)
        elif d_op == 1:
            removed, new_a, new_s = _shaw_removal(cur_a, cur_s, robots, tasks, M, rng, q)
        else:
            removed, new_a, new_s = _worst_removal(cur_a, cur_s, robots, tasks, M, rng, q)

        if not removed:
            continue

        # select repair operator
        r_probs = r_weights / r_weights.sum()
        r_op = rng.choice(n_repair, p=r_probs)

        if r_op == 0:
            new_a, new_s = _greedy_insertion(removed, new_a, new_s, robots, tasks, N)
        else:
            new_a, new_s = _regret2_insertion(removed, new_a, new_s, robots, tasks, N)

        new_obj, *_ = compute_objective(new_a, new_s, robots, tasks)

        # acceptance (simulated annealing)
        delta = new_obj - cur_obj
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            cur_a, cur_s, cur_obj = new_a, new_s, new_obj
            score = 2 if delta < 0 else 1

            if new_obj < best_obj:
                best_a = cur_a[:]
                best_s = {k: list(v) for k, v in cur_s.items()}
                best_obj = new_obj
                score = 3
        else:
            score = 0

        d_scores[d_op] += score
        r_scores[r_op] += score
        d_counts[d_op] += 1
        r_counts[r_op] += 1

        # update weights every 100 iters
        if (it + 1) % 100 == 0:
            for k in range(n_destroy):
                if d_counts[k] > 0:
                    d_weights[k] = max(0.1, 0.7 * d_weights[k] + 0.3 * d_scores[k] / d_counts[k])
            for k in range(n_repair):
                if r_counts[k] > 0:
                    r_weights[k] = max(0.1, 0.7 * r_weights[k] + 0.3 * r_scores[k] / r_counts[k])
            d_scores[:] = 0
            r_scores[:] = 0
            d_counts[:] = 0
            r_counts[:] = 0

        T *= ALNS_COOLING

    obj, energy, makespan, tw_pen, cvr, tw_sat = compute_objective(best_a, best_s, robots, tasks)
    elapsed = time.time() - start
    stats = {
        "obj": obj, "energy": energy, "makespan": makespan,
        "tw_penalty": tw_pen, "cvr_count": cvr, "cvr_pct": cvr / M * 100,
        "tw_sat": tw_sat / M * 100 if M > 0 else 100,
        "time": elapsed, "iters": min(it + 1, max_iters),
    }
    return best_a, best_s, obj, stats


def solve_batch(instances, time_limit=None, seed=42, verbose=True):
    """Solve a list of instances and return list of (assignment, sequence, obj, stats)."""
    results = []
    for idx, inst in enumerate(instances):
        a, s, obj, st = solve_alns(inst, time_limit=time_limit, seed=seed + idx)
        results.append((a, s, obj, st))
        if verbose and (idx + 1) % 50 == 0:
            avg_obj = np.mean([r[2] for r in results])
            avg_cvr = np.mean([r[3]["cvr_pct"] for r in results])
            print(f"  ALNS solved {idx+1}/{len(instances)}, avg_obj={avg_obj:.2f}, avg_cvr={avg_cvr:.1f}%")
    return results

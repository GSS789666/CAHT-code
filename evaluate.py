"""
CAHT V2 — Unified evaluation framework.
Evaluates all methods on test sets and computes metrics.
"""
import torch
import numpy as np
import time, os, pickle
from config import (
    DEVICE, DATA_DIR, CKPT_DIR, OUTPUT_DIR, SCALES,
    W_ENERGY, W_MAKESPAN, W_TW, SEED, set_seed,
)
from alns_solver import compute_objective, solve_alns
from ortools_solver import solve_ortools
from baselines.nearest_greedy import solve_nearest_greedy
from baselines.pomo import POMO, pomo_assignment_to_sequence
from model import CAHT


def eval_caht(model, instances, greedy=True):
    """Evaluate CAHT on a list of instances. Returns list of stats dicts."""
    model.eval()
    results = []
    with torch.no_grad():
        for inst in instances:
            t0 = time.perf_counter()
            assignment, sequences = model([inst], greedy=greedy)
            elapsed = (time.perf_counter() - t0) * 1000

            a = assignment[0].cpu().tolist()
            s = sequences[0]
            M = len(inst["tasks"])
            obj, energy, makespan, tw_pen, cvr, tw_sat = compute_objective(
                a, s, inst["robots"], inst["tasks"])
            results.append({
                "obj": obj, "energy": energy, "makespan": makespan,
                "tw_penalty": tw_pen, "cvr_count": cvr,
                "cvr_pct": cvr / M * 100 if M > 0 else 0,
                "tw_sat": tw_sat / M * 100 if M > 0 else 100,
                "time_ms": elapsed,
            })
    return results


def eval_pomo(model, instances, greedy=True):
    """Evaluate POMO baseline."""
    model.eval()
    results = []
    with torch.no_grad():
        for inst in instances:
            t0 = time.perf_counter()
            assignment, _ = model([inst], greedy=greedy)
            seqs = pomo_assignment_to_sequence(assignment, [inst])
            elapsed = (time.perf_counter() - t0) * 1000

            a = assignment[0].cpu().tolist()
            s = seqs[0]
            M = len(inst["tasks"])
            obj, energy, makespan, tw_pen, cvr, tw_sat = compute_objective(
                a, s, inst["robots"], inst["tasks"])
            results.append({
                "obj": obj, "energy": energy, "makespan": makespan,
                "tw_penalty": tw_pen, "cvr_count": cvr,
                "cvr_pct": cvr / M * 100 if M > 0 else 0,
                "tw_sat": tw_sat / M * 100 if M > 0 else 100,
                "time_ms": elapsed,
            })
    return results


def aggregate_results(results):
    """Compute mean metrics from list of result dicts."""
    keys = ["obj", "makespan", "cvr_pct", "tw_sat", "time_ms"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = np.mean(vals)
    return agg


def eval_all_methods(scale, caht_sl=None, caht_rl=None, pomo_model=None):
    """
    Evaluate all methods on a given scale's test set.
    Returns dict of {method_name: aggregated_metrics}.
    """
    with open(os.path.join(DATA_DIR, f"{scale}.pkl"), "rb") as f:
        data = pickle.load(f)
    test_insts = data["test"]

    alns_path = os.path.join(DATA_DIR, f"{scale}_labels.pkl")
    has_alns_labels = os.path.exists(alns_path)

    results = {}
    print(f"\n  Evaluating {scale} scale ({len(test_insts)} instances)...")

    # 1. Nearest Greedy
    print("    Nearest Greedy...")
    ng_results = [solve_nearest_greedy(inst) for inst in test_insts]
    ng_stats = [r[3] for r in ng_results]
    results["Nearest Greedy"] = aggregate_results(ng_stats)

    # 2. ALNS
    if has_alns_labels:
        print("    ALNS (from labels)...")
        with open(alns_path, "rb") as f:
            labels = pickle.load(f)
        alns_stats = []
        for idx in range(min(len(test_insts), len(labels["test"]))):
            a, s, obj_val, st = labels["test"][idx]
            if "time_ms" not in st:
                st["time_ms"] = st.get("time", 0) * 1000
            alns_stats.append(st)
        results["ALNS(30s)"] = aggregate_results(alns_stats)
    else:
        print("    ALNS (solving)...")
        alns_results = []
        for inst in test_insts:
            a, s, obj_val, st = solve_alns(inst, time_limit=10)
            if "time_ms" not in st:
                st["time_ms"] = st.get("time", 0) * 1000
            alns_results.append(st)
        results["ALNS(30s)"] = aggregate_results(alns_results)

    # 3. OR-Tools
    print("    OR-Tools...")
    ort_stats = []
    for inst in test_insts:
        _, _, _, st = solve_ortools(inst, time_limit=10)
        if "time_ms" not in st:
            st["time_ms"] = st.get("time", 0) * 1000
        ort_stats.append(st)
    results["OR-Tools"] = aggregate_results(ort_stats)

    # 4. POMO
    if pomo_model is not None:
        print("    POMO...")
        pomo_results = eval_pomo(pomo_model, test_insts)
        results["POMO"] = aggregate_results(pomo_results)

    # 5. CAHT (SL only)
    if caht_sl is not None:
        print("    CAHT (SL)...")
        caht_sl_results = eval_caht(caht_sl, test_insts)
        results["CAHT(SL)"] = aggregate_results(caht_sl_results)

    # 6. CAHT (SL+RL)
    if caht_rl is not None:
        print("    CAHT (SL+RL)...")
        caht_rl_results = eval_caht(caht_rl, test_insts)
        results["CAHT(SL+RL)"] = aggregate_results(caht_rl_results)

    # compute Gap vs ALNS
    alns_obj = results["ALNS(30s)"]["obj"]
    for method, metrics in results.items():
        if alns_obj > 0:
            metrics["gap_pct"] = (metrics["obj"] - alns_obj) / alns_obj * 100
        else:
            metrics["gap_pct"] = 0.0

    return results


def latency_breakdown(model, instances, N):
    """Time each module separately."""
    model.eval()
    device = next(model.parameters()).device

    times = {"embedding": [], "encoder": [], "assign_dec": [], "seq_dec": []}

    with torch.no_grad():
        for inst in instances[:20]:
            robot_feat, task_feat, robot_types, dist_matrix, robots_data, tasks_data = \
                model._build_features([inst], device)

            t0 = time.perf_counter()
            H0 = model.embedding(robot_feat, task_feat, robot_types)
            times["embedding"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            H = model.encoder(H0, dist_matrix)
            times["encoder"].append((time.perf_counter() - t0) * 1000)

            h_r = H[:, :N, :]
            h_t = H[:, N:, :]

            t0 = time.perf_counter()
            assignment, _ = model.assign_dec(h_r, h_t, robots_data, tasks_data,
                                             greedy=True, return_log_probs=True)
            times["assign_dec"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            sequences = model.seq_dec(H, assignment, N, tasks_data=tasks_data, greedy=True)
            times["seq_dec"].append((time.perf_counter() - t0) * 1000)

    return {k: np.mean(v) for k, v in times.items()}

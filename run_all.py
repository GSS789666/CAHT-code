"""
CAHT V2 — Master experiment runner.
Executes all 11 steps sequentially and generates output files.
"""
import os, sys, pickle, time, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ensure project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    SEED, set_seed, DEVICE, DATA_DIR, OUTPUT_DIR, CKPT_DIR,
    SCALES, TRAIN_SIZES, VAL_SIZE, TEST_SIZE, ALNS_TIME, SCALING_SIZES,
    REDUCED_RUN, REDUCED_NOTE, SL_EPOCHS, RL_EPOCHS,
)

set_seed(SEED)


def write_csv(filename, headers, rows, note=None):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if note:
            w.writerow([f"# {note}"])
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print(f"  Saved {filename}")


# ══════════════════════════════════════════════════════
# STEP 1: Generate datasets
# ══════════════════════════════════════════════════════
def step1():
    print("\n" + "="*60)
    print("STEP 1: Generate datasets")
    print("="*60)
    from data_generator import generate_all_datasets
    stats = generate_all_datasets()
    return stats


# ══════════════════════════════════════════════════════
# STEP 2: Generate ALNS labels
# ══════════════════════════════════════════════════════
def step2():
    print("\n" + "="*60)
    print("STEP 2: Generate ALNS labels")
    print("="*60)
    from alns_solver import solve_alns

    label_stats = {}
    for scale in ["S", "M", "L", "XL"]:
        print(f"\n  Processing {scale}...")
        with open(os.path.join(DATA_DIR, f"{scale}.pkl"), "rb") as f:
            data = pickle.load(f)

        labels = {}
        # Only solve ALNS for non-augmented splits to save time
        # For "train": solve only "train_orig", then replicate labels 4x
        for split in data.keys():
            if split == "train":
                continue  # handle separately below
            if split == "train_orig":
                continue  # handle separately below
            insts = data[split]
            print(f"    {split}: {len(insts)} instances")
            split_labels = []
            for idx, inst in enumerate(insts):
                a, s, obj, st = solve_alns(inst, time_limit=ALNS_TIME, seed=SEED + idx)
                split_labels.append((a, s, obj, st))
                if (idx + 1) % 50 == 0:
                    avg_cvr = np.mean([l[3]["cvr_pct"] for l in split_labels])
                    print(f"      {idx+1}/{len(insts)}, avg_cvr={avg_cvr:.1f}%")
            labels[split] = split_labels

            avg_obj = np.mean([l[2] for l in split_labels])
            avg_cvr = np.mean([l[3]["cvr_pct"] for l in split_labels])
            avg_tw  = np.mean([l[3]["tw_sat"] for l in split_labels])
            print(f"    {split} done: avg_obj={avg_obj:.2f}, avg_cvr={avg_cvr:.1f}%, avg_tw_sat={avg_tw:.1f}%")

            if split == "test":
                label_stats[scale] = {
                    "avg_obj": avg_obj, "avg_cvr": avg_cvr, "avg_tw_sat": avg_tw,
                }

        # Handle training data: solve original, replicate for augmented
        if "train_orig" in data:
            orig_insts = data["train_orig"]
            print(f"    train_orig: {len(orig_insts)} instances (ALNS solving)")
            orig_labels = []
            for idx, inst in enumerate(orig_insts):
                a, s, obj, st = solve_alns(inst, time_limit=ALNS_TIME, seed=SEED + idx + 10000)
                orig_labels.append((a, s, obj, st))
                if (idx + 1) % 50 == 0:
                    avg_cvr = np.mean([l[3]["cvr_pct"] for l in orig_labels])
                    print(f"      {idx+1}/{len(orig_insts)}, avg_cvr={avg_cvr:.1f}%")

            avg_obj = np.mean([l[2] for l in orig_labels])
            avg_cvr = np.mean([l[3]["cvr_pct"] for l in orig_labels])
            print(f"    train_orig done: avg_obj={avg_obj:.2f}, avg_cvr={avg_cvr:.1f}%")

            # Replicate labels for augmented copies (orig + 3 mirrors = 4x)
            # The assignment and sequence are the same for mirrored instances
            train_labels = []
            for lab in orig_labels:
                train_labels.append(lab)       # original
                train_labels.append(lab)       # mirror 1
                train_labels.append(lab)       # mirror 2
                train_labels.append(lab)       # mirror 3
            labels["train"] = train_labels
            print(f"    train (with augmented labels): {len(train_labels)} labels")
        elif "train" in data:
            # Fallback: solve all training instances (XL case)
            insts = data["train"]
            print(f"    train: {len(insts)} instances")
            split_labels = []
            for idx, inst in enumerate(insts):
                a, s, obj, st = solve_alns(inst, time_limit=ALNS_TIME, seed=SEED + idx)
                split_labels.append((a, s, obj, st))
            labels["train"] = split_labels

        with open(os.path.join(DATA_DIR, f"{scale}_labels.pkl"), "wb") as f:
            pickle.dump(labels, f)

    # Table 1: dataset config + label quality
    rows = []
    for scale in ["S", "M", "L", "XL"]:
        cfg = SCALES[scale]
        st = label_stats.get(scale, {})
        train_n = TRAIN_SIZES.get(scale, 0)
        rows.append([
            scale, cfg["N"], cfg["M"],
            train_n, train_n * 4 if train_n > 0 else "—",
            VAL_SIZE if scale != "XL" else "—", TEST_SIZE,
            f"{st.get('avg_obj', 0):.2f}",
            f"{st.get('avg_cvr', 0):.1f}",
            f"{st.get('avg_tw_sat', 0):.1f}",
        ])
    write_csv("table1_dataset_config.csv",
              ["Scale", "N(robots)", "M(tasks)", "Train", "Train(aug)",
               "Val", "Test", "ALNS_Obj", "ALNS_CVR(%)", "ALNS_TW_Sat(%)"],
              rows, note=REDUCED_NOTE if REDUCED_RUN else None)

    return label_stats


# ══════════════════════════════════════════════════════
# STEP 3: Train POMO baseline
# ══════════════════════════════════════════════════════
def step3():
    print("\n" + "="*60)
    print("STEP 3: Train POMO baseline")
    print("="*60)
    from train_sl import train_pomo_sl
    pomo = train_pomo_sl(epochs=SL_EPOCHS)
    return pomo


# ══════════════════════════════════════════════════════
# STEP 4: Train CAHT Stage I (SL)
# ══════════════════════════════════════════════════════
def step4():
    print("\n" + "="*60)
    print("STEP 4: Train CAHT Stage I (SL)")
    print("="*60)
    from train_sl import train_sl
    caht_sl = train_sl(epochs=SL_EPOCHS)
    return caht_sl


# ══════════════════════════════════════════════════════
# STEP 5: Train CAHT Stage II (RL)
# ══════════════════════════════════════════════════════
def step5(caht_sl):
    print("\n" + "="*60)
    print("STEP 5: Train CAHT Stage II (RL)")
    print("="*60)
    from train_rl import train_rl
    import copy
    caht_rl = train_rl(model=copy.deepcopy(caht_sl), epochs=RL_EPOCHS)
    return caht_rl


# ══════════════════════════════════════════════════════
# STEP 6: Evaluate all methods
# ══════════════════════════════════════════════════════
def step6(caht_sl, caht_rl, pomo):
    print("\n" + "="*60)
    print("STEP 6: Evaluate all methods")
    print("="*60)
    from evaluate import eval_all_methods

    all_results = {}
    for scale in ["S", "M", "L"]:
        results = eval_all_methods(scale, caht_sl, caht_rl, pomo)
        all_results[scale] = results

    # Generate table2a/b/c
    for scale, suffix in [("S", "a"), ("M", "b"), ("L", "c")]:
        results = all_results[scale]
        rows = []
        for method in ["Nearest Greedy", "OR-Tools", "ALNS(30s)", "POMO",
                       "CAHT(SL)", "CAHT(SL+RL)"]:
            if method in results:
                m = results[method]
                rows.append([
                    method,
                    f"{m['obj']:.2f}",
                    f"{m.get('gap_pct', 0):.1f}",
                    f"{m['cvr_pct']:.1f}",
                    f"{m['tw_sat']:.1f}",
                    f"{m['makespan']:.2f}",
                    f"{m['time_ms']:.1f}",
                ])
        write_csv(f"table2{suffix}_{scale.lower()}_results.csv",
                  ["Method", "Obj", "Gap(%)", "CVR(%)", "TW_Sat(%)", "Makespan", "Time(ms)"],
                  rows, note=REDUCED_NOTE if REDUCED_RUN else None)

    return all_results


# ══════════════════════════════════════════════════════
# STEP 7: Ablation study (M scale)
# ══════════════════════════════════════════════════════
def step7(caht_rl):
    print("\n" + "="*60)
    print("STEP 7: Ablation study")
    print("="*60)
    from evaluate import eval_caht, aggregate_results
    from model.caht import CAHT
    import copy

    with open(os.path.join(DATA_DIR, "M.pkl"), "rb") as f:
        data = pickle.load(f)
    test_insts = data["test"]

    variants = {}

    # Full model
    print("  Full model (SL+RL)...")
    results = eval_caht(caht_rl, test_insts)
    variants["CAHT Full"] = aggregate_results(results)

    # w/o RL (load SL checkpoint)
    print("  w/o RL...")
    caht_sl = CAHT().to(DEVICE)
    sl_ckpt = os.path.join(CKPT_DIR, "caht_sl.pt")
    if os.path.exists(sl_ckpt):
        caht_sl.load_state_dict(torch.load(sl_ckpt, map_location=DEVICE, weights_only=True))
    results = eval_caht(caht_sl, test_insts)
    variants["w/o RL"] = aggregate_results(results)

    # w/o dynamic masking
    print("  w/o dynamic masking...")
    caht_nomask = copy.deepcopy(caht_rl)
    orig_forward = caht_nomask.assign_dec.forward
    def no_mask_forward(h_robots, h_tasks, robots_data, tasks_data, greedy=True, return_log_probs=False):
        import copy as cp
        modified_rd = []
        for b_rd in robots_data:
            new_rd = []
            for r in b_rd:
                r2 = cp.copy(r)
                r2["cap"] = 99999
                r2["battery"] = 99999
                new_rd.append(r2)
            modified_rd.append(new_rd)
        return orig_forward(h_robots, h_tasks, modified_rd, tasks_data, greedy, return_log_probs)
    caht_nomask.assign_dec.forward = no_mask_forward
    results = eval_caht(caht_nomask, test_insts)
    variants["w/o mask"] = aggregate_results(results)

    # w/o spatial bias
    print("  w/o spatial bias...")
    caht_nospatial = copy.deepcopy(caht_rl)
    for layer in caht_nospatial.encoder.layers:
        for p in layer.attn.spatial_mlp.parameters():
            p.data.zero_()
    results = eval_caht(caht_nospatial, test_insts)
    variants["w/o spatial"] = aggregate_results(results)

    # MLP encoder
    print("  MLP encoder...")
    caht_mlp = copy.deepcopy(caht_rl)
    class MLPEncoder(torch.nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, 512), torch.nn.ReLU(),
                torch.nn.Linear(512, d),
            )
        def forward(self, H0, dist_matrix):
            return self.net(H0)
    caht_mlp.encoder = MLPEncoder().to(DEVICE)
    results = eval_caht(caht_mlp, test_insts)
    variants["MLP encoder"] = aggregate_results(results)

    # output table
    full_obj = variants["CAHT Full"]["obj"]
    rows = []
    for name, m in variants.items():
        delta = (m["obj"] - full_obj) / full_obj * 100 if full_obj > 0 else 0
        rows.append([
            name, f"{m['obj']:.2f}", f"{delta:+.1f}",
            f"{m['cvr_pct']:.1f}", f"{m['tw_sat']:.1f}", f"{m['time_ms']:.1f}",
        ])
    write_csv("table3_ablation.csv",
              ["Variant", "Obj", "ΔObj(%)", "CVR(%)", "TW_Sat(%)", "Time(ms)"],
              rows, note=REDUCED_NOTE if REDUCED_RUN else None)

    return variants


# ══════════════════════════════════════════════════════
# STEP 8: Generalization (train on S+M+L, test on XL)
# ══════════════════════════════════════════════════════
def step8(caht_rl):
    print("\n" + "="*60)
    print("STEP 8: Generalization experiment")
    print("="*60)
    from evaluate import eval_caht, aggregate_results

    rows = []
    for scale in ["S", "M", "L", "XL"]:
        with open(os.path.join(DATA_DIR, f"{scale}.pkl"), "rb") as f:
            data = pickle.load(f)
        test_insts = data["test"]

        caht_results = eval_caht(caht_rl, test_insts)
        caht_agg = aggregate_results(caht_results)

        # ALNS baseline
        alns_path = os.path.join(DATA_DIR, f"{scale}_labels.pkl")
        if os.path.exists(alns_path):
            with open(alns_path, "rb") as f:
                labels = pickle.load(f)
            alns_objs = [labels["test"][i][3]["obj"]
                         for i in range(min(len(test_insts), len(labels["test"])))]
            alns_obj = np.mean(alns_objs)
        else:
            alns_obj = caht_agg["obj"]

        gap = (caht_agg["obj"] - alns_obj) / alns_obj * 100 if alns_obj > 0 else 0
        rows.append([
            scale, f"{caht_agg['obj']:.2f}", f"{gap:.1f}",
            f"{caht_agg['cvr_pct']:.1f}", f"{caht_agg['tw_sat']:.1f}",
        ])

    write_csv("table4_generalization.csv",
              ["Test_Scale", "Obj", "Gap_vs_ALNS(%)", "CVR(%)", "TW_Sat(%)"],
              rows, note=REDUCED_NOTE if REDUCED_RUN else None)


# ══════════════════════════════════════════════════════
# STEP 9: Latency breakdown
# ══════════════════════════════════════════════════════
def step9(caht_rl):
    print("\n" + "="*60)
    print("STEP 9: Latency breakdown")
    print("="*60)
    from evaluate import latency_breakdown

    rows = []
    for scale in ["S", "M", "L"]:
        with open(os.path.join(DATA_DIR, f"{scale}.pkl"), "rb") as f:
            data = pickle.load(f)
        test_insts = data["test"][:20]
        N = SCALES[scale]["N"]

        breakdown = latency_breakdown(caht_rl, test_insts, N)
        total = sum(breakdown.values())
        rows.append([
            scale,
            f"{breakdown['embedding']:.2f}",
            f"{breakdown['encoder']:.2f}",
            f"{breakdown['assign_dec']:.2f}",
            f"{breakdown['seq_dec']:.2f}",
            f"{total:.2f}",
        ])

    write_csv("table5_latency_breakdown.csv",
              ["Scale", "Embedding(ms)", "Encoder(ms)", "Assign_Dec(ms)", "Seq_Dec(ms)", "Total(ms)"],
              rows)


# ══════════════════════════════════════════════════════
# STEP 10: Online simulation
# ══════════════════════════════════════════════════════
def step10(caht_rl, pomo):
    print("\n" + "="*60)
    print("STEP 10: Online simulation")
    print("="*60)
    from online_simulation import run_all_online

    results = run_all_online(caht_model=caht_rl, pomo_model=pomo)
    rows = []
    for r in results:
        rows.append([
            r["method"], r["completed"],
            f"{r['tw_sat']:.1f}", f"{r['avg_wait']:.1f}",
            f"{r['throughput']:.2f}", f"{r['solve_time_ms']:.1f}",
        ])
    write_csv("table6_online_simulation.csv",
              ["Method", "Completed", "TW_Sat(%)", "Avg_Wait(s)", "Throughput(tasks/min)", "Solve_Time(ms)"],
              rows)


# ══════════════════════════════════════════════════════
# STEP 11: Inference time scaling curve
# ══════════════════════════════════════════════════════
def step11(caht_rl):
    print("\n" + "="*60)
    print("STEP 11: Inference time scaling curve")
    print("="*60)
    from data_generator import generate_instance
    from alns_solver import solve_alns
    from ortools_solver import solve_ortools

    caht_times = []
    caht_edge_times = []
    alns_times = []
    ortools_times = []
    problem_sizes = []

    rng = np.random.RandomState(SEED)
    for N, M in SCALING_SIZES:
        print(f"  N={N}, M={M} (N+M={N+M})...")
        inst = generate_instance(N, M, rng)
        problem_sizes.append(N + M)

        # CAHT GPU
        caht_rl.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(5):
                caht_rl([inst], greedy=True)
            caht_t = (time.perf_counter() - t0) / 5 * 1000
        caht_times.append(caht_t)
        caht_edge_times.append(caht_t * 5)

        # ALNS
        t0 = time.perf_counter()
        solve_alns(inst, time_limit=5)
        alns_times.append((time.perf_counter() - t0) * 1000)

        # OR-Tools
        t0 = time.perf_counter()
        solve_ortools(inst, time_limit=5)
        ortools_times.append((time.perf_counter() - t0) * 1000)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
    ax.plot(problem_sizes, caht_times, 'b-o', label='CAHT-GPU', markersize=6)
    ax.plot(problem_sizes, caht_edge_times, color='orange', linestyle='--', marker='s',
            label='CAHT-Edge(est.)', markersize=6)
    ax.plot(problem_sizes, alns_times, 'g-^', label='ALNS', markersize=6)
    ax.plot(problem_sizes, ortools_times, color='purple', linestyle='-', marker='D',
            label='OR-Tools', markersize=6)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, label='Real-time threshold (100ms)')

    ax.set_xlabel('Problem Scale (N+M)', fontsize=12)
    ax.set_ylabel('Inference Time (ms)', fontsize=12)
    ax.set_yscale('log')
    ax.set_xticks(problem_sizes)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_title('Inference Time Scaling', fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_scaling_curve.png"), dpi=300, bbox_inches='tight')
    print("  Saved fig2_scaling_curve.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-step", type=int, default=1,
                        help="Resume from this step (skip earlier steps)")
    args, _ = parser.parse_known_args()
    start = args.start_step

    print("=" * 60)
    print("CAHT V2 — Full Experiment Pipeline")
    print(f"Device: {DEVICE}")
    if start > 1:
        print(f"Resuming from Step {start}")
    if REDUCED_RUN:
        print(f"NOTE: {REDUCED_NOTE}")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Data generation
    if start <= 1:
        step1()

    # Step 2: ALNS labels
    if start <= 2:
        step2()

    # Step 3: Train POMO
    pomo = step3()

    # Step 4: Train CAHT SL
    caht_sl = step4()

    # Step 5: Train CAHT RL
    caht_rl = step5(caht_sl)

    # Step 6: Evaluate
    step6(caht_sl, caht_rl, pomo)

    # Step 7: Ablation
    step7(caht_rl)

    # Step 8: Generalization
    step8(caht_rl)

    # Step 9: Latency breakdown
    step9(caht_rl)

    # Step 10: Online simulation
    step10(caht_rl, pomo)

    # Step 11: Scaling curve
    step11(caht_rl)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {total_time/60:.1f} minutes")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

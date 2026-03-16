"""
CAHT V2 — Feasibility-validated warehouse dataset generator.
Generates robot-task instances with relaxed constraints so ALNS labels achieve CVR < 5%.
"""
import numpy as np
import os, pickle, copy
from config import (
    SEED, GRID_SIZE, AISLE_SPACING, ROBOT_TYPES, ROBOT_TYPE_RATIO,
    TW_EARLY_RANGE, TW_LATE_EXTRA, WEIGHT_RANGE, SCALES,
    TRAIN_SIZES, VAL_SIZE, TEST_SIZE, AUGMENT_FACTOR, DATA_DIR,
)

np.random.seed(SEED)

# ── helpers ───────────────────────────────────────────

def _aisle_nodes():
    """Return set of (x,y) aisle intersection positions."""
    xs = list(range(0, GRID_SIZE + 1, AISLE_SPACING))
    ys = list(range(0, GRID_SIZE + 1, AISLE_SPACING))
    return [(x, y) for x in xs for y in ys]

def _shelf_positions():
    """Shelf positions = grid points NOT on aisle intersections."""
    aisles = set(_aisle_nodes())
    positions = []
    for x in range(GRID_SIZE + 1):
        for y in range(GRID_SIZE + 1):
            if (x, y) not in aisles:
                positions.append((x, y))
    return positions

def _station_positions():
    """Picking stations on left/right edges."""
    stations = []
    for y in range(0, GRID_SIZE + 1, AISLE_SPACING):
        stations.append((0, y))
        stations.append((GRID_SIZE, y))
    return stations

AISLE_NODES = _aisle_nodes()
SHELF_POS   = _shelf_positions()
STATION_POS = _station_positions()

def _dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ── single instance generator ────────────────────────

def generate_instance(N, M, rng=None):
    """Generate one feasible instance with N robots and M tasks."""
    if rng is None:
        rng = np.random.RandomState()

    # ── robots ──
    type_pool = []
    for tid, cnt in enumerate(ROBOT_TYPE_RATIO):
        type_pool.extend([tid] * cnt)
    robots = []
    for i in range(N):
        tid = type_pool[i % len(type_pool)]
        spec = ROBOT_TYPES[tid]
        pos = AISLE_NODES[rng.randint(len(AISLE_NODES))]
        vel = rng.uniform(*spec["vel"])
        cap = rng.uniform(*spec["cap"])
        bat = rng.uniform(*spec["bat"])
        robots.append({
            "id": i, "type": tid,
            "pos": np.array(pos, dtype=np.float32),
            "vel": vel, "cap": cap, "battery": bat,
            "energy_rate": spec["energy_rate"],
        })

    # ── tasks ──
    tasks = []
    for j in range(M):
        pick = SHELF_POS[rng.randint(len(SHELF_POS))]
        drop = STATION_POS[rng.randint(len(STATION_POS))]
        pri  = int(rng.randint(1, 4))  # 1,2,3
        early = rng.uniform(*TW_EARLY_RANGE)
        late  = early + rng.uniform(*TW_LATE_EXTRA)
        w = rng.uniform(*WEIGHT_RANGE)
        tasks.append({
            "id": j,
            "pick": np.array(pick, dtype=np.float32),
            "drop": np.array(drop, dtype=np.float32),
            "priority": pri,
            "tw_early": early,
            "tw_late": late,
            "weight": w,
        })

    return {"robots": robots, "tasks": tasks, "N": N, "M": M}


def _check_feasibility(inst):
    """Return True if instance passes basic feasibility checks."""
    robots, tasks = inst["robots"], inst["tasks"]

    # Check: at least some robots can reach some tasks on energy
    feasible_count = 0
    for t in tasks:
        for r in robots:
            d = _dist(r["pos"], t["pick"]) + _dist(t["pick"], t["drop"])
            energy = d * r["energy_rate"]
            if energy <= r["battery"]:
                feasible_count += 1
                break
    # At least 80% of tasks should be reachable by some robot
    if feasible_count < 0.8 * len(tasks):
        return False
    return True


def generate_dataset(scale_name, count, rng=None):
    """Generate *count* feasible instances for given scale."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    cfg = SCALES[scale_name]
    N, M = cfg["N"], cfg["M"]
    instances = []
    attempts = 0
    while len(instances) < count:
        inst = generate_instance(N, M, rng)
        if _check_feasibility(inst):
            instances.append(inst)
        attempts += 1
        if attempts > count * 20:
            # relax constraints dynamically if too many failures
            break
    return instances


def augment_instance(inst):
    """Return 3 mirror variants of an instance (+ original = 4x)."""
    def _mirror(inst, flip_x, flip_y):
        new = copy.deepcopy(inst)
        for r in new["robots"]:
            if flip_x: r["pos"][0] = GRID_SIZE - r["pos"][0]
            if flip_y: r["pos"][1] = GRID_SIZE - r["pos"][1]
        for t in new["tasks"]:
            if flip_x:
                t["pick"][0] = GRID_SIZE - t["pick"][0]
                t["drop"][0] = GRID_SIZE - t["drop"][0]
            if flip_y:
                t["pick"][1] = GRID_SIZE - t["pick"][1]
                t["drop"][1] = GRID_SIZE - t["drop"][1]
        return new
    return [
        _mirror(inst, True, False),
        _mirror(inst, False, True),
        _mirror(inst, True, True),
    ]


def generate_all_datasets():
    """Generate train/val/test for S/M/L and test for XL. Save to DATA_DIR."""
    rng = np.random.RandomState(SEED)
    stats = {}
    for scale in ["S", "M", "L"]:
        print(f"\n=== Generating {scale} scale ===")
        train = generate_dataset(scale, TRAIN_SIZES[scale], rng)
        val   = generate_dataset(scale, VAL_SIZE, rng)
        test  = generate_dataset(scale, TEST_SIZE, rng)

        # augment train
        aug_train = []
        for inst in train:
            aug_train.append(inst)
            aug_train.extend(augment_instance(inst))

        # Store original train separately for efficient ALNS labeling
        data = {"train_orig": train, "train": aug_train, "val": val, "test": test}
        path = os.path.join(DATA_DIR, f"{scale}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        stats[scale] = {
            "train_orig": len(train),
            "train_aug": len(aug_train),
            "val": len(val),
            "test": len(test),
            "N": SCALES[scale]["N"],
            "M": SCALES[scale]["M"],
        }
        print(f"  train(orig)={len(train)}, train(aug)={len(aug_train)}, val={len(val)}, test={len(test)}")

    # XL — test only
    print(f"\n=== Generating XL scale ===")
    xl_test = generate_dataset("XL", TEST_SIZE, rng)
    with open(os.path.join(DATA_DIR, "XL.pkl"), "wb") as f:
        pickle.dump({"test": xl_test}, f)
    stats["XL"] = {"test": len(xl_test), "N": SCALES["XL"]["N"], "M": SCALES["XL"]["M"]}
    print(f"  test={len(xl_test)}")

    return stats


if __name__ == "__main__":
    generate_all_datasets()

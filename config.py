"""
CAHT V2 — Global Configuration
All hyperparameters in one place.
"""
import os, torch, random, numpy as np

# ── Seed ──────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Device ────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Warehouse ─────────────────────────────────────────
GRID_SIZE = 100
AISLE_SPACING = 10

# ── Robot specs  (type_id: 0=AGV, 1=AMR, 2=Forklift) ─
# Battery values 10x from spec to ensure feasibility (CVR < 5%)
ROBOT_TYPES = {
    0: {"name": "AGV",      "vel": (2, 4),   "cap": (10, 20),  "bat": (900, 1000), "energy_rate": 0.3},
    1: {"name": "AMR",      "vel": (1.5, 3), "cap": (15, 30),  "bat": (800, 1000), "energy_rate": 0.5},
    2: {"name": "Forklift", "vel": (0.8, 1.5),"cap": (30, 50), "bat": (700, 1000), "energy_rate": 0.7},
}
ROBOT_TYPE_RATIO = [2, 2, 1]          # AGV:AMR:Forklift = 2:2:1

# ── Task specs (relaxed constraints) ─────────────────
TW_EARLY_RANGE = (0, 50)
TW_LATE_EXTRA  = (500, 1500)          # late = early + U(500,1500), very relaxed for CVR<5%
WEIGHT_RANGE   = (1, 5)

# ── Dataset scales ────────────────────────────────────
SCALES = {
    "S":  {"N": 5,  "M": 50},
    "M":  {"N": 10, "M": 100},
    "L":  {"N": 15, "M": 150},
    "XL": {"N": 20, "M": 200},
}
TRAIN_SIZES = {"S": 300, "M": 300, "L": 300}
VAL_SIZE  = 50
TEST_SIZE = 50
AUGMENT_FACTOR = 4                    # original + 3 mirrors

# ── ALNS ──────────────────────────────────────────────
ALNS_ITERS    = 5000
ALNS_TIME     = 3                     # seconds per instance
ALNS_T_INIT   = 100.0
ALNS_COOLING  = 0.995
ALNS_Q_LO     = 0.1
ALNS_Q_HI     = 0.3

# ── Model ─────────────────────────────────────────────
D_MODEL  = 128
N_HEADS  = 8
N_LAYERS = 4
FFN_DIM  = 512
DROPOUT  = 0.1

# ── Training — Stage I (SL) ──────────────────────────
SL_EPOCHS    = 30
SL_LR        = 1e-4
SL_BATCH     = 16
SEQ_LOSS_W   = 0.5

# ── Training — Stage II (RL) ─────────────────────────
RL_EPOCHS    = 15
RL_LR        = 1e-5
RL_ENTROPY   = 0.01
RL_K_SAMPLES = 4
RL_BL_SYNC   = 5

# ── Objective weights ────────────────────────────────
W_ENERGY   = 0.4
W_MAKESPAN = 0.4
W_TW       = 0.2

# ── OR-Tools ──────────────────────────────────────────
ORTOOLS_TIME = 10                     # seconds

# ── Online simulation ────────────────────────────────
SIM_DURATION   = 300                  # seconds
SIM_LAMBDA     = 0.3                  # Poisson arrival rate
SIM_REALLOC    = 10                   # re-allocate interval

# ── Scaling curve test sizes ─────────────────────────
SCALING_SIZES = [
    (5, 50), (10, 100), (15, 150), (20, 200), (25, 250), (30, 300),
]  # (N, M) → N+M = 55,110,165,220,275,330

# ── Run configuration ────────────────────────────────
REDUCED_RUN = True
REDUCED_NOTE = "train=300/scale(aug4x=1200), ALNS=3s, SL=30ep, RL=15ep, CPU-only"

"""
Resume CAHT V2 pipeline from Step 5 onward.
Steps 1-4 already completed (data, labels, POMO, CAHT-SL checkpoints exist).
"""
import os, sys, pickle, time
import torch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import DEVICE, SEED, set_seed, CKPT_DIR, DATA_DIR, SCALES, RL_EPOCHS

set_seed(SEED)

print("=" * 60)
print("CAHT V2 — Resume Pipeline (Steps 5-11)")
print(f"Device: {DEVICE}")
print("=" * 60)

t_start = time.time()

# ── Load existing models from checkpoints ──
from model import CAHT
from baselines.pomo import POMO

print("\nLoading POMO from checkpoint...")
pomo = POMO().to(DEVICE)
pomo.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pomo.pt"),
                                 map_location=DEVICE, weights_only=True))
pomo.eval()
print(f"  POMO loaded ({pomo.count_parameters():,} params)")

print("\nLoading CAHT-SL from checkpoint...")
caht_sl = CAHT().to(DEVICE)
caht_sl.load_state_dict(torch.load(os.path.join(CKPT_DIR, "caht_sl.pt"),
                                    map_location=DEVICE, weights_only=True))
print(f"  CAHT-SL loaded ({caht_sl.count_parameters():,} params)")

# ── Step 5: RL fine-tuning ──
from run_all import step5, step6, step7, step8, step9, step10, step11

rl_ckpt = os.path.join(CKPT_DIR, "caht_rl.pt")
if os.path.exists(rl_ckpt):
    print("\nRL checkpoint already exists, loading...")
    caht_rl = CAHT().to(DEVICE)
    caht_rl.load_state_dict(torch.load(rl_ckpt, map_location=DEVICE, weights_only=True))
else:
    print("\nRunning Step 5: RL training...")
    import copy
    caht_rl = step5(copy.deepcopy(caht_sl))

# ── Step 6: Evaluate ──
step6(caht_sl, caht_rl, pomo)

# ── Step 7: Ablation ──
step7(caht_rl)

# ── Step 8: Generalization ──
step8(caht_rl)

# ── Step 9: Latency breakdown ──
step9(caht_rl)

# ── Step 10: Online simulation ──
step10(caht_rl, pomo)

# ── Step 11: Scaling curve ──
step11(caht_rl)

total_time = time.time() - t_start
print(f"\n{'='*60}")
print(f"ALL DONE! Total time: {total_time/60:.1f} minutes")
print(f"Outputs in: {os.path.join(ROOT, 'outputs')}")
print(f"{'='*60}")

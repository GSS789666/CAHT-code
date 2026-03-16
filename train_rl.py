"""
CAHT V2 — Stage II: REINFORCE fine-tuning.
Reward R = -J(π), curriculum learning across scales.
"""
import torch
import torch.optim as optim
import os, pickle, numpy as np, copy
from config import (
    DEVICE, RL_EPOCHS, RL_LR, RL_ENTROPY, RL_K_SAMPLES, RL_BL_SYNC,
    DATA_DIR, CKPT_DIR, SEED, set_seed, W_ENERGY, W_MAKESPAN, W_TW,
)
from model import CAHT
from alns_solver import compute_objective


def compute_reward_single(assignment, sequences, inst):
    """Compute reward for a single instance. R = -J."""
    if torch.is_tensor(assignment):
        a = assignment[0].cpu().tolist() if assignment.dim() > 1 else assignment.cpu().tolist()
    else:
        a = assignment[0] if isinstance(assignment, list) and isinstance(assignment[0], list) else assignment
    s = sequences[0] if isinstance(sequences, list) else sequences
    obj, *_ = compute_objective(a, s, inst["robots"], inst["tasks"])
    return -obj


def train_rl(model=None, epochs=RL_EPOCHS, save_name="caht_rl"):
    set_seed(SEED)

    if model is None:
        model = CAHT().to(DEVICE)
        ckpt = os.path.join(CKPT_DIR, "caht_sl.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            print("Loaded SL checkpoint for RL fine-tuning")

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=RL_LR)

    # load data by scale
    scale_data = {}
    for s in ("S", "M", "L"):
        with open(os.path.join(DATA_DIR, f"{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        scale_data[s] = data["train"]

    best_reward = -float("inf")

    for epoch in range(epochs):
        model.train()

        # curriculum: determine which scales to use
        if epoch < int(epochs * 0.3):
            scales = ["S"]
        elif epoch < int(epochs * 0.6):
            scales = ["S", "M"]
        else:
            scales = ["S", "M", "L"]

        chosen_scale = np.random.choice(scales)
        pool = scale_data[chosen_scale]

        # sample instances
        n_samples = min(8, len(pool))
        indices = np.random.permutation(len(pool))[:n_samples]

        epoch_reward = 0.0
        epoch_loss = 0.0

        optimizer.zero_grad()

        for idx in indices:
            inst = pool[idx]

            # K sampling trajectories - keep the best
            best_reward_k = -float("inf")
            best_lp = None

            for k in range(RL_K_SAMPLES):
                assignment, sequences, lp_a, lp_s = model(
                    [inst], greedy=False, return_log_probs=True)
                reward = compute_reward_single(assignment, sequences, inst)
                total_lp = lp_a.sum() + lp_s.sum()

                if reward > best_reward_k:
                    best_reward_k = reward
                    best_lp = total_lp

            # baseline from greedy rollout
            with torch.no_grad():
                bl_assign, bl_seq = baseline_model([inst], greedy=True)
                bl_reward = compute_reward_single(bl_assign, bl_seq, inst)

            advantage = best_reward_k - bl_reward

            # entropy bonus
            entropy_bonus = RL_ENTROPY * best_lp

            loss = -(advantage * best_lp + entropy_bonus) / n_samples
            loss.backward()

            epoch_reward += best_reward_k
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_reward = epoch_reward / n_samples

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  RL Epoch {epoch+1}/{epochs}, reward={avg_reward:.2f}, "
                  f"scale={chosen_scale}, loss={epoch_loss:.4f}")

        # sync baseline
        if (epoch + 1) % RL_BL_SYNC == 0:
            baseline_model.load_state_dict(model.state_dict())
            baseline_model.eval()

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{save_name}.pt"))

    print(f"RL training done. Best reward: {best_reward:.2f}")
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{save_name}.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


if __name__ == "__main__":
    train_rl()

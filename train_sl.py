"""
CAHT V2 — Stage I: Supervised pre-training with ALNS labels.
Loss = CrossEntropy(assignment) + 0.5 * NLL(sequence)
Uses teacher forcing with actual ALNS labels as targets.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os, pickle, time, numpy as np
from config import (
    DEVICE, SL_EPOCHS, SL_LR, SL_BATCH, SEQ_LOSS_W,
    DATA_DIR, CKPT_DIR, SEED, set_seed,
)
from model import CAHT


def load_labeled_data_by_scale(scales=("S", "M", "L"), split="train"):
    """Load instances + ALNS labels grouped by scale."""
    data_by_scale = {}
    for s in scales:
        with open(os.path.join(DATA_DIR, f"{s}.pkl"), "rb") as f:
            data = pickle.load(f)
        with open(os.path.join(DATA_DIR, f"{s}_labels.pkl"), "rb") as f:
            lab = pickle.load(f)
        insts = data[split]
        labs = lab[split]
        data_by_scale[s] = list(zip(insts, labs))
    return data_by_scale


def _train_epoch(model, optimizer, data_by_scale, batch_size):
    """Train one epoch using teacher-forced supervised loss with ALNS labels."""
    model.train()
    total_loss = 0.0
    total_a_loss = 0.0
    total_s_loss = 0.0
    n_batches = 0

    for scale, pairs in data_by_scale.items():
        indices = np.random.permutation(len(pairs))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 1:
                continue

            optimizer.zero_grad()
            batch_loss = torch.zeros(1, device=DEVICE)
            batch_a = torch.zeros(1, device=DEVICE)
            batch_s = torch.zeros(1, device=DEVICE)

            # process instances one at a time (CPU-friendly)
            for bi in batch_idx:
                inst = pairs[bi][0]
                label = pairs[bi][1]
                # label format: (assignment_list, sequence_dict, obj, stats)
                target_assign = label[0]  # list[j] = robot_id
                target_seq = label[1]     # dict{robot_id: [task_ids]}

                assign_loss, seq_loss = model.compute_sl_loss(
                    [inst], [target_assign], [target_seq])

                loss = (assign_loss + SEQ_LOSS_W * seq_loss) / len(batch_idx)
                loss.backward()
                batch_loss = batch_loss + loss.detach()
                batch_a = batch_a + assign_loss.detach() / len(batch_idx)
                batch_s = batch_s + seq_loss.detach() / len(batch_idx)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            total_a_loss += batch_a.item()
            total_s_loss += batch_s.item()
            n_batches += 1

    avg = lambda x: x / max(n_batches, 1)
    return avg(total_loss), avg(total_a_loss), avg(total_s_loss)


def train_sl(model=None, epochs=SL_EPOCHS, save_name="caht_sl"):
    set_seed(SEED)

    if model is None:
        model = CAHT().to(DEVICE)
    print(f"CAHT parameters: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=SL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    data_by_scale = load_labeled_data_by_scale(("S", "M", "L"), "train")
    total = sum(len(v) for v in data_by_scale.values())
    print(f"Training instances: {total}")

    best_loss = float("inf")
    for epoch in range(epochs):
        t0 = time.time()
        avg_loss, avg_a, avg_s = _train_epoch(model, optimizer, data_by_scale, SL_BATCH)
        scheduler.step()
        elapsed = time.time() - t0

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f} "
                  f"(assign={avg_a:.4f}, seq={avg_s:.4f}), "
                  f"lr={scheduler.get_last_lr()[0]:.6f}, time={elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{save_name}.pt"))

    print(f"SL training done. Best loss: {best_loss:.4f}")
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{save_name}.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


def train_pomo_sl(epochs=SL_EPOCHS, save_name="pomo"):
    """
    Train POMO baseline with REINFORCE using actual objective as reward.
    POMO architecture doesn't support teacher forcing easily,
    so we use REINFORCE with ALNS baseline.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from baselines.pomo import POMO, pomo_assignment_to_sequence
    from alns_solver import compute_objective

    set_seed(SEED)
    model = POMO().to(DEVICE)
    print(f"POMO parameters: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=SL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Load training data with labels (for baseline comparison)
    data_by_scale = load_labeled_data_by_scale(("S", "M", "L"), "train")
    total = sum(len(v) for v in data_by_scale.values())
    print(f"POMO training instances: {total}")

    best_reward = -float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_reward = 0.0
        n_samples = 0

        for scale, pairs in data_by_scale.items():
            indices = np.random.permutation(len(pairs))
            for start in range(0, len(indices), SL_BATCH):
                batch_idx = indices[start:start + SL_BATCH]
                if len(batch_idx) < 1:
                    continue

                optimizer.zero_grad()
                batch_loss = torch.zeros(1, device=DEVICE)

                for bi in batch_idx:
                    inst = pairs[bi][0]
                    label = pairs[bi][1]
                    alns_obj = label[2]  # ALNS objective value as baseline

                    # sample from policy
                    assignment, log_probs = model([inst], greedy=False)
                    seqs = pomo_assignment_to_sequence(assignment, [inst])

                    a_list = assignment[0].cpu().tolist()
                    s_dict = seqs[0]
                    obj, *_ = compute_objective(a_list, s_dict,
                                                inst["robots"], inst["tasks"])

                    reward = -obj
                    baseline = -alns_obj
                    advantage = reward - baseline

                    # REINFORCE loss
                    loss = -(advantage * log_probs.sum()) / len(batch_idx)
                    loss.backward()
                    batch_loss = batch_loss + loss.detach()
                    total_reward += reward
                    n_samples += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += batch_loss.item()

        scheduler.step()
        avg_reward = total_reward / max(n_samples, 1)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  POMO Epoch {epoch+1}/{epochs}, avg_reward={avg_reward:.2f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{save_name}.pt"))

    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{save_name}.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


if __name__ == "__main__":
    train_sl()

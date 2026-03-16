"""
CAHT V2 — Constraint-aware assignment decoder with dynamic masking.
Bilinear attention: s_{ij} = h_i^T W h_j + v^T [h_i || h_j]
Dynamic mask: capacity + energy feasibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import D_MODEL


class AssignmentDecoder(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d = d_model
        self.W_a = nn.Linear(d_model, d_model, bias=False)
        self.v_a = nn.Linear(2 * d_model, 1, bias=False)

    def _compute_scores(self, h_robots, h_tasks):
        """
        h_robots: (B, N, d)
        h_tasks:  (B, M, d)
        Returns: (B, N, M)
        """
        B, N, d = h_robots.shape
        M = h_tasks.shape[1]
        # bilinear: h_i^T W h_j
        Wh = self.W_a(h_robots)                       # (B,N,d)
        bilinear = torch.bmm(Wh, h_tasks.transpose(1, 2))  # (B,N,M)
        # concat: v^T [h_i || h_j]
        h_r_exp = h_robots.unsqueeze(2).expand(B, N, M, d)
        h_t_exp = h_tasks.unsqueeze(1).expand(B, N, M, d)
        concat = torch.cat([h_r_exp, h_t_exp], dim=-1)  # (B,N,M,2d)
        additive = self.v_a(concat).squeeze(-1)          # (B,N,M)
        return (bilinear + additive) / math.sqrt(d)

    def _compute_mask(self, b, j, robots_data, tasks_data, cap_rem, bat_rem, pos_cur, e_rate, N, device):
        """Compute feasibility mask for task j in batch element b."""
        mask = torch.zeros(N, device=device)
        t = tasks_data[b][j]
        tw = t["weight"]
        pick = torch.tensor([t["pick"][0], t["pick"][1]], device=device)
        drop = torch.tensor([t["drop"][0], t["drop"][1]], device=device)
        for i in range(N):
            # capacity check: per-task (robot can carry this item)
            if robots_data[b][i]["cap"] < tw:
                mask[i] = float("-inf")
                continue
            # energy check: enough battery for round trip
            d_total = torch.dist(pos_cur[b, i], pick) + torch.dist(pick, drop)
            energy_needed = d_total * e_rate[b, i]
            if energy_needed > bat_rem[b, i]:
                mask[i] = float("-inf")
        return mask

    def _update_state(self, b, i, j, tasks_data, bat_rem, pos_cur, e_rate, device):
        """Update robot i's state after assigning task j."""
        t = tasks_data[b][j]
        pick = torch.tensor([t["pick"][0], t["pick"][1]], device=device)
        drop = torch.tensor([t["drop"][0], t["drop"][1]], device=device)
        d = torch.dist(pos_cur[b, i], pick) + torch.dist(pick, drop)
        bat_rem[b, i] -= d * e_rate[b, i]
        pos_cur[b, i] = drop

    def _init_state(self, B, N, robots_data, device):
        """Initialize robot states for sequential assignment."""
        bat_rem = torch.zeros(B, N, device=device)
        pos_cur = torch.zeros(B, N, 2, device=device)
        e_rate = torch.zeros(B, N, device=device)
        for b in range(B):
            for i in range(N):
                r = robots_data[b][i]
                bat_rem[b, i] = float(r["battery"])
                pos_cur[b, i, 0] = float(r["pos"][0])
                pos_cur[b, i, 1] = float(r["pos"][1])
                e_rate[b, i] = float(r["energy_rate"])
        return bat_rem, pos_cur, e_rate

    def forward(self, h_robots, h_tasks, robots_data, tasks_data,
                greedy=True, return_log_probs=False):
        """
        Greedy sequential assignment with dynamic state update.
        Returns: assignment (B, M) LongTensor, log_probs (B, M) if requested
        """
        B, N, d = h_robots.shape
        M = h_tasks.shape[1]
        device = h_robots.device

        scores = self._compute_scores(h_robots, h_tasks)  # (B,N,M)
        bat_rem, pos_cur, e_rate = self._init_state(B, N, robots_data, device)

        assignment = torch.full((B, M), -1, dtype=torch.long, device=device)
        log_probs_list = []

        # task priority ordering
        priorities = torch.zeros(B, M, device=device)
        for b in range(B):
            for j in range(M):
                priorities[b, j] = tasks_data[b][j]["priority"]
        task_order = torch.argsort(priorities, dim=1, descending=True)  # (B, M)

        for step in range(M):
            j_indices = task_order[:, step]  # (B,)

            # gather scores for task j
            j_scores = torch.zeros(B, N, device=device)
            for b in range(B):
                j = j_indices[b].item()
                j_scores[b] = scores[b, :, j]

            # compute mask
            mask = torch.zeros(B, N, device=device)
            for b in range(B):
                j = j_indices[b].item()
                mask[b] = self._compute_mask(
                    b, j, robots_data, tasks_data,
                    None, bat_rem, pos_cur, e_rate, N, device)

            logits = j_scores + mask
            # handle all-masked
            all_masked = (mask == float("-inf")).all(dim=-1)
            logits[all_masked] = 0.0
            probs = torch.softmax(logits, dim=-1)

            if greedy:
                chosen = torch.argmax(probs, dim=-1)
            else:
                chosen = torch.multinomial(probs, 1).squeeze(-1)

            log_p = torch.log(probs.gather(1, chosen.unsqueeze(1)) + 1e-8).squeeze(-1)
            log_probs_list.append(log_p)

            # update state
            for b in range(B):
                j = j_indices[b].item()
                i = chosen[b].item()
                assignment[b, j] = i
                self._update_state(b, i, j, tasks_data, bat_rem, pos_cur, e_rate, device)

        log_probs = torch.stack(log_probs_list, dim=1)  # (B, M)
        if return_log_probs:
            return assignment, log_probs
        return assignment

    def supervised_loss(self, h_robots, h_tasks, robots_data, tasks_data,
                        target_assignment):
        """
        Teacher-forced supervised loss for assignment.
        target_assignment: list of lists, target_assignment[b][j] = robot_id
        Returns: scalar loss (mean cross-entropy)
        """
        B, N, d = h_robots.shape
        M = h_tasks.shape[1]
        device = h_robots.device

        scores = self._compute_scores(h_robots, h_tasks)  # (B,N,M)
        bat_rem, pos_cur, e_rate = self._init_state(B, N, robots_data, device)

        # task priority ordering
        priorities = torch.zeros(B, M, device=device)
        for b in range(B):
            for j in range(M):
                priorities[b, j] = tasks_data[b][j]["priority"]
        task_order = torch.argsort(priorities, dim=1, descending=True)

        total_loss = torch.zeros(1, device=device)
        n_valid = 0

        for step in range(M):
            j_indices = task_order[:, step]

            # gather scores and masks
            logits_batch = torch.zeros(B, N, device=device)
            targets_batch = torch.zeros(B, dtype=torch.long, device=device)

            for b in range(B):
                j = j_indices[b].item()
                logits_batch[b] = scores[b, :, j]

                # No feasibility mask during SL training — let the model
                # learn the assignment pattern purely from ALNS labels.
                # Mask is only applied at inference time.

                # target
                target_rid = target_assignment[b][j]
                if target_rid < 0:
                    target_rid = 0  # fallback if unassigned in label
                targets_batch[b] = target_rid

            # cross-entropy loss
            loss = F.cross_entropy(logits_batch, targets_batch, reduction='sum')
            total_loss = total_loss + loss
            n_valid += B

            # teacher forcing: update state using TARGET assignment
            for b in range(B):
                j = j_indices[b].item()
                target_rid = target_assignment[b][j]
                if target_rid >= 0:
                    self._update_state(b, target_rid, j, tasks_data,
                                       bat_rem, pos_cur, e_rate, device)

        return total_loss / max(n_valid, 1)

"""
CAHT V2 — Autoregressive sequencing decoder (GRU + Pointer attention).
Per-robot: orders the tasks assigned to that robot.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import D_MODEL


class SequencingDecoder(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d = d_model
        self.gru = nn.GRUCell(d_model, d_model)
        self.start_token = nn.Parameter(torch.randn(d_model))

    def _decode_robot_sequence(self, h_cands, h_robot, task_ids, tasks_data_b,
                               greedy, device):
        """
        Decode task sequence for a single robot.
        h_cands: (K, d) - embeddings of assigned tasks
        h_robot: (d,) - robot embedding
        task_ids: list of task indices
        Returns: seq (list), log_probs (list of scalars)
        """
        K = len(task_ids)
        d = self.d
        hidden = h_robot
        remaining = list(range(K))
        seq = []
        log_probs = []
        inp = self.start_token
        cur_time = 0.0
        cur_pos = None

        for step in range(K):
            hidden = self.gru(inp.unsqueeze(0), hidden.unsqueeze(0)).squeeze(0)

            rem_h = h_cands[remaining]  # (R, d)
            scores = torch.matmul(rem_h, hidden) / math.sqrt(d)  # (R,)

            # TW mask
            if tasks_data_b is not None:
                mask = torch.zeros(len(remaining), device=device)
                for ri, kidx in enumerate(remaining):
                    j = task_ids[kidx]
                    t = tasks_data_b[j]
                    if cur_time > t["tw_late"]:
                        mask[ri] = float("-inf")
                scores = scores + mask

            if (scores == float("-inf")).all():
                scores = torch.zeros_like(scores)

            probs = torch.softmax(scores, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum()

            if greedy:
                chosen_ri = torch.argmax(probs).item()
            else:
                chosen_ri = torch.multinomial(probs, 1).item()

            lp = torch.log(probs[chosen_ri] + 1e-8)
            log_probs.append(lp)

            chosen_kidx = remaining[chosen_ri]
            chosen_j = task_ids[chosen_kidx]
            seq.append(chosen_j)
            inp = h_cands[chosen_kidx]
            remaining.remove(chosen_kidx)

            # update time estimate
            if tasks_data_b is not None:
                t = tasks_data_b[chosen_j]
                if cur_pos is not None:
                    d1 = math.sqrt((cur_pos[0] - t["pick"][0])**2 +
                                   (cur_pos[1] - t["pick"][1])**2)
                else:
                    d1 = 0
                d2 = math.sqrt((t["pick"][0] - t["drop"][0])**2 +
                               (t["pick"][1] - t["drop"][1])**2)
                cur_time += (d1 + d2) / 2.0
                cur_pos = t["drop"]

        return seq, log_probs

    def forward(self, h_all, assignment, N, tasks_data=None,
                greedy=True, return_log_probs=False):
        """
        h_all: (B, N+M, d)
        assignment: (B, M) LongTensor
        N: number of robots
        Returns: sequences dict, log_probs (B, total_steps)
        """
        B, S, d = h_all.shape
        M = S - N
        device = h_all.device

        h_robots = h_all[:, :N, :]
        h_tasks = h_all[:, N:, :]

        all_log_probs = []
        sequences = [{} for _ in range(B)]

        for b in range(B):
            for i in range(N):
                task_ids = (assignment[b] == i).nonzero(as_tuple=True)[0].tolist()
                if not task_ids:
                    sequences[b][i] = []
                    continue

                h_cands = h_tasks[b, task_ids, :]
                tasks_data_b = tasks_data[b] if tasks_data else None

                seq, lps = self._decode_robot_sequence(
                    h_cands, h_robots[b, i, :], task_ids, tasks_data_b,
                    greedy, device)

                sequences[b][i] = seq
                all_log_probs.extend(lps)

        if all_log_probs:
            log_probs = torch.stack(all_log_probs)
        else:
            log_probs = torch.zeros(1, device=device)

        if return_log_probs:
            return sequences, log_probs
        return sequences

    def supervised_loss(self, h_all, target_assignment, target_sequences,
                        N, tasks_data=None):
        """
        Teacher-forced supervised loss for sequencing.
        target_assignment: list of lists [B][j] = robot_id
        target_sequences: list of dicts [B]{robot_id: [task_ids in order]}
        Returns: scalar loss (mean NLL)
        """
        B, S, d = h_all.shape
        M = S - N
        device = h_all.device

        h_robots = h_all[:, :N, :]
        h_tasks = h_all[:, N:, :]

        total_loss = torch.zeros(1, device=device)
        n_steps = 0

        for b in range(B):
            for i in range(N):
                target_seq = target_sequences[b].get(i, [])
                if len(target_seq) < 2:
                    continue  # nothing to order

                # Get task embeddings for this robot's assigned tasks
                task_ids = target_seq  # use target assignment's task order
                h_cands = h_tasks[b, task_ids, :]  # (K, d)
                K = len(task_ids)

                hidden = h_robots[b, i, :]
                inp = self.start_token

                # Map task_id to local index
                id_to_local = {tid: k for k, tid in enumerate(task_ids)}

                remaining = list(range(K))
                cur_time = 0.0
                cur_pos = None

                for step in range(K):
                    hidden = self.gru(inp.unsqueeze(0), hidden.unsqueeze(0)).squeeze(0)

                    rem_h = h_cands[remaining]  # (R, d)
                    scores = torch.matmul(rem_h, hidden) / math.sqrt(d)

                    # TW mask
                    if tasks_data is not None:
                        mask = torch.zeros(len(remaining), device=device)
                        for ri, kidx in enumerate(remaining):
                            j = task_ids[kidx]
                            t = tasks_data[b][j]
                            if cur_time > t["tw_late"]:
                                mask[ri] = float("-inf")
                        scores = scores + mask

                    if (scores == float("-inf")).all():
                        scores = torch.zeros_like(scores)

                    log_probs = F.log_softmax(scores, dim=-1)

                    # target: the task that should be selected at this step
                    target_j = target_seq[step]
                    target_local = id_to_local[target_j]
                    # find index of target_local in remaining
                    if target_local in remaining:
                        target_ri = remaining.index(target_local)
                        step_loss = -log_probs[target_ri]
                        step_loss = torch.clamp(step_loss, max=100.0)
                        total_loss = total_loss + step_loss
                        n_steps += 1

                    # teacher forcing: use target task as next input
                    inp = h_cands[target_local]
                    if target_local in remaining:
                        remaining.remove(target_local)

                    # update time
                    if tasks_data is not None:
                        t = tasks_data[b][target_j]
                        if cur_pos is not None:
                            d1 = math.sqrt((cur_pos[0] - t["pick"][0])**2 +
                                           (cur_pos[1] - t["pick"][1])**2)
                        else:
                            d1 = 0
                        d2 = math.sqrt((t["pick"][0] - t["drop"][0])**2 +
                                       (t["pick"][1] - t["drop"][1])**2)
                        cur_time += (d1 + d2) / 2.0
                        cur_pos = t["drop"]

        return total_loss / max(n_steps, 1)

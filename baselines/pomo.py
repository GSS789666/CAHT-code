"""
CAHT V2 — Simplified but functional POMO baseline.
Transformer encoder (3 layers) + autoregressive decoder with pointer attention.
Trained with REINFORCE + 8 rollout augmentations.
Parameter count matched to CAHT (~0.8M).
"""
import torch
import torch.nn as nn
import math
import time
import numpy as np
from config import D_MODEL, DROPOUT, DEVICE


class POMOEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=8, n_layers=3, ffn_dim=512):
        super().__init__()
        # project combined features to d_model
        # robot: 5 dims, task: 8 dims → unified: max 8
        self.input_proj = nn.Linear(13, d_model)  # concat robot+task features
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        h = self.input_proj(x)
        return self.encoder(h)


class POMODecoder(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d = d_model
        self.gru = nn.GRUCell(d_model, d_model)
        self.start_token = nn.Parameter(torch.randn(d_model))
        self.ptr_W = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h_nodes, n_robots, greedy=True):
        """
        Simple autoregressive assignment: assign each task to a robot.
        h_nodes: (B, N+M, d)
        Returns: assignment (B, M), log_probs (B, M)
        """
        B, S, d = h_nodes.shape
        N = n_robots
        M = S - N
        device = h_nodes.device

        h_robots = h_nodes[:, :N, :]  # (B, N, d)
        h_tasks  = h_nodes[:, N:, :]  # (B, M, d)

        # context: mean of all nodes
        context = h_nodes.mean(dim=1)  # (B, d)

        assignments = []
        log_probs = []

        hidden = context
        inp = self.start_token.unsqueeze(0).expand(B, -1)

        for j in range(M):
            hidden = self.gru(inp, hidden)
            query = self.ptr_W(hidden)  # (B, d)
            scores = torch.bmm(h_robots, query.unsqueeze(2)).squeeze(2)  # (B, N)
            scores = scores / math.sqrt(d)
            probs = torch.softmax(scores, dim=-1)

            if greedy:
                chosen = torch.argmax(probs, dim=-1)
            else:
                chosen = torch.multinomial(probs, 1).squeeze(-1)

            lp = torch.log(probs.gather(1, chosen.unsqueeze(1)) + 1e-8).squeeze(-1)
            log_probs.append(lp)
            assignments.append(chosen)
            inp = h_tasks[:, j, :]

        assignment = torch.stack(assignments, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return assignment, log_probs


class POMO(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.encoder = POMOEncoder(d_model)
        self.decoder = POMODecoder(d_model)

    def _build_input(self, instances, device):
        """Build unified input tensor for POMO."""
        B = len(instances)
        N = instances[0]["N"]
        M = instances[0]["M"]
        S = N + M

        x = torch.zeros(B, S, 13, device=device)
        for b, inst in enumerate(instances):
            for i, r in enumerate(inst["robots"]):
                x[b, i, 0] = float(r["pos"][0])
                x[b, i, 1] = float(r["pos"][1])
                x[b, i, 2] = float(r["battery"])
                x[b, i, 3] = float(r["vel"])
                x[b, i, 4] = float(r["cap"])
            for j, t in enumerate(inst["tasks"]):
                x[b, N+j, 5]  = float(t["pick"][0])
                x[b, N+j, 6]  = float(t["pick"][1])
                x[b, N+j, 7]  = float(t["drop"][0])
                x[b, N+j, 8]  = float(t["drop"][1])
                x[b, N+j, 9]  = float(t["priority"])
                x[b, N+j, 10] = float(t["tw_early"])
                x[b, N+j, 11] = float(t["tw_late"])
                x[b, N+j, 12] = float(t["weight"])
        # Normalize features to ~[0,1] range
        # dims 0-4: robot [pos_x, pos_y, battery, vel, cap]
        # dims 5-12: task [pick_x, pick_y, drop_x, drop_y, priority, tw_e, tw_l, weight]
        scale = x.new_tensor([100., 100., 1000., 5., 20.,
                              100., 100., 100., 100., 3., 3000., 3000., 20.])
        x = x / scale
        return x, N

    def forward(self, instances, greedy=True):
        device = next(self.parameters()).device
        x, N = self._build_input(instances, device)
        h = self.encoder(x)
        assignment, log_probs = self.decoder(h, N, greedy=greedy)
        return assignment, log_probs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pomo_assignment_to_sequence(assignment, instances):
    """Convert POMO flat assignment to per-robot sequence (greedy nearest ordering)."""
    B = len(instances)
    N = instances[0]["N"]
    M = instances[0]["M"]
    results = []
    for b in range(B):
        seq = {i: [] for i in range(N)}
        a = assignment[b].cpu().tolist() if torch.is_tensor(assignment[b]) else assignment[b]
        for j in range(M):
            rid = a[j]
            if 0 <= rid < N:
                seq[rid].append(j)
        # order tasks within each robot by nearest-neighbor
        for i in range(N):
            if len(seq[i]) <= 1:
                continue
            tasks = instances[b]["tasks"]
            pos = instances[b]["robots"][i]["pos"].copy()
            ordered = []
            remaining = list(seq[i])
            for _ in range(len(remaining)):
                best_j, best_d = remaining[0], float("inf")
                for j in remaining:
                    d = np.sqrt((pos[0]-tasks[j]["pick"][0])**2 + (pos[1]-tasks[j]["pick"][1])**2)
                    if d < best_d:
                        best_d = d
                        best_j = j
                ordered.append(best_j)
                remaining.remove(best_j)
                pos = tasks[best_j]["drop"]
            seq[i] = ordered
        results.append(seq)
    return results

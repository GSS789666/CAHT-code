"""
CAHT V2 — Full model assembly.
Embedding → Spatial-Bias Encoder → Assignment Decoder → Sequencing Decoder
"""
import torch
import torch.nn as nn
import numpy as np
from config import D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT
from .embeddings import HeteroEmbedding
from .encoder import SpatialBiasEncoder
from .assign_decoder import AssignmentDecoder
from .seq_decoder import SequencingDecoder


class CAHT(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = HeteroEmbedding(d_model)
        self.encoder   = SpatialBiasEncoder(d_model, n_heads, n_layers, ffn_dim, dropout)
        self.assign_dec = AssignmentDecoder(d_model)
        self.seq_dec    = SequencingDecoder(d_model)

    def _build_features(self, instances, device):
        """
        Convert list of instance dicts to tensors.
        Returns: robot_feat, task_feat, robot_types, dist_matrix, robots_data, tasks_data
        """
        B = len(instances)
        N = instances[0]["N"]
        M = instances[0]["M"]
        S = N + M

        robot_feat  = torch.zeros(B, N, 5, device=device)
        task_feat   = torch.zeros(B, M, 8, device=device)
        robot_types = torch.zeros(B, N, dtype=torch.long, device=device)

        robots_data = []
        tasks_data  = []

        for b, inst in enumerate(instances):
            rd = []
            for i, r in enumerate(inst["robots"]):
                robot_feat[b, i] = torch.tensor([
                    r["pos"][0], r["pos"][1],
                    r["battery"], r["vel"], r["cap"]
                ])
                robot_types[b, i] = r["type"]
                rd.append(r)
            robots_data.append(rd)

            td = []
            for j, t in enumerate(inst["tasks"]):
                task_feat[b, j] = torch.tensor([
                    t["pick"][0], t["pick"][1],
                    t["drop"][0], t["drop"][1],
                    t["priority"], t["tw_early"], t["tw_late"], t["weight"]
                ])
                td.append(t)
            tasks_data.append(td)

        # pairwise distance matrix
        positions = torch.zeros(B, S, 2, device=device)
        for b, inst in enumerate(instances):
            for i, r in enumerate(inst["robots"]):
                positions[b, i, 0] = float(r["pos"][0])
                positions[b, i, 1] = float(r["pos"][1])
            for j, t in enumerate(inst["tasks"]):
                positions[b, N+j, 0] = float(t["pick"][0])
                positions[b, N+j, 1] = float(t["pick"][1])

        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist_matrix = torch.norm(diff, dim=-1)

        return robot_feat, task_feat, robot_types, dist_matrix, robots_data, tasks_data

    def forward(self, instances, greedy=True, return_log_probs=False):
        """
        instances: list of instance dicts (batch)
        Returns: assignments, sequences, (log_probs_assign, log_probs_seq)
        """
        device = next(self.parameters()).device
        N = instances[0]["N"]

        robot_feat, task_feat, robot_types, dist_matrix, robots_data, tasks_data = \
            self._build_features(instances, device)

        H0 = self.embedding(robot_feat, task_feat, robot_types)
        H = self.encoder(H0, dist_matrix)

        h_robots = H[:, :N, :]
        h_tasks  = H[:, N:, :]

        assignment, lp_assign = self.assign_dec(
            h_robots, h_tasks, robots_data, tasks_data,
            greedy=greedy, return_log_probs=True)

        sequences, lp_seq = self.seq_dec(
            H, assignment, N, tasks_data=tasks_data,
            greedy=greedy, return_log_probs=True)

        if return_log_probs:
            return assignment, sequences, lp_assign, lp_seq
        return assignment, sequences

    def compute_sl_loss(self, instances, target_assignments, target_sequences):
        """
        Compute supervised learning loss using ALNS labels.
        target_assignments: list of lists [B][j] = robot_id
        target_sequences: list of dicts [B]{robot_id: [task_ids in order]}
        Returns: total_loss, assign_loss, seq_loss
        """
        device = next(self.parameters()).device
        N = instances[0]["N"]

        robot_feat, task_feat, robot_types, dist_matrix, robots_data, tasks_data = \
            self._build_features(instances, device)

        H0 = self.embedding(robot_feat, task_feat, robot_types)
        H = self.encoder(H0, dist_matrix)

        h_robots = H[:, :N, :]
        h_tasks  = H[:, N:, :]

        # Assignment loss (teacher-forced CE)
        assign_loss = self.assign_dec.supervised_loss(
            h_robots, h_tasks, robots_data, tasks_data, target_assignments)

        # Sequence loss (teacher-forced NLL)
        seq_loss = self.seq_dec.supervised_loss(
            H, target_assignments, target_sequences, N, tasks_data=tasks_data)

        return assign_loss, seq_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

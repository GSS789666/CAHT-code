"""
CAHT V2 — Heterogeneous input embedding layer.
Robot: Linear(5→d) + type_emb + kin_emb
Task:  Linear(8→d) + type_emb
"""
import torch
import torch.nn as nn
from config import D_MODEL


class HeteroEmbedding(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d = d_model
        self.robot_proj = nn.Linear(5, d_model)
        self.task_proj  = nn.Linear(8, d_model)

        # type embedding: 0=robot, 1=task
        self.type_emb = nn.Embedding(2, d_model)
        # kinematic embedding: 0=AGV, 1=AMR, 2=Forklift
        self.kin_emb  = nn.Embedding(3, d_model)

    def forward(self, robot_feat, task_feat, robot_types):
        """
        robot_feat: (B, N, 5)  — [pos_x, pos_y, battery, velocity, capacity]
        task_feat:  (B, M, 8)  — [pick_x, pick_y, drop_x, drop_y, priority, tw_e, tw_l, weight]
        robot_types: (B, N) LongTensor — kinematic type ids
        Returns: H0 (B, N+M, d)
        """
        B, N, _ = robot_feat.shape
        M = task_feat.shape[1]

        # Normalize features to ~[0,1] range for stable training
        robot_scale = robot_feat.new_tensor([100., 100., 1000., 5., 20.])
        task_scale = task_feat.new_tensor([100., 100., 100., 100., 3., 3000., 3000., 20.])
        robot_feat = robot_feat / robot_scale
        task_feat = task_feat / task_scale

        h_r = self.robot_proj(robot_feat)                       # (B,N,d)
        h_r = h_r + self.type_emb(torch.zeros(B, N, dtype=torch.long, device=h_r.device))
        h_r = h_r + self.kin_emb(robot_types)                   # (B,N,d)

        h_t = self.task_proj(task_feat)                          # (B,M,d)
        h_t = h_t + self.type_emb(torch.ones(B, M, dtype=torch.long, device=h_t.device))

        H0 = torch.cat([h_r, h_t], dim=1)                       # (B,N+M,d)
        return H0

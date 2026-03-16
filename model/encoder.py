"""
CAHT V2 — Spatial-Bias Transformer Encoder.
a_{ij} = QK^T/√d_k + g_{ij}
g_{ij} = MLP(||pos_i - pos_j||)
"""
import torch
import torch.nn as nn
import math
from config import D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT


class SpatialBiasMLP(nn.Module):
    """g_{ij} = MLP(dist) : scalar bias per head."""
    def __init__(self, n_heads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, n_heads),
        )

    def forward(self, dist_matrix):
        """dist_matrix: (B, S, S) → bias: (B, heads, S, S)"""
        B, S, _ = dist_matrix.shape
        g = self.net(dist_matrix.unsqueeze(-1))  # (B,S,S,heads)
        return g.permute(0, 3, 1, 2)              # (B,heads,S,S)


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.spatial_mlp = SpatialBiasMLP(n_heads)

    def forward(self, x, dist_matrix):
        B, S, D = x.shape
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        g = self.spatial_mlp(dist_matrix)
        scores = scores + g

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, V)  # (B,heads,S,d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.attn = SpatialMultiHeadAttention(d_model, n_heads, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dist_matrix):
        x = x + self.drop(self.attn(self.ln1(x), dist_matrix))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class SpatialBiasEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, H0, dist_matrix):
        """
        H0: (B, S, d)
        dist_matrix: (B, S, S) — pairwise Euclidean distances
        """
        x = H0
        for layer in self.layers:
            x = layer(x, dist_matrix)
        return x

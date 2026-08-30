"""
src/models/gatv2_model.py
Graph Attention Networks for node-level classification (each node = tender)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, GATConv


class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3, edge_dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads, dropout=dropout, concat=True)
        self.norm = GraphNorm(out_dim * heads)
        self.residual = nn.Linear(in_dim, out_dim * heads) if in_dim != out_dim * heads else nn.Identity()
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index, edge_weight=None):
        # ละเว้น edge_weight
        if self.training and self.edge_dropout > 0:
            mask = torch.rand(edge_index.size(1), device=x.device) > self.edge_dropout
            edge_index = edge_index[:, mask]
            # edge_weight ไม่ถูกใช้
        out = self.gat(x, edge_index)  # ไม่ส่ง edge_weight
        out = self.norm(out)
        out = out + self.residual(x)
        return F.elu(out)


class GATv2Model(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, num_layers=2, heads=4, dropout=0.3, edge_dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Layer(in_dim, hidden_dim, heads, dropout, edge_dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Layer(hidden_dim * heads, hidden_dim, heads, dropout, edge_dropout))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        logits = self.classifier(x)   # (num_nodes, out_dim)
        return logits


class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, num_layers=2, heads=4, dropout=0.3, edge_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        logits = self.classifier(x)
        return logits


def create_model(model_type='gatv2', in_dim=7, **kwargs):
    if model_type == 'simple_gat':
        return SimpleGAT(in_dim=in_dim, **kwargs)
    elif model_type == 'gatv2':
        return GATv2Model(in_dim=in_dim, **kwargs)
    elif model_type == 'hybrid':
        return GATv2Model(in_dim=in_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
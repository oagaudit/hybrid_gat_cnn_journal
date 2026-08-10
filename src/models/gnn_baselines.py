"""
src/models/gnn_baselines.py  --  NEW FILE

Extra baselines. The current ablation only contrasts GATv1 against GATv2, which
does not rule out the simpler explanation that any message passing would do just
as well, nor the even simpler one that the graph adds nothing over an MLP on the
same node features. Reviewers ask for exactly these two controls.

  GCNBaseline       - fixed degree-normalised aggregation, no attention.
  GraphSAGEBaseline - learned neighbour aggregation, no attention.
  MLPBaseline       - identical feature input, edges deleted entirely.

MLPBaseline is the most important of the three: if it matches the hybrid model,
the graph component is not carrying its weight and the paper should say so.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphNorm


class GCNBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, num_layers=2,
                 dropout=0.3, use_norm=True, **_):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.norms.append(GraphNorm(dims[i + 1]) if use_norm else nn.Identity())
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GraphSAGEBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, num_layers=2,
                 dropout=0.3, use_norm=True, **_):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.norms.append(GraphNorm(dims[i + 1]) if use_norm else nn.Identity())
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class MLPBaseline(nn.Module):
    """Same node features, no edges. Isolates the contribution of the graph."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, num_layers=2,
                 dropout=0.3, **_):
        super().__init__()
        layers, d = [], in_dim
        for _i in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(64, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.net(x)


BASELINE_REGISTRY = {'gcn': GCNBaseline, 'sage': GraphSAGEBaseline, 'mlp': MLPBaseline}


def create_gnn_baseline(model_type, in_dim, **kwargs):
    if model_type not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{model_type}'. Options: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[model_type](in_dim=in_dim, **kwargs)

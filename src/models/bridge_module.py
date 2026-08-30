"""
src/models/bridge_module.py

WHAT CHANGED AND WHY
--------------------
In the original pipeline, `ContextualBridgeModule` was instantiated with random
weights inside `prepare_pair_sets.py`, put in `.eval()` mode, and run under
`torch.no_grad()`. It was never optimised and its weights were never saved.
The 128-dim "visual embedding" was therefore a *fixed random projection* of the
CNN pair embeddings, and the query built from the statistical screens could not
have learned to up-weight statistically anomalous pairs. No random seed was set
either, so the node features were not reproducible across re-runs.

This version makes the bridge a first-class trainable component:

  * `ContextualBridgeModule` now consumes a RAGGED batch of pair embeddings
    (flat tensor + node index) so it can run inside the Stage-2 forward pass
    over the whole graph, and receives gradients from the classification loss.
  * It optionally returns the per-pair attention weights, which is what makes
    the interpretability analysis (which bidder pair drove the flag?) possible.
  * Tenders with no observed pairs now yield an exact zero vector instead of a
    learned bias vector, so "no visual evidence" is represented as "no signal"
    rather than as a constant that the network can key on.

The legacy `BridgeModule` (fixed learnable query, no screen conditioning) is
kept only as an ablation arm: it isolates the contribution of *screen-guided*
querying versus generic attention pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.utils import softmax as scatter_softmax
    _HAS_PYG = True
except ImportError:  # pragma: no cover - fallback so the file imports standalone
    _HAS_PYG = False


def _segment_softmax(scores, index, num_segments):
    """Numerically stable softmax over variable-length segments."""
    if _HAS_PYG:
        return scatter_softmax(scores, index, num_nodes=num_segments)
    seg_max = scores.new_full((num_segments,), float('-inf'))
    seg_max = seg_max.scatter_reduce(0, index, scores, reduce='amax',
                                     include_self=True)
    scores = scores - seg_max[index]
    exp = scores.exp()
    denom = torch.zeros(num_segments, device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(0, index, exp)
    return exp / (denom[index] + 1e-16)


class ContextualBridgeModule(nn.Module):
    """Screen-guided attention pooling from bidder-pair level to tender level.

    Args:
        pair_embed_dim: dimension of each CNN pair embedding (64).
        screen_dim: number of statistical screens used to build the query (7).
        visual_embed_dim: dimension of the pooled tender-level visual vector (128).
        hidden_dim: dimension of the query/key space (64).
    """

    def __init__(self, pair_embed_dim=64, screen_dim=7, visual_embed_dim=128,
                 hidden_dim=64):
        super().__init__()
        self.visual_embed_dim = visual_embed_dim
        self.hidden_dim = hidden_dim

        self.screen_proj = nn.Linear(screen_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(pair_embed_dim, hidden_dim)
        self.value_proj = nn.Linear(pair_embed_dim, visual_embed_dim)

    def forward(self, pair_emb, pair_node_idx, screens, num_nodes=None,
                return_attention=False):
        """
        Args:
            pair_emb: (P, pair_embed_dim) flat tensor of all pair embeddings in
                the graph, concatenated across tenders.
            pair_node_idx: (P,) long tensor; entry p gives the tender-node index
                that pair p belongs to.
            screens: (N, screen_dim) tender-level statistical screens.
            num_nodes: N. Inferred from `screens` when omitted.
            return_attention: if True, also return the (P,) attention weights.

        Returns:
            visual: (N, visual_embed_dim). Rows for tenders with no observed
                pairs are exactly zero.
            attn (optional): (P,) attention weight of each pair within its tender.
        """
        if num_nodes is None:
            num_nodes = screens.size(0)

        context = F.relu(self.screen_proj(screens))          # (N, H)
        query = self.query_proj(context)                     # (N, H)

        visual = pair_emb.new_zeros(num_nodes, self.visual_embed_dim)
        if pair_emb.numel() == 0:
            return (visual, pair_emb.new_zeros(0)) if return_attention else visual

        keys = self.key_proj(pair_emb)                       # (P, H)
        values = self.value_proj(pair_emb)                   # (P, V)

        scores = (query[pair_node_idx] * keys).sum(dim=-1)
        scores = scores / (self.hidden_dim ** 0.5)           # (P,)
        attn = _segment_softmax(scores, pair_node_idx, num_nodes)

        visual = visual.index_add(0, pair_node_idx, attn.unsqueeze(-1) * values)

        if return_attention:
            return visual, attn
        return visual


class BridgeModule(nn.Module):
    """Ablation arm: attention pooling with a learnable but *unconditioned*
    query. Comparing this against ContextualBridgeModule isolates how much the
    statistical screens contribute as a query signal, separately from how much
    attention pooling contributes over mean pooling."""

    def __init__(self, pair_embed_dim=64, visual_embed_dim=128, hidden_dim=64):
        super().__init__()
        self.visual_embed_dim = visual_embed_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.key_proj = nn.Linear(pair_embed_dim, hidden_dim)
        self.value_proj = nn.Linear(pair_embed_dim, visual_embed_dim)

    def forward(self, pair_emb, pair_node_idx, screens=None, num_nodes=None,
                return_attention=False):
        if num_nodes is None:
            num_nodes = int(pair_node_idx.max().item()) + 1 if pair_node_idx.numel() else 0

        visual = pair_emb.new_zeros(num_nodes, self.visual_embed_dim)
        if pair_emb.numel() == 0:
            return (visual, pair_emb.new_zeros(0)) if return_attention else visual

        keys = self.key_proj(pair_emb)
        values = self.value_proj(pair_emb)
        scores = (self.query.expand(keys.size(0), -1) * keys).sum(dim=-1)
        scores = scores / (self.hidden_dim ** 0.5)
        attn = _segment_softmax(scores, pair_node_idx, num_nodes)
        visual = visual.index_add(0, pair_node_idx, attn.unsqueeze(-1) * values)

        if return_attention:
            return visual, attn
        return visual


class MeanPoolBridge(nn.Module):
    """Ablation arm: plain mean pooling, no attention at all.

    The paper argues attention is needed because averaging would "attenuate the
    spatial anomalies of suspicious companies". That claim is only supported if
    mean pooling is actually run as a comparison, which this arm provides."""

    def __init__(self, pair_embed_dim=64, visual_embed_dim=128):
        super().__init__()
        self.visual_embed_dim = visual_embed_dim
        self.value_proj = nn.Linear(pair_embed_dim, visual_embed_dim)

    def forward(self, pair_emb, pair_node_idx, screens=None, num_nodes=None,
                return_attention=False):
        if num_nodes is None:
            num_nodes = int(pair_node_idx.max().item()) + 1 if pair_node_idx.numel() else 0

        visual = pair_emb.new_zeros(num_nodes, self.visual_embed_dim)
        if pair_emb.numel() == 0:
            return (visual, pair_emb.new_zeros(0)) if return_attention else visual

        values = self.value_proj(pair_emb)
        counts = torch.zeros(num_nodes, device=pair_emb.device, dtype=pair_emb.dtype)
        counts = counts.index_add(0, pair_node_idx,
                                  torch.ones_like(pair_node_idx, dtype=pair_emb.dtype))
        visual = visual.index_add(0, pair_node_idx, values)
        visual = visual / counts.clamp(min=1).unsqueeze(-1)

        if return_attention:
            attn = 1.0 / counts.clamp(min=1)[pair_node_idx]
            return visual, attn
        return visual


BRIDGE_REGISTRY = {
    'contextual': ContextualBridgeModule,
    'uncond': BridgeModule,
    'mean': MeanPoolBridge,
}


def create_bridge(kind='contextual', **kwargs):
    if kind not in BRIDGE_REGISTRY:
        raise ValueError(f"Unknown bridge type '{kind}'. "
                         f"Options: {list(BRIDGE_REGISTRY)}")
    cls = BRIDGE_REGISTRY[kind]
    if kind in ('uncond', 'mean'):
        kwargs.pop('screen_dim', None)
    if kind == 'mean':
        kwargs.pop('hidden_dim', None)
    return cls(**kwargs)

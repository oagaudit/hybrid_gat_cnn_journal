"""
src/models/hybrid_model.py

End-to-end hybrid model: trainable Contextual Bridge + GATv2.

The original pipeline pre-computed 135-dim node features offline with an
untrained bridge, then fed those frozen vectors to GATv2. That made the bridge
a fixed random projection. Here the bridge lives inside the model, so the
classification loss flows back through the attention weights and the module
actually learns which bidder pairs to attend to.

The Stage-1 CNN stays frozen exactly as before: this model consumes the
pre-extracted 64-dim pair embeddings as input and never backpropagates into
the CNN. The two-stage design and its leakage guarantees are unchanged.
"""

import torch
import torch.nn as nn

from src.models.bridge_module import create_bridge
from src.models.gatv2_model import GATv2Model, SimpleGAT


class HybridGATv2Model(nn.Module):
    """Trainable bridge -> [visual || screens] -> GATv2 -> tender-level logits.

    Args:
        pair_embed_dim: 64, dimension of frozen CNN pair embeddings.
        screen_dim: 7, statistical screens per tender.
        visual_embed_dim: 128, pooled visual representation.
        bridge_kind: 'contextual' | 'uncond' | 'mean' (see bridge_module).
        normalize_visual: apply LayerNorm to the pooled visual vector before
            concatenating it with the screens. Strongly recommended: the raw
            screens and the visual embedding live on very different scales, and
            without this the graph layers are dominated by whichever block has
            the larger variance.
    """

    def __init__(self, pair_embed_dim=64, screen_dim=7, visual_embed_dim=128,
                 bridge_hidden=64, bridge_kind='contextual',
                 hidden_dim=128, out_dim=2, num_layers=2, heads=4,
                 dropout=0.3, edge_dropout=0.2, normalize_visual=True,
                 gat_variant='gatv2'):
        super().__init__()
        self.bridge = create_bridge(
            bridge_kind,
            pair_embed_dim=pair_embed_dim,
            screen_dim=screen_dim,
            visual_embed_dim=visual_embed_dim,
            hidden_dim=bridge_hidden,
        )
        self.normalize_visual = normalize_visual
        self.visual_norm = nn.LayerNorm(visual_embed_dim) if normalize_visual else nn.Identity()

        in_dim = visual_embed_dim + screen_dim
        gat_cls = GATv2Model if gat_variant == 'gatv2' else SimpleGAT
        gat_kwargs = dict(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                          num_layers=num_layers, heads=heads, dropout=dropout)
        if gat_variant == 'gatv2':
            gat_kwargs['edge_dropout'] = edge_dropout
        self.gat = gat_cls(**gat_kwargs)

    def forward(self, pair_emb, pair_node_idx, screens, edge_index,
                edge_weight=None, return_attention=False):
        out = self.bridge(pair_emb, pair_node_idx, screens,
                          num_nodes=screens.size(0),
                          return_attention=return_attention)
        if return_attention:
            visual, bridge_attn = out
        else:
            visual, bridge_attn = out, None

        visual = self.visual_norm(visual)
        x = torch.cat([visual, screens], dim=1)
        logits = self.gat(x, edge_index, edge_weight)

        if return_attention:
            return logits, bridge_attn, x
        return logits

    @torch.no_grad()
    def node_features(self, pair_emb, pair_node_idx, screens):
        """Return the 135-dim node features the GAT actually sees.

        Useful for making the classical baselines (LR / RF / XGBoost) a fair
        comparison: they should be fed the *learned* features, not the old
        random-projection ones.
        """
        self.eval()
        visual = self.bridge(pair_emb, pair_node_idx, screens,
                             num_nodes=screens.size(0))
        visual = self.visual_norm(visual)
        return torch.cat([visual, screens], dim=1)


def create_hybrid_model(**kwargs):
    return HybridGATv2Model(**kwargs)

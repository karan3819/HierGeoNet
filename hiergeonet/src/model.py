"""
HierGeoNet — Hierarchical Multi-Scale Graph Attention Network
=============================================================
Architecture described in:
  "HierGeoNet: Graph Attention with Cross-Scale Regularisation
   for Landslide Susceptibility"
  Pathania K., Singh S., Sharma A. — Panjab University, Chandigarh

Module layout
-------------
GATLayer          : single GAT layer with skip connection + LayerNorm
GATStack          : n_layers of GATLayer stacked
CrossScaleBridge  : gated top-down context fusion (Equations 10–12)
GeoPositionalEncoding : learnable sinusoidal lat/lon PE (Equations 5–8)
HierGeoNet        : full three-scale hierarchy + risk head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single Graph Attention layer with residual skip connection.

    Implements Equation 2–4 of the paper:
        e_ij  = LeakyReLU(a^T · [Wh_i ‖ Wh_j])
        α_ij  = softmax over neighbours
        h_i'  = LN(GELU(Σ_j α_ij Wh_j) + W_skip·h_i)

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension (must be divisible by heads)
    heads   : number of attention heads (concatenated, not averaged)
    dropout : dropout rate on attention coefficients
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.conv = GATConv(
            in_dim,
            out_dim // heads,   # per-head dim; concat → out_dim
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.conv(x, edge_index)) + self.skip(x))


class GATStack(nn.Module):
    """
    Sequential stack of GATLayer modules.

    All layers share the same output dimension; only the first layer
    projects from in_dim → out_dim.

    Parameters
    ----------
    in_dim   : input feature dimension
    out_dim  : output (and hidden) feature dimension
    heads    : attention heads per layer
    n_layers : number of stacked GAT layers
    dropout  : dropout rate
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        dims = [in_dim] + [out_dim] * n_layers
        self.layers = nn.ModuleList(
            [GATLayer(dims[i], dims[i + 1], heads, dropout) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class CrossScaleBridge(nn.Module):
    """
    Gated top-down context fusion module (Equations 10–12).

    Projects coarse-scale global context into the fine-scale embedding
    space and applies a learned per-dimension gate:

        c̃_i  = W_C · h_coarse[parent(i)]          (Eq. 10)
        g_i   = σ(W_g · [h_fine_i ; c̃_i])         (Eq. 11)
        h_i'  = LN(h_fine_i + g_i ⊙ tanh(W_ctx·c̃_i))  (Eq. 12)

    When g_i → 0 the bridge is a no-op and HierGeoNet reduces to a
    single-scale GNN (Theorem 1 in the paper).

    Parameters
    ----------
    d_fine   : fine-scale embedding dimension
    d_coarse : coarse-scale embedding dimension (projected to d_fine)
    """

    def __init__(self, d_fine: int, d_coarse: int):
        super().__init__()
        self.proj    = nn.Linear(d_coarse, d_fine)
        self.gate    = nn.Linear(d_fine * 2, d_fine)
        self.context = nn.Linear(d_fine, d_fine)
        self.norm    = nn.LayerNorm(d_fine)

    def forward(
        self,
        h_fine: torch.Tensor,
        h_coarse_bcast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_fine          : [N_fine, d_fine]
        h_coarse_bcast  : [N_fine, d_coarse]  (coarse context already
                          broadcast to fine via assignment map indexing)
        """
        hc   = self.proj(h_coarse_bcast)                          # [N, d_fine]
        gate = torch.sigmoid(self.gate(torch.cat([h_fine, hc], dim=-1)))  # [N, d_fine]
        ctx  = torch.tanh(self.context(hc))                       # [N, d_fine]
        return self.norm(h_fine + gate * ctx)


class GeoPositionalEncoding(nn.Module):
    """
    Learnable sinusoidal geographic positional encoding (Equations 5–8).

    Encodes (latitude, longitude) pixel coordinates into a d_model-dim
    vector using sin/cos at multiple frequencies, with a learnable
    per-frequency scale vector λ initialised to 1.

        GP(i, 4k)   = λ_{4k}   · sin(lat_i / τ^{4k/d})
        GP(i, 4k+1) = λ_{4k+1} · cos(lat_i / τ^{4k/d})
        GP(i, 4k+2) = λ_{4k+2} · sin(lon_i / τ^{4k/d})
        GP(i, 4k+3) = λ_{4k+3} · cos(lon_i / τ^{4k/d})

    τ = 10,000 (same as original Transformer paper).
    Fully vectorised — no Python for-loop over k.

    Parameters
    ----------
    d_model : must equal d_coarse (256 in the default config)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))           # λ — learnable
        k   = torch.arange(d_model // 4, dtype=torch.float32)
        div = 10_000.0 ** (2.0 * k / d_model)                   # τ^{2k/d}
        self.register_buffer("div", div)                          # non-trainable

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : [N, 2] — (row, col) pixel coordinates for coarse nodes

        Returns
        -------
        pe : [N, d_model]
        """
        lat = coords[:, 0:1]                                      # [N, 1]
        lon = coords[:, 1:2]                                      # [N, 1]
        d   = self.div.unsqueeze(0)                               # [1, d//4]
        pe  = torch.cat(
            [
                torch.sin(lat / d),
                torch.cos(lat / d),
                torch.sin(lon / d),
                torch.cos(lon / d),
            ],
            dim=-1,
        )                                                          # [N, d_model]
        return pe * self.scale.unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class HierGeoNet(nn.Module):
    """
    HierGeoNet: Hierarchical Multi-Scale Graph Attention Network
    for Landslide Susceptibility Mapping.

    Architecture overview
    ---------------------
    Bottom-up pass (scale-specific GAT stacks):
        Fine   [16384 nodes, 128-d] → H(f)
        Medium [ 1849 nodes, 192-d] → H(m)
        Coarse [  225 nodes, 256-d] → H(c)

    Global context (coarse Transformer):
        H(c) + GeoPositionalEncoding → TransformerEncoder → H_global(c)

    Top-down bridges:
        H_global(c) → CrossScaleBridge → H(m)'
        H(m)'       → CrossScaleBridge → H(f)'     (medium path)
        H_global(c) → CrossScaleBridge → H(f)''    (direct skip)

    Fusion & risk head:
        [H(f)' ‖ H(f)''] → Linear → GELU → 2-layer MLP → ŷ ∈ [0,1]^{N_fine}

    Parameters
    ----------
    cfg : Config object (see configs/default.yaml or src/config.py)
    """

    def __init__(self, cfg):
        super().__init__()
        d_f, d_m, d_c = cfg.D_FINE, cfg.D_MED, cfg.D_COARSE

        # Input projections: 14d → scale-specific embedding space
        self.proj_fine   = nn.Sequential(
            nn.Linear(cfg.IN_DIM, d_f), nn.LayerNorm(d_f), nn.GELU()
        )
        self.proj_med    = nn.Sequential(
            nn.Linear(cfg.IN_DIM, d_m), nn.LayerNorm(d_m), nn.GELU()
        )
        self.proj_coarse = nn.Sequential(
            nn.Linear(cfg.IN_DIM, d_c), nn.LayerNorm(d_c), nn.GELU()
        )

        # Scale-specific GAT stacks
        self.gat_fine   = GATStack(d_f, d_f, cfg.GAT_HEADS, cfg.GAT_LAYERS, cfg.DROPOUT)
        self.gat_med    = GATStack(d_m, d_m, cfg.GAT_HEADS, cfg.GAT_LAYERS, cfg.DROPOUT)
        self.gat_coarse = GATStack(d_c, d_c, cfg.GAT_HEADS, cfg.GAT_LAYERS, cfg.DROPOUT)

        # Geographic positional encoding
        self.geo_pe = GeoPositionalEncoding(d_c)

        # Coarse Transformer (full O(N²) attention — only 225 nodes)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_c,
            nhead=cfg.TF_HEADS,
            dim_feedforward=cfg.FF_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.TF_LAYERS,
            enable_nested_tensor=False,
        )

        # Cross-scale bridge modules
        self.bridge_c2m = CrossScaleBridge(d_m, d_c)   # coarse → medium
        self.bridge_m2f = CrossScaleBridge(d_f, d_m)   # medium → fine
        self.bridge_c2f = CrossScaleBridge(d_f, d_c)   # coarse → fine (skip)

        # Fusion: merge two fine streams
        self.fusion = nn.Sequential(
            nn.Linear(d_f * 2, d_f), nn.LayerNorm(d_f), nn.GELU()
        )

        # Risk head: 2-layer MLP → scalar per node
        self.head = nn.Sequential(
            nn.Linear(d_f, 64),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all linear layers; zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_f:      torch.Tensor,   # [N_fine,   IN_DIM]
        ei_f:     torch.Tensor,   # [2, E_fine]  edge index fine
        x_m:      torch.Tensor,   # [N_med,    IN_DIM]
        ei_m:     torch.Tensor,   # [2, E_med]
        x_c:      torch.Tensor,   # [N_coarse, IN_DIM]
        ei_c:     torch.Tensor,   # [2, E_coarse]
        coords_c: torch.Tensor,   # [N_coarse, 2]  (row, col) for geo-PE
        f2m:      torch.Tensor,   # [N_fine]   fine→medium assignment
        m2c:      torch.Tensor,   # [N_med]    medium→coarse assignment
        f2c:      torch.Tensor,   # [N_fine]   fine→coarse assignment (skip)
        return_embeddings: bool = False,
    ):
        """
        Single-patch forward pass.

        Returns
        -------
        logits : [N_fine]  raw (pre-sigmoid) risk scores
        h_final : [N_fine, D_FINE]  returned only if return_embeddings=True
        """
        # ── Bottom-up: scale-specific GAT ────────────────────────────────────
        h_f = self.gat_fine  (self.proj_fine  (x_f), ei_f)   # [16384, 128]
        h_m = self.gat_med   (self.proj_med   (x_m), ei_m)   # [ 1849, 192]
        h_c = self.gat_coarse(self.proj_coarse(x_c), ei_c)   # [  225, 256]

        # ── Coarse Transformer: full global self-attention ────────────────────
        pe    = self.geo_pe(coords_c)                          # [225, 256]
        h_cg  = self.transformer(
            (h_c + pe).unsqueeze(0)
        ).squeeze(0)                                           # [225, 256]

        # ── Top-down bridges: inject global context ───────────────────────────
        h_m2 = self.bridge_c2m(h_m, h_cg[m2c])     # coarse→medium
        h_f2 = self.bridge_m2f(h_f, h_m2[f2m])     # medium→fine   (main path)
        h_fs = self.bridge_c2f(h_f, h_cg[f2c])     # coarse→fine   (skip path)

        # ── Fusion & risk head ────────────────────────────────────────────────
        h_final = self.fusion(torch.cat([h_f2, h_fs], dim=-1))  # [16384, 128]
        logits  = self.head(h_final).squeeze(-1)                 # [16384]

        if return_embeddings:
            return logits, h_final
        return logits


def _unwrap(model: nn.Module) -> HierGeoNet:
    """Strip nn.DataParallel wrapper if present."""
    return model.module if isinstance(model, nn.DataParallel) else model

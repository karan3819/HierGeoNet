"""
HierGeoNet — Multi-Scale Graph Topology Builder
================================================
Builds the three spatial graphs (fine / medium / coarse) and the
cross-scale nearest-neighbour assignment maps.

All tensors are computed once on CPU and then pinned to the target
device.  They never change across patches because the 128×128 grid
geometry is fixed for all L4S patches.

Key outputs
-----------
ei_f, ei_m, ei_c     : edge index tensors [2, E_s] for each scale
f2m_t, m2c_t, f2c_t  : assignment-map tensors [N_source] (long)
coords_f_t, coords_c_t : node coordinate tensors [N, 2] (float32)
gpu_scatter_mean       : utility function for on-GPU feature pooling
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_edges(
    coords: np.ndarray,
    prox_dist: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build undirected KNN edges for a node set defined by `coords`.

    Gaussian edge weights:
        w_ij = exp(-d_ij² / (2σ²))   where σ = prox_dist / 2

    Parameters
    ----------
    coords    : (N, 2) float32 array of (row, col) node coordinates
    prox_dist : δ_s — maximum connection distance (pixels)

    Returns
    -------
    edge_index : (2, 2E) long tensor — both directions included
    edge_weight: (2E,)  float32 tensor — Gaussian weights
    """
    tree  = cKDTree(coords)
    pairs = tree.query_pairs(r=prox_dist, output_type="ndarray")

    if len(pairs) == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0,      dtype=torch.float32),
        )

    ii, jj  = pairs[:, 0], pairs[:, 1]
    dists   = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    sigma2  = (prox_dist * 0.5) ** 2
    weights = np.exp(-dists ** 2 / (2 * sigma2))

    # undirected → add both directions
    src = np.concatenate([ii, jj])
    dst = np.concatenate([jj, ii])
    wts = np.concatenate([weights, weights])

    return (
        torch.tensor(np.stack([src, dst]), dtype=torch.long),
        torch.tensor(wts, dtype=torch.float32),
    )


def build_graph_topology(
    H: int = 128,
    W: int = 128,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Build the complete three-scale graph topology for an H×W patch.

    Parameters
    ----------
    H, W   : patch height and width in pixels (128 for L4S)
    device : target device; all tensors are moved here

    Returns
    -------
    A dict with keys:
        ei_f, ew_f     : fine edge index / weights
        ei_m, ew_m     : medium edge index / weights
        ei_c, ew_c     : coarse edge index / weights
        f2m_t          : [N_fine]  fine → medium assignment
        m2c_t          : [N_med]   medium → coarse assignment
        f2c_t          : [N_fine]  fine → coarse assignment (skip)
        coords_f_t     : [N_fine,   2] fine coordinates
        coords_c_t     : [N_coarse, 2] coarse coordinates
        N_fine, N_med, N_coarse : node counts
    """
    print(f"🏗️  Building graph topology for {H}×{W} patches...")

    # ── Node coordinate grids ──────────────────────────────────────────────────
    def _grid(stride):
        y, x = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
        return np.column_stack([y.ravel(), x.ravel()]).astype(np.float32)

    coords_fine   = _grid(1)   # 128×128 = 16,384 nodes
    coords_med    = _grid(3)   # 43×43  =  1,849 nodes
    coords_coarse = _grid(9)   # 15×15  =    225 nodes

    N_fine, N_med, N_coarse = len(coords_fine), len(coords_med), len(coords_coarse)
    print(f"   Nodes — Fine: {N_fine:,} | Med: {N_med:,} | Coarse: {N_coarse}")

    # ── Edges (Gaussian-weighted KNN) ─────────────────────────────────────────
    ei_f, ew_f = _build_edges(coords_fine,   prox_dist=1.5)
    ei_m, ew_m = _build_edges(coords_med,    prox_dist=4.5)
    ei_c, ew_c = _build_edges(coords_coarse, prox_dist=13.5)
    print(f"   Edges — Fine: {ei_f.shape[1]:,} | Med: {ei_m.shape[1]:,} | Coarse: {ei_c.shape[1]:,}")

    # ── Cross-scale assignment maps (nearest-neighbour) ────────────────────────
    _, fine2med    = cKDTree(coords_med).query(coords_fine,    k=1)
    _, med2coarse  = cKDTree(coords_coarse).query(coords_med, k=1)
    _, fine2coarse = cKDTree(coords_coarse).query(coords_fine, k=1)

    # ── Move everything to device ──────────────────────────────────────────────
    def _t(arr, dtype):
        return torch.tensor(arr, dtype=dtype).to(device)

    topo = dict(
        ei_f=ei_f.to(device),  ew_f=ew_f.to(device),
        ei_m=ei_m.to(device),  ew_m=ew_m.to(device),
        ei_c=ei_c.to(device),  ew_c=ew_c.to(device),
        f2m_t      = _t(fine2med,    torch.long),
        m2c_t      = _t(med2coarse,  torch.long),
        f2c_t      = _t(fine2coarse, torch.long),
        coords_f_t = _t(coords_fine,   torch.float32),
        coords_c_t = _t(coords_coarse, torch.float32),
        N_fine=N_fine, N_med=N_med, N_coarse=N_coarse,
    )

    print(f"✅ Graph topology ready and pinned to {device}")
    return topo


# ──────────────────────────────────────────────────────────────────────────────
# GPU scatter mean (on-the-fly feature pooling)
# ──────────────────────────────────────────────────────────────────────────────

def gpu_scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    Compute the mean of `src` rows grouped by `index`.

    Equivalent to:
        out[i] = mean(src[j] for j where index[j] == i)

    Fully GPU-side — no CPU round-trip, no Python loop.
    Used to derive medium/coarse node features from fine features.

    Parameters
    ----------
    src      : [N_src, D]  source feature matrix
    index    : [N_src]     integer group index for each source row
    dim_size : number of target nodes (N_target)

    Returns
    -------
    out : [N_target, D]
    """
    D   = src.shape[1]
    idx = index.unsqueeze(1).expand(-1, D)
    s   = torch.zeros(dim_size, D, dtype=src.dtype, device=src.device)
    c   = torch.zeros(dim_size, D, dtype=src.dtype, device=src.device)
    s.scatter_add_(0, idx, src)
    c.scatter_add_(0, idx, torch.ones_like(src))
    return s / c.clamp(min=1)

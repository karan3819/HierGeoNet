"""
HierGeoNet — Loss Functions
============================
Implements the two-part training objective (Section IV-E, IV-F):

    L = L_BCE + λ · L_C

where:
    L_BCE = weighted binary cross-entropy (Equation 15)
    L_C   = geographic contrastive regulariser (Equation 16)
    λ     = 0.1 (LAMBDA_CONTRAST in Config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_geo_loss(
    embeddings: torch.Tensor,
    coords:     torch.Tensor,
    y:          torch.Tensor,
    radius:     float = 5.0,
    margin:     float = 2.0,
    n_pairs:    int   = 2048,
) -> torch.Tensor:
    """
    Geographic contrastive regulariser (Equation 16).

    Positive pairs  P+ = {(i,j) : geo_dist(i,j) < r  AND  y_i = y_j}
    Negative pairs  P- = {(i,j) : geo_dist(i,j) > 2r  OR  y_i ≠ y_j}

    Loss:
        L_C = (1/|P+|) Σ_{P+} ‖z_i−z_j‖²
            + (1/|P-|) Σ_{P-} max(0, m − ‖z_i−z_j‖)²

    Proposition 1 (Lipschitz bound): under this loss,
        ‖z_i − z_j‖ ≤ √(L_C) · ⌈geo_dist(i,j) / r⌉
    ensuring the embedding map is Lipschitz-continuous w.r.t. geography.

    Parameters
    ----------
    embeddings : [N, D]  fine-scale embeddings (H_final from forward pass)
    coords     : [N, 2]  (row, col) pixel coordinates for each node
    y          : [N]     binary labels (0/1)
    radius     : r in pixels — positive pair threshold
    margin     : m — minimum separation for negative pairs
    n_pairs    : number of nodes to sub-sample per step (efficiency)

    Returns
    -------
    scalar loss tensor
    """
    N   = embeddings.size(0)
    idx = torch.randperm(N, device=embeddings.device)[:min(n_pairs * 2, N)]
    emb, crd, ys = embeddings[idx], coords[idx], y[idx]

    if emb.size(0) < 4:
        return emb.new_tensor(0.0)

    # Pairwise distances in embedding space and geographic space
    ed = torch.cdist(emb, emb.detach(), p=2.0)   # [n_sub, n_sub]
    gd = torch.cdist(crd, crd,          p=2.0)   # [n_sub, n_sub]

    same     = (ys.unsqueeze(1) == ys.unsqueeze(0))   # [n_sub, n_sub] bool
    pos_mask = same  & (gd <  radius)
    neg_mask = (~same) | (gd > 2 * radius)
    pos_mask.fill_diagonal_(False)
    neg_mask.fill_diagonal_(False)

    lp = (ed[pos_mask] ** 2).mean()                       if pos_mask.any() else emb.new_tensor(0.0)
    ln = (F.relu(margin - ed[neg_mask]) ** 2).mean()      if neg_mask.any() else emb.new_tensor(0.0)

    return lp + ln


def build_criterion(pos_weight: float, device: torch.device) -> nn.BCEWithLogitsLoss:
    """
    Weighted BCE loss (Equation 15).

    pos_weight = w+ = 3.5 upweights the rare landslide class (~58:1 ratio).
    BCEWithLogitsLoss is numerically stable (fused sigmoid + log).

    Parameters
    ----------
    pos_weight : weight applied to positive (landslide) pixels
    device     : target device for the weight tensor

    Returns
    -------
    nn.BCEWithLogitsLoss instance
    """
    pw = torch.tensor([pos_weight], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)

"""
HierGeoNet — Dataset & DataLoader utilities
============================================
Handles the Landslide4Sense (L4S) HDF5 dataset format.

Dataset structure expected on disk
-----------------------------------
<BASE_DATA_DIR>/
    TrainData/
        img/   image_*.h5   (14-channel, 128×128)
        mask/  mask_*.h5    (binary, 128×128)
    ValidData/
        img/   image_*.h5
        mask/  mask_*.h5

Download L4S from: https://www.research-collection.ethz.ch/handle/20.500.11850/510392
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")


class L4SDataset(Dataset):
    """
    PyTorch Dataset for the Landslide4Sense benchmark.

    Each item returns:
        img  : torch.FloatTensor of shape (N_fine, IN_DIM) = (16384, 14)
        mask : torch.FloatTensor of shape (N_fine,)        = (16384,)

    Both tensors are on CPU; the DataLoader / training loop transfers
    them to GPU with non_blocking=True.

    Parameters
    ----------
    data_dir   : base directory containing TrainData/ and ValidData/
    split      : 'TrainData' or 'ValidData'
    norm_stats : optional (mean, std) tuple of shape [(14,), (14,)]
                 if None, per-patch z-score normalisation is used
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.img_dir   = Path(data_dir) / split / "img"
        self.mask_dir  = Path(data_dir) / split / "mask"
        self.files     = sorted(self.img_dir.glob("*.h5"))
        self.norm_stats = norm_stats

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .h5 files found in {self.img_dir}.\n"
                "Check that BASE_DATA_DIR points to the L4S root and that "
                "the folder structure is TrainData/img/*.h5 and TrainData/mask/*.h5."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fp      = self.files[idx]
        mask_fp = self.mask_dir / fp.name.replace("image", "mask")

        # Load from HDF5
        with h5py.File(fp, "r") as f:
            img = f["img"][:]          # (14, 128, 128) or (128, 128, 14)
        with h5py.File(mask_fp, "r") as f:
            mask = f["mask"][:]        # (128, 128) binary

        # Ensure channel-last layout → flatten to (N_fine, 14)
        if img.ndim == 3 and img.shape[0] == 14:
            img = img.transpose(1, 2, 0)   # CHW → HWC
        img  = img.reshape(-1, 14).astype(np.float32)   # (16384, 14)
        mask = mask.ravel().astype(np.float32)           # (16384,)

        img  = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        # Replace NaN / Inf before normalisation
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            img = (img - mean) / std
        else:
            img = (img - img.mean(0)) / (img.std(0) + 1e-8)

        return img, mask


def compute_norm_stats(
    data_dir: str | Path,
    n_sample: int = 200,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute global channel-wise mean and std from a random subset of
    training patches.  Call once and cache the result.

    Parameters
    ----------
    data_dir : base directory containing TrainData/
    n_sample : number of patches to sample (200 is sufficient)
    seed     : random seed for reproducibility

    Returns
    -------
    mean : (14,) float32 tensor
    std  : (14,) float32 tensor (clamped to ≥ 1e-8)
    """
    rng     = np.random.default_rng(seed)
    raw_ds  = L4SDataset(data_dir, "TrainData", norm_stats=None)
    n       = min(n_sample, len(raw_ds))
    idxs    = rng.choice(len(raw_ds), size=n, replace=False)

    imgs = torch.cat([raw_ds[int(i)][0] for i in idxs], dim=0)  # (n×16384, 14)
    mean = imgs.mean(0)
    std  = imgs.std(0).clamp(min=1e-8)
    return mean, std


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    n_norm_sample: int = 200,
) -> Tuple[DataLoader, DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convenience function: compute norm stats, build train/val loaders.

    Returns
    -------
    train_loader, val_loader, (norm_mean, norm_std)
    """
    print("📊 Computing global normalisation stats...")
    norm_stats = compute_norm_stats(data_dir, n_sample=n_norm_sample)

    train_ds = L4SDataset(data_dir, "TrainData", norm_stats=norm_stats)
    val_ds   = L4SDataset(data_dir, "ValidData", norm_stats=norm_stats)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    print(f"✅ Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    print(f"   Batches — train: {len(train_loader)} | val: {len(val_loader)}")

    return train_loader, val_loader, norm_stats

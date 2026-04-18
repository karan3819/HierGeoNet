#!/usr/bin/env python3
"""
HierGeoNet — Standalone Training Script
========================================
Usage:
    python scripts/train.py --data_dir /path/to/l4sdataset
    python scripts/train.py --data_dir /path/to/l4sdataset --epochs 100 --batch_size 4

For full argument list:
    python scripts/train.py --help
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model   import HierGeoNet, _unwrap
from src.dataset import build_dataloaders
from src.graph   import build_graph_topology, gpu_scatter_mean
from src.losses  import build_criterion, contrastive_geo_loss


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Input
    IN_DIM   = 14
    # Model dims
    D_FINE   = 128
    D_MED    = 192
    D_COARSE = 256
    GAT_HEADS  = 4
    GAT_LAYERS = 2
    TF_HEADS   = 8
    TF_LAYERS  = 3
    FF_DIM     = 512
    DROPOUT    = 0.15
    # Contrastive
    LAMBDA_CONTRAST     = 0.1
    CONTRAST_MARGIN     = 2.0
    CONTRAST_RADIUS_PIX = 5.0
    # Training
    EPOCHS        = 300
    BATCH_SIZE    = 8
    LR            = 5e-4
    WEIGHT_DECAY  = 1e-4
    POS_WEIGHT    = 3.5
    GRAD_CLIP     = 1.0
    PATIENCE      = 40
    WARMUP_EPOCHS = 20
    # DataLoader
    NUM_WORKERS = 0
    PIN_MEMORY  = True
    SEED        = 42


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HierGeoNet on Landslide4Sense")
    p.add_argument("--data_dir",    required=True, help="Path to L4S dataset root")
    p.add_argument("--epochs",      type=int,   default=Config.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=Config.LR)
    p.add_argument("--workers",     type=int,   default=Config.NUM_WORKERS)
    p.add_argument("--seed",        type=int,   default=Config.SEED)
    p.add_argument("--out_dir",     default="outputs")
    p.add_argument("--model_dir",   default="models")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    CFG  = Config()
    CFG.EPOCHS     = args.epochs
    CFG.BATCH_SIZE = args.batch_size
    CFG.LR         = args.lr
    CFG.NUM_WORKERS = args.workers
    CFG.SEED       = args.seed

    # Seeding
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    # Device
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        DEVICE          = torch.device("cuda:0")
        AMP_DEVICE_TYPE = "cuda"
        USE_MULTI_GPU   = torch.cuda.device_count() > 1
        USE_AMP         = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark        = True
        torch.cuda.manual_seed_all(CFG.SEED)
        print(f"🎮 Using {torch.cuda.device_count()} GPU(s)")
    else:
        DEVICE          = torch.device("cpu")
        AMP_DEVICE_TYPE = "cpu"
        USE_MULTI_GPU   = False
        USE_AMP         = False
        CFG.PIN_MEMORY  = False
        print("⚠️  No GPU — running on CPU (slow)")

    # Directories
    MODEL_DIR = Path(args.model_dir);  MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR   = Path(args.out_dir);    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, norm_stats = build_dataloaders(
        args.data_dir,
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )

    # Graph topology (built once, pinned to GPU)
    topo = build_graph_topology(128, 128, DEVICE)
    ei_f, ei_m, ei_c     = topo["ei_f"], topo["ei_m"], topo["ei_c"]
    f2m_t, m2c_t, f2c_t  = topo["f2m_t"], topo["m2c_t"], topo["f2c_t"]
    coords_f_t, coords_c_t = topo["coords_f_t"], topo["coords_c_t"]
    N_med, N_coarse       = topo["N_med"], topo["N_coarse"]

    # Model
    model = HierGeoNet(CFG).to(DEVICE)
    if USE_MULTI_GPU:
        model = nn.DataParallel(model)
        print(f"⚡ DataParallel across {torch.cuda.device_count()} GPUs")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ HierGeoNet | {n_params/1e6:.2f}M params")

    # Loss, optimiser, scheduler, scaler
    criterion = build_criterion(CFG.POS_WEIGHT, DEVICE)
    optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    _warmup, _total, _min_r = CFG.WARMUP_EPOCHS, CFG.EPOCHS, 1e-6 / CFG.LR

    def _lr_lambda(epoch):
        if epoch < _warmup:
            return float(epoch + 1) / float(_warmup)
        prog = (epoch - _warmup) / max(1, _total - _warmup)
        return max(_min_r, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = LambdaLR(optimizer, _lr_lambda)
    scaler    = GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)

    # Resume
    start_epoch, best_auc, patience = 0, 0.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        _unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_auc    = ckpt["best_auc"]
        print(f"▶  Resumed from epoch {start_epoch}, best AUC={best_auc:.4f}")

    # Training loop
    history = dict(train_loss=[], val_auc=[], val_f1=[], lr=[])
    CKPT    = MODEL_DIR / "hiergeonet_best.pt"
    print(f"\n🚀 Training | Device={DEVICE} | Batch={CFG.BATCH_SIZE} | Epochs={CFG.EPOCHS}")
    print("-" * 72)

    for epoch in range(start_epoch, CFG.EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
            y_batch = y_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)

            total_bce = 0.0
            all_embs, all_labels = [], []

            with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                for b in range(x_batch.shape[0]):
                    xf = x_batch[b]
                    xm = gpu_scatter_mean(xf, f2m_t, N_med)
                    xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                    lg, emb = _unwrap(model)(
                        xf, ei_f, xm, ei_m, xc, ei_c,
                        coords_c_t, f2m_t, m2c_t, f2c_t,
                        return_embeddings=True,
                    )
                    total_bce += criterion(lg, y_batch[b])
                    all_embs.append(emb)
                    all_labels.append(y_batch[b])

                embs_cat   = torch.cat(all_embs)
                labels_cat = torch.cat(all_labels)
                lc = contrastive_geo_loss(
                    embs_cat, coords_f_t.repeat(x_batch.shape[0], 1),
                    labels_cat,
                    radius=CFG.CONTRAST_RADIUS_PIX,
                    margin=CFG.CONTRAST_MARGIN,
                )
                loss = total_bce / x_batch.shape[0] + CFG.LAMBDA_CONTRAST * lc

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_probs, all_gt = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE, non_blocking=CFG.PIN_MEMORY)
                for b in range(x_batch.shape[0]):
                    xf = x_batch[b]
                    xm = gpu_scatter_mean(xf, f2m_t, N_med)
                    xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                    with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                        lg = _unwrap(model)(xf, ei_f, xm, ei_m, xc, ei_c,
                                           coords_c_t, f2m_t, m2c_t, f2c_t)
                    all_probs.append(torch.sigmoid(lg).float().cpu())
                    all_gt.append(y_batch[b])

        probs_np = torch.cat(all_probs).numpy()
        gt_np    = torch.cat(all_gt).numpy()
        val_auc  = roc_auc_score(gt_np, probs_np)
        val_f1   = f1_score(gt_np, (probs_np > 0.5).astype(int), average="macro")
        cur_lr   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t0

        mean_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(mean_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["lr"].append(cur_lr)

        print(f"Ep {epoch+1:03d}/{CFG.EPOCHS} | {elapsed:.1f}s | "
              f"loss={mean_loss:.4f} | auc={val_auc:.4f} | f1={val_f1:.4f} | lr={cur_lr:.2e}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save({
                "epoch":     epoch + 1,
                "model":     _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_auc":  best_auc,
            }, CKPT)
            print(f"   🌟 Best AUC → {best_auc:.4f}  (saved)")
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"⏹  Early stop at epoch {epoch+1}")
                break

    print(f"\n✅ Done. Best Val AUC = {best_auc:.4f}")
    with open(OUT_DIR / "history.json", "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"📄 History saved → {OUT_DIR}/history.json")


if __name__ == "__main__":
    main()

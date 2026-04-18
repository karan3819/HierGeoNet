#!/usr/bin/env python3
"""
HierGeoNet — Evaluation Script
================================
Loads a saved checkpoint and evaluates on the validation set.

Usage:
    python scripts/evaluate.py --data_dir /path/to/l4sdataset \
                                --checkpoint models/hiergeonet_best.pt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, average_precision_score,
    roc_curve, confusion_matrix, precision_recall_curve,
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model   import HierGeoNet, _unwrap
from src.dataset import build_dataloaders
from src.graph   import build_graph_topology, gpu_scatter_mean


class Config:
    IN_DIM=14; D_FINE=128; D_MED=192; D_COARSE=256
    GAT_HEADS=4; GAT_LAYERS=2; TF_HEADS=8; TF_LAYERS=3
    FF_DIM=512; DROPOUT=0.15
    LAMBDA_CONTRAST=0.1; CONTRAST_MARGIN=2.0; CONTRAST_RADIUS_PIX=5.0
    EPOCHS=300; BATCH_SIZE=8; LR=5e-4; WEIGHT_DECAY=1e-4; POS_WEIGHT=3.5
    GRAD_CLIP=1.0; PATIENCE=40; WARMUP_EPOCHS=20; NUM_WORKERS=0; PIN_MEMORY=True; SEED=42


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HierGeoNet on L4S validation set")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--out_dir",     default="outputs")
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--threshold",   type=float, default=None,
                   help="Decision threshold (default: auto-select via F1)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT  = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    CFG  = Config()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = torch.cuda.is_available()

    _, val_loader, _ = build_dataloaders(
        args.data_dir, batch_size=CFG.BATCH_SIZE,
        num_workers=args.workers, pin_memory=CFG.PIN_MEMORY and torch.cuda.is_available()
    )

    topo = build_graph_topology(128, 128, DEVICE)
    ei_f, ei_m, ei_c = topo["ei_f"], topo["ei_m"], topo["ei_c"]
    f2m_t, m2c_t, f2c_t = topo["f2m_t"], topo["m2c_t"], topo["f2c_t"]
    coords_c_t = topo["coords_c_t"]
    N_med, N_coarse = topo["N_med"], topo["N_coarse"]

    model = HierGeoNet(CFG).to(DEVICE)
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} (best AUC={ckpt['best_auc']:.4f})")

    all_probs, all_gt = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            for b in range(x_batch.shape[0]):
                xf = x_batch[b]
                xm = gpu_scatter_mean(xf, f2m_t, N_med)
                xc = gpu_scatter_mean(xm, m2c_t, N_coarse)
                with autocast(device_type=AMP_DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_AMP):
                    lg = model(xf, ei_f, xm, ei_m, xc, ei_c,
                               coords_c_t, f2m_t, m2c_t, f2c_t)
                all_probs.append(torch.sigmoid(lg).float().cpu())
                all_gt.append(y_batch[b])

    probs = torch.cat(all_probs).numpy()
    gt    = torch.cat(all_gt).numpy()

    # Threshold
    if args.threshold is None:
        prec_, rec_, thr_ = precision_recall_curve(gt, probs)
        f1_  = 2 * prec_ * rec_ / (prec_ + rec_ + 1e-8)
        thr  = float(thr_[np.argmax(f1_[:-1])])
        print(f"   Auto-selected threshold: {thr:.3f}")
    else:
        thr = args.threshold

    preds = (probs > thr).astype(int)

    metrics = {
        "AUC":       float(roc_auc_score(gt, probs)),
        "AP":        float(average_precision_score(gt, probs)),
        "F1_macro":  float(f1_score(gt, preds, average="macro")),
        "Precision": float(precision_score(gt, preds, zero_division=0)),
        "Recall":    float(recall_score(gt, preds, zero_division=0)),
        "MCC":       float(matthews_corrcoef(gt, preds)),
        "Threshold": thr,
    }

    print("\n" + "=" * 52)
    print("  HierGeoNet — Evaluation Results")
    print("=" * 52)
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.4f}")
    print("=" * 52)

    out_json = OUT / "eval_metrics.json"
    with open(out_json, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\n📄 Metrics saved → {out_json}")

    # ROC plot
    fpr, tpr, _ = roc_curve(gt, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={metrics['AUC']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="darkorange")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — HierGeoNet on L4S Validation"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=150)
    print(f"📊 ROC curve saved → {OUT}/roc_curve.png")


if __name__ == "__main__":
    main()

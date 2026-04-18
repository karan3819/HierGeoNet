# HierGeoNet 🏔️

> **Hierarchical Graph Attention Network with Cross-Scale Regularisation for Landslide Susceptibility Mapping**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.4%2B-informational)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![AUC](https://img.shields.io/badge/Val%20AUC-0.9877-brightgreen)](https://github.com)

<p align="center">
  <img src="assets/arch_overview.png" width="720" alt="HierGeoNet Architecture"/>
  <br/>
  <em>HierGeoNet processes a 128×128 Sentinel-2 patch through three simultaneous spatial graphs, connects them with learned cross-scale bridges, and assigns a landslide risk score to every pixel.</em>
</p>

---

## 📌 Overview

Landslide susceptibility mapping is inherently a **multi-scale problem**:

| Scale | Physical process | Graph |
|---|---|---|
| Local (10 m) | Slope geometry & debris mechanics | Fine graph — 16,384 nodes |
| Sub-catchment (300 m) | Drainage connectivity & runoff routing | Medium graph — 1,849 nodes |
| Regional (1.3 km) | Orographic rainfall gradients | Coarse graph + Transformer — 225 nodes |

Existing GNN approaches work at **a single spatial scale** and cannot capture all three simultaneously. HierGeoNet addresses this gap by:

1. Building three spatial graphs from Sentinel-2 patches simultaneously.
2. Connecting them through **cross-scale bridge modules** with learned per-dimension gating.
3. Applying full Transformer self-attention to the compact coarse graph for genuine global reasoning.
4. Adding a **geographic contrastive regulariser** with a provable Lipschitz embedding bound.

### Results on Landslide4Sense (validation set)

| Model | AUC | F1 | Prec. | Rec. | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9060 | 0.5618 | 0.1068 | 0.8301 | 0.2709 |
| Random Forest | 0.9341 | 0.6401 | 0.6925 | 0.1818 | 0.3498 |
| XGBoost | 0.9295 | 0.7765 | 0.5540 | 0.5721 | 0.5532 |
| CNN (ResNet-style) | 0.9330 | 0.7578 | 0.5069 | 0.5478 | 0.5161 |
| GCN | 0.9197 | 0.8102 | 0.6838 | 0.5806 | 0.6226 |
| GraphSAGE | 0.9424 | 0.8183 | 0.6903 | 0.6036 | 0.6382 |
| GAT (single-scale) | 0.9419 | 0.8312 | 0.7200 | 0.6252 | 0.6642 |
| SGCN-LSTM | 0.9446 | 0.8314 | 0.6834 | 0.6570 | 0.6629 |
| **HierGeoNet (ours)** | **0.9877** | **0.8326** | 0.6376 | **0.7087** | **0.6662** |

---

## 🗂️ Repository Structure

```
hiergeonet/
│
├── HierGeoNet_Notebook.ipynb   # Full training + evaluation notebook
│
├── src/                        # Core Python modules
│   ├── __init__.py
│   ├── model.py                # HierGeoNet, GATLayer, CrossScaleBridge, GeoPE
│   ├── dataset.py              # L4SDataset, DataLoader utilities
│   ├── graph.py                # Multi-scale graph topology builder
│   └── losses.py               # Weighted BCE + geographic contrastive loss
│
├── scripts/
│   ├── train.py                # Standalone training script (CLI)
│   └── evaluate.py             # Evaluation + metric reporting script
│
├── configs/
│   ├── default.yaml            # Paper-exact hyperparameters
│   └── himalaya.yaml           # 18-feature Himalayan terrain config
│
├── assets/                     # Images used in README
│
├── docs/                       # Additional documentation
│
├── requirements.txt            # pip dependencies
├── environment.yml             # conda environment spec
├── .gitignore
└── LICENSE                     # MIT
```

---

## ⚙️ Installation

### Option A — pip (recommended for GPU servers)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/hiergeonet.git
cd hiergeonet

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install PyTorch (match your CUDA version)
# CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch==2.1.0 torchvision==0.16.0

# 4. Install PyG and scatter ops (match your CUDA/torch version)
pip install torch-geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 5. Install remaining dependencies
pip install -r requirements.txt
```

### Option B — conda

```bash
git clone https://github.com/<your-username>/hiergeonet.git
cd hiergeonet
conda env create -f environment.yml
conda activate hiergeonet
```

### Verify installation

```python
import torch
from torch_geometric.nn import GATConv
print(torch.__version__)        # should be 2.1+
print(torch.cuda.is_available())
```

---

## 📦 Dataset

HierGeoNet is evaluated on **Landslide4Sense (L4S)** — a public benchmark of Sentinel-2 patches for global landslide detection.

### Download

1. Visit the official repository: https://www.research-collection.ethz.ch/handle/20.500.11850/510392
2. Download and unzip. Your directory should look like:

```
l4sdataset/
    TrainData/
        img/    image_00001.h5  image_00002.h5  ...   (3,799 patches)
        mask/   mask_00001.h5   mask_00002.h5   ...
    ValidData/
        img/    image_00001.h5  ...                   (245 patches)
        mask/   mask_00001.h5  ...
```

3. Each `.h5` file contains a 128×128 patch with 14 channels (12 Sentinel-2 bands + DEM slope + elevation) and a binary landslide mask.

### Dataset statistics

| Split | Patches | Pixels | Landslide pixels | Class ratio |
|---|---|---|---|---|
| Train | 3,799 | 62.5M | ~1.07M | ~58:1 |
| Val | 245 | 4.01M | ~68,745 | ~57:1 |

---

## 🚀 Quickstart

### Option 1 — Jupyter Notebook (recommended)

Open `HierGeoNet_Notebook.ipynb` and run cells top to bottom. Update `BASE_DATA_DIR` in Cell 4 to point to your L4S download.

```bash
jupyter notebook HierGeoNet_Notebook.ipynb
```

### Option 2 — Command line

**Train:**
```bash
python scripts/train.py \
    --data_dir /path/to/l4sdataset \
    --epochs 300 \
    --batch_size 8
```

**Evaluate a saved checkpoint:**
```bash
python scripts/evaluate.py \
    --data_dir /path/to/l4sdataset \
    --checkpoint models/hiergeonet_best.pt
```

### Option 3 — Import as a library

```python
from src.model import HierGeoNet
from src.graph import build_graph_topology, gpu_scatter_mean

class Config:
    IN_DIM=14; D_FINE=128; D_MED=192; D_COARSE=256
    GAT_HEADS=4; GAT_LAYERS=2; TF_HEADS=8; TF_LAYERS=3
    FF_DIM=512; DROPOUT=0.15
    # ... (see configs/default.yaml for all fields)

model = HierGeoNet(Config()).cuda()
topo  = build_graph_topology(128, 128, device='cuda')

# Forward pass for a single patch
logits = model(x_f, topo['ei_f'], x_m, topo['ei_m'], x_c, topo['ei_c'],
               topo['coords_c_t'], topo['f2m_t'], topo['m2c_t'], topo['f2c_t'])
probs  = torch.sigmoid(logits)   # [16384] — one risk score per pixel
```

---

## 🏗️ Architecture

### Multi-Scale Graph Definition

Given a 128×128 Sentinel-2 patch, three spatial graphs are constructed:

```
Scale    Stride    Nodes     Edges      δ (connect radius)
Fine     1×1       16,384    129,540    1.5 px  (8-neighbours)
Medium   3×3        1,849     14,280    4.5 px
Coarse   9×9          225      1,624   13.5 px
```

Node features: 14-dimensional (12 spectral bands + slope + elevation).  
Edge weights: Gaussian decay `w_ij = exp(−d²/2σ²)`, `σ = δ/2`.

### Bottom-Up Pass — Scale-Specific GAT Stacks

Each scale runs an independent 2-layer Graph Attention Network:

```
Attention score:  e_ij  = LeakyReLU(a^T · [Wh_i ‖ Wh_j])
Normalised weight: α_ij  = softmax over N(i)
Updated embedding: h_i'  = LN(GELU(Σ_j α_ij Wh_j) + W_skip·h_i)
```

Output embeddings: H(f) ∈ ℝ^{16384×128}, H(m) ∈ ℝ^{1849×192}, H(c) ∈ ℝ^{225×256}

### Geographic Positional Encoding

Learnable sinusoidal encoding of (lat, lon) coordinates:

```
GP(i, 4k)   = λ_{4k}   · sin(lat_i / τ^{4k/d})
GP(i, 4k+1) = λ_{4k+1} · cos(lat_i / τ^{4k/d})
GP(i, 4k+2) = λ_{4k+2} · sin(lon_i / τ^{4k/d})
GP(i, 4k+3) = λ_{4k+3} · cos(lon_i / τ^{4k/d})
```

where τ=10,000 and λ ∈ ℝ^256 is a learnable scale initialised to 1.

### Coarse Transformer

Full O(N²_c) multi-head self-attention over the 225 coarse nodes:

```
H_global = TransformerEncoder(H(c) + GP)   ∈ ℝ^{225×256}
```

Cost: 225² = 50,625 attention pairs — negligible on GPU.  
Architecture: 3 encoder layers, 8 heads, 512-dim feedforward, pre-norm.

### Cross-Scale Bridge (Key Contribution)

Gated top-down context injection (Equations 10–12):

```
c̃_i  = W_C  · H_global[parent(i)]          (project coarse to fine dim)
g_i   = σ(W_g · [h_fine_i ; c̃_i])          (per-dimension gate ∈ (0,1)^d)
h_i'  = LN(h_fine_i + g_i ⊙ tanh(W_ctx·c̃_i))  (gated residual update)
```

**Theorem 1:** When g_i → 0, HierGeoNet degenerates to a single-scale GNN. The bridge's hypothesis class strictly contains all single-scale GNNs.

### Geographic Contrastive Regulariser

Prevents over-smoothing by enforcing spatial consistency:

```
L_C = (1/|P+|) Σ_{P+} ‖z_i−z_j‖²  +  (1/|P-|) Σ_{P-} max(0, m−‖z_i−z_j‖)²
```

where P+ = nearby same-label pairs (r=5px), P− = far or different-label pairs, m=2.0.

**Proposition 1 (Lipschitz Bound):** `‖z_i − z_j‖ ≤ √(L_C) · ⌈geo_dist(i,j)/r⌉`

### Total Training Objective

```
L = L_BCE + 0.1 · L_C
```

where L_BCE uses positive weight w+ = 3.5 to handle 58:1 class imbalance.

---

## 🔬 Ablation Study

| Configuration | AUC | F1 | ΔAUC |
|---|---|---|---|
| **HierGeoNet (full)** | **0.9877** | **0.7877** | — |
| w/o cross-scale bridges | 0.9736 | 0.7084 | −0.0141 |
| w/o medium graph | 0.9751 | 0.7974 | −0.0126 |
| w/o coarse Transformer | 0.9873 | 0.7835 | −0.0004 |

The cross-scale bridges account for the largest single improvement (+1.41 AUC), confirming that carrying coarse global context into fine-scale predictions is the primary driver of HierGeoNet's advantage.

---

## ⚡ Hardware & Training Details

| Setting | Value |
|---|---|
| GPUs | 2× NVIDIA L4 (24 GB each) |
| Training time | ~41 min to best checkpoint (epoch 27) |
| GPU memory per card | < 8 GB |
| Parameters | 2.43M |
| Mixed precision | BF16 (autocast + GradScaler) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| LR schedule | 20-epoch linear warmup → cosine to 1e-6 |
| Early stopping | patience = 40 epochs |
| Best epoch | 27 / 300 |

---

## 🏔️ Himalayan Extension

HierGeoNet is architecturally designed for Himalayan terrain. The only change required is expanding the input feature vector from 14 → 18 dimensions:

| New Feature | Source | Physical motivation |
|---|---|---|
| Slope angle | SRTM / Cartosat DEM | Primary failure driver |
| Plan curvature | DEM-derived | Concave slopes concentrate runoff |
| Aspect (sin, cos) | DEM-derived | SW-facing slopes receive 3× more rain in HP |
| 30-day rainfall | NASA-GPM IMERG | Dynamic monsoon trigger |
| NDVI | Sentinel-2 composite | Root reinforcement reduces risk |
| Lithology (7-class) | Bhukosh GSI maps | Phyllites fail at lower angles than granite |
| Land cover (5-class) | MODIS MCD12Q1 | Deforested slopes = higher risk |

Use `configs/himalaya.yaml` to activate this configuration.

> **Note:** A standardised polygon-level landslide inventory for Himachal Pradesh and Uttarakhand at Sentinel-2 resolution is currently unavailable. This is the primary barrier to direct empirical validation in that region.

---

## 📋 Configuration Reference

All hyperparameters are documented in `configs/default.yaml`. Key settings:

```yaml
model:
  d_fine:   128    # fine-scale embedding dim
  d_med:    192    # medium-scale embedding dim
  d_coarse: 256    # coarse-scale embedding dim
  gat_heads:  4
  gat_layers: 2
  tf_heads:   8
  tf_layers:  3

training:
  lr:          5.0e-4
  pos_weight:  3.5     # handles 58:1 class imbalance
  warmup_epochs: 20
  patience:    40
```

---

## 📄 Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{pathania2025hiergeonet,
  title   = {HierGeoNet: Graph Attention with Cross-Scale Regularisation
             for Landslide Susceptibility},
  author  = {Pathania, Karan and Singh, Sukhdeep and Sharma, Anuj},
  journal = {arXiv preprint},
  year    = {2025},
  institution = {Department of Computer Science and Applications,
                 Panjab University, Chandigarh, India}
}
```

---

## 🙏 Acknowledgements

- **Landslide4Sense** dataset providers: Ghorbanzadeh et al. (2022)
- **ESA Copernicus Programme** for Sentinel-2 satellite data
- **Geological Survey of India (GSI)** for Bhukosh lithology maps
- **NASA Earthdata / GPM** for IMERG rainfall data
- **Panjab University, Chandigarh** for computational resources

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change. For major changes, please fork the repository and submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

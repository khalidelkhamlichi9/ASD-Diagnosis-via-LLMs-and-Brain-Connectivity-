"""
baselines/brainnetcnn_baseline.py
──────────────────────────────────
BrainNetCNN (Kannah et al., NeuroImage 2016) : Baseline Deep Learning de référence 
pour les matrices de connectivité fonctionnelle (fMRI FC).

Architecture :
  - E2E (Edge-to-Edge) Convolution (1xN et Nx1)
  - E2N (Edge-to-Node) Convolution (1xN)
  - N2G (Node-to-Graph) Dense Layer
"""

import argparse
import json
import os
import sys
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dataset import load_abide, inter_site_split
from baselines.shared_eval import evaluate_metrics, print_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# BrainNetCNN Layers (Edge-to-Edge, Edge-to-Node, Node-to-Graph)
# ─────────────────────────────────────────────────────────────────────────────

class E2EBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, n_rois: int = 200):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1, n_rois), bias=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, (n_rois, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        # x shape: (B, in_planes, N, N)
        a = self.conv1(x) # (B, out_planes, N, 1)
        b = self.conv2(x) # (B, out_planes, 1, N)
        # Broadcast sum to (B, out_planes, N, N)
        return F.leaky_relu(self.bn1(a) + self.bn2(b), 0.33)

class E2NBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, n_rois: int = 200):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, (1, n_rois), bias=False)
        self.bn = nn.BatchNorm1d(out_planes)

    def forward(self, x):
        # x shape: (B, in_planes, N, N)
        out = self.conv(x) # (B, out_planes, N, 1)
        out = out.squeeze(-1) # (B, out_planes, N)
        return F.leaky_relu(self.bn(out), 0.33)

class BrainNetCNN(nn.Module):
    def __init__(self, n_rois: int = 200, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.e2e = E2EBlock(1, 16, n_rois)
        self.e2n = E2NBlock(16, 32, n_rois)
        self.n2g = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32 * n_rois, 128),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(128, 30),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(30, num_classes)
        )

    def forward(self, x):
        # x shape: (B, N, N) -> add channel dim (B, 1, N, N)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.e2e(x)
        h = self.e2n(h) # (B, 32, N)
        h = h.view(h.size(0), -1) # Flatten (B, 32*N)
        return self.n2g(h)


class FCMultiSiteDataset(Dataset):
    def __init__(self, fc_all: np.ndarray, labels: np.ndarray, sites: np.ndarray, augment: bool = False):
        self.fc = torch.from_numpy(fc_all.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sites = torch.from_numpy(sites.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        fc = self.fc[i].clone()
        if self.augment:
            noise = torch.randn_like(fc) * 0.005
            fc = fc + (noise + noise.t()) / 2
        return fc, self.labels[i], self.sites[i]

def _eval_loader(model, loader, device):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for fc, lbl, site in loader:
            logits = model(fc.to(device))
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            gts.extend(lbl.numpy())
            sites.extend(site.numpy())
    return np.array(probs), np.array(gts), np.array(sites)

def run_brainnetcnn(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  BrainNetCNN Baseline | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    (ts, fc_corr_all, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_ds = FCMultiSiteDataset(fc_corr_all[train_idx], lbl_all[train_idx], site_idx[train_idx], augment=True)
    val_ds   = FCMultiSiteDataset(fc_corr_all[val_idx], lbl_all[val_idx], site_idx[val_idx], augment=False)
    test_ds  = FCMultiSiteDataset(fc_corr_all[test_idx], lbl_all[test_idx], site_idx[test_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BrainNetCNN(n_rois=200, num_classes=2, dropout=args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    print(f"\n[4/5] Entraînement BrainNetCNN ({args.epochs} epochs max)...")

    best_sel = -1.
    best_ep = 0
    best_weights = None
    patience_cnt = 0

    for ep in range(args.epochs):
        model.train()
        for fc, lbl, _ in train_loader:
            fc, lbl = fc.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(fc)
            loss = F.cross_entropy(logits, lbl, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        vp, vg, _ = _eval_loader(model, val_loader, DEVICE)
        v_acc  = accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
        v_bacc = balanced_accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
        v_auc  = roc_auc_score(vg, vp) * 100
        sel    = 0.5 * v_acc + 0.5 * v_auc

        if sel > best_sel:
            best_sel = sel
            best_ep = ep
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            break

    print(f"\n[5/5] Évaluation finale (Checkpoint Ep {best_ep+1})...")
    model.load_state_dict(best_weights)
    tp, tg, ts_arr = _eval_loader(model, test_loader, DEVICE)
    metrics = evaluate_metrics(tp, tg, ts_arr, idx2site)
    print_results(f"BrainNetCNN (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "BrainNetCNN",
        "seed": args.seed,
        "acc": float(metrics["acc"]),
        "bacc": float(metrics["bacc"]),
        "auc": float(metrics["auc"]),
        "worst_site_acc": float(metrics["worst_site_acc"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"brainnetcnn_results_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./abide_data")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=10)
    run_brainnetcnn(p.parse_args())

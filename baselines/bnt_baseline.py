"""
baselines/bnt_baseline.py
──────────────────────────
Baseline Brain Network Transformer (BNT) : Version optimisée pour 
maximiser l'Accuracy/AUC (60%+) sur les sites invisibles.

Recette magique :
  1. K=16 (clusters) au lieu de 8, pour plus de granularité.
  2. Architecture allégée : 1 seule couche de BNT Block pour éviter
     le surapprentissage des paramètres d'attention sur un petit dataset.
  3. Suppression du Edge Dropout qui dégradait la topologie du graphe.
  4. Label Smoothing (0.2) + L2 modéré.
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
from baselines.bnt_model import BrainNetworkTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCMatrixDataset(Dataset):
    def __init__(self, fc_all: np.ndarray, labels: np.ndarray,
                 sites: np.ndarray, augment: bool = False):
        self.fc = torch.from_numpy(fc_all.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sites = torch.from_numpy(sites.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        fc = self.fc[i].clone()
        if self.augment:
            # Très léger bruit gaussien
            noise = torch.randn_like(fc) * 0.005
            fc = fc + (noise + noise.t()) / 2
        return fc, self.labels[i], self.sites[i]


def make_bnt_loaders(fc_all, lbl_all, site_idx,
                     train_idx, val_idx, test_idx,
                     batch_size: int = 16):
    def _mk(idx, aug, shuf):
        ds = FCMatrixDataset(fc_all[idx], lbl_all[idx], site_idx[idx], aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuf and len(idx)>0))
    return _mk(train_idx, True, True), _mk(val_idx, False, False), _mk(test_idx, False, False)

def _eval_loader(model, loader, device):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for fc, lbl, site in loader:
            logits, _ = model(fc.to(device))
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            gts.extend(lbl.numpy())
            sites.extend(site.numpy())
    return np.array(probs), np.array(gts), np.array(sites)


def run_bnt(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  BNT Baseline Optimisée | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    # 1. Données
    (_, fc_corr_all, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_loader, val_loader, test_loader = make_bnt_loaders(
        fc_corr_all, lbl_all, site_idx, train_idx, val_idx, test_idx,
        batch_size=args.batch_size
    )

    # Modèle BNT (K=16, 1 layer)
    model = BrainNetworkTransformer(
        n_roi=200, n_clusters=args.n_clusters, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        n_classes=2, lambda_oc=args.lambda_oc
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    print(f"\n[4/5] Entraînement BNT ({args.epochs} epochs max)...")

    best_sel = -1.
    best_ep = 0
    best_weights = None
    patience_cnt = 0

    for ep in range(args.epochs):
        model.train()
        for fc, lbl, _ in train_loader:
            fc, lbl = fc.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits, l_oc = model(fc)
            
            loss_cls = F.cross_entropy(logits, lbl, label_smoothing=0.2)
            loss = loss_cls + args.lambda_oc * l_oc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        vp, vg, _ = _eval_loader(model, val_loader, DEVICE)
        v_acc  = accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
        v_bacc = balanced_accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
        v_auc  = roc_auc_score(vg, vp) * 100
        sel    = 0.5 * v_bacc + 0.5 * v_auc

        if sel > best_sel:
            best_sel = sel
            best_ep = ep
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            break

    # Évaluation
    print(f"\n[5/5] Évaluation finale (Checkpoint Ep {best_ep+1})...")
    model.load_state_dict(best_weights)
    tp, tg, ts_arr = _eval_loader(model, test_loader, DEVICE)
    metrics = evaluate_metrics(tp, tg, ts_arr, idx2site)
    print_results(f"BNT K={args.n_clusters} (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "BNT (Optimized)",
        "seed": args.seed,
        "n_clusters": args.n_clusters,
        "d_model": args.d_model,
        "acc": float(metrics["acc"]),
        "bacc": float(metrics["bacc"]),
        "auc": float(metrics["auc"]),
        "worst_site_acc": float(metrics["worst_site_acc"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"bnt_results_K{args.n_clusters}_d{args.d_model}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./abide_data")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3) 
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.4) 
    p.add_argument("--lambda_oc", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n_clusters", type=int, default=8)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    run_bnt(p.parse_args())

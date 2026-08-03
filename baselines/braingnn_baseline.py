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
            noise = torch.randn_like(fc) * 0.005
            fc = fc + (noise + noise.t()) / 2
        return fc, self.labels[i], self.sites[i]

def make_loaders(fc_all, lbl_all, site_idx,
                 train_idx, val_idx, test_idx,
                 batch_size: int = 32):
    def _mk(idx, aug, shuf):
        ds = FCMatrixDataset(fc_all[idx], lbl_all[idx], site_idx[idx], aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuf and len(idx)>0))
    return _mk(train_idx, True, True), _mk(val_idx, False, False), _mk(test_idx, False, False)

class BrainGNN(nn.Module):
    def __init__(self, n_roi=200, k=30, dropout=0.5):
        super().__init__()
        self.k = k
        self.dropout = dropout
        
        self.gc1 = nn.Linear(n_roi, 64)
        self.gc2 = nn.Linear(64, 32)
        self.node_weights = nn.Parameter(torch.randn(n_roi))
        
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)
        )
        
    def forward(self, fc):
        B, N, _ = fc.shape
        
        fc_abs = torch.abs(fc)
        _, indices = torch.topk(fc_abs, self.k, dim=-1)
        
        A = torch.zeros_like(fc)
        A.scatter_(2, indices, fc)
        
        I = torch.eye(N, device=fc.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = A + I
        row_sum = A_hat.abs().sum(dim=-1, keepdim=True) + 1e-6
        A_hat = A_hat / row_sum
        
        X = fc 
        X = torch.bmm(A_hat, X)
        X = F.relu(self.gc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        
        X = torch.bmm(A_hat, X)
        X = F.relu(self.gc2(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        
        w = torch.sigmoid(self.node_weights).unsqueeze(0).unsqueeze(-1)
        X = X * w 
        X = X.sum(dim=1) 
        
        logits = self.mlp(X)
        return logits

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

def run_braingnn(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  BrainGNN Baseline | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    (_, fc_corr_all, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_loader, val_loader, test_loader = make_loaders(
        fc_corr_all, lbl_all, site_idx, train_idx, val_idx, test_idx,
        batch_size=args.batch_size
    )

    model = BrainGNN(n_roi=200, k=30, dropout=0.5).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    print(f"\n[4/5] Training BrainGNN ({args.epochs} epochs max)...")

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

    print(f"\n[5/5] Evaluation (Checkpoint Ep {best_ep+1})...")
    model.load_state_dict(best_weights)
    tp, tg, ts_arr = _eval_loader(model, test_loader, DEVICE)
    metrics = evaluate_metrics(tp, tg, ts_arr, idx2site)
    print_results(f"BrainGNN (seed={args.seed})", metrics, test_sites)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results = {
        "method": "BrainGNN",
        "seed": args.seed,
        "n_params": n_params,
        "acc": float(metrics["acc"]),
        "bacc": float(metrics["bacc"]),
        "auc": float(metrics["auc"]),
        "specificity": float(metrics.get("specificity", 0)),
        "mcc": float(metrics.get("mcc", 0)),
        "worst_site_acc": float(metrics["worst_site_acc"]),
        "avg_site_acc": float(metrics.get("avg_site_acc", 0)),
        "std_site_acc": float(metrics.get("std_site_acc", 0)),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"braingnn_results_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./abide_data")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4) 
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=15)
    run_braingnn(p.parse_args())

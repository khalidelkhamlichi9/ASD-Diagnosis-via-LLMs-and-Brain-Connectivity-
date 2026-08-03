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

class FBNETDataset(Dataset):
    def __init__(self, ts_all: np.ndarray, fc_all: np.ndarray, labels: np.ndarray,
                 sites: np.ndarray, augment: bool = False):
        self.ts = torch.from_numpy(ts_all.astype(np.float32))
        self.fc = torch.from_numpy(fc_all.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sites = torch.from_numpy(sites.astype(np.int64))
        self.augment = augment

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        ts = self.ts[i].clone()
        fc = self.fc[i].clone()
        if self.augment:
            noise_fc = torch.randn_like(fc) * 0.005
            fc = fc + (noise_fc + noise_fc.t()) / 2
            
            noise_ts = torch.randn_like(ts) * 0.005
            ts = ts + noise_ts
        return ts, fc, self.labels[i], self.sites[i]

def make_loaders(ts_all, fc_all, lbl_all, site_idx,
                 train_idx, val_idx, test_idx,
                 batch_size: int = 32):
    def _mk(idx, aug, shuf):
        ds = FBNETDataset(ts_all[idx], fc_all[idx], lbl_all[idx], site_idx[idx], aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuf and len(idx)>0))
    return _mk(train_idx, True, True), _mk(val_idx, False, False), _mk(test_idx, False, False)

class FBNetGen(nn.Module):
    def __init__(self, n_roi=200, seq_len=160, dropout=0.4):
        super().__init__()
        
        self.ts_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        
        self.gc1 = nn.Linear(n_roi, 64)
        self.gc2 = nn.Linear(64, 32)
        
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)
        )
        self.dropout = dropout
        
    def forward(self, ts, fc):
        B, T, N = ts.shape
        
        x = ts.transpose(1, 2).reshape(B*N, 1, T)
        emb = self.ts_conv(x).view(B, N, 32)
        
        emb_norm = F.normalize(emb, p=2, dim=-1)
        gen_fc = torch.bmm(emb_norm, emb_norm.transpose(1, 2))
        
        alpha = torch.sigmoid(self.alpha)
        A = alpha * gen_fc + (1 - alpha) * fc
        
        I = torch.eye(N, device=fc.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = A + I
        row_sum = A_hat.abs().sum(dim=-1, keepdim=True) + 1e-6
        A_hat = A_hat / row_sum
        
        X = A
        X = torch.bmm(A_hat, X)
        X = F.relu(self.gc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        
        X = torch.bmm(A_hat, X)
        X = F.relu(self.gc2(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        
        X = X.mean(dim=1)
        
        logits = self.mlp(X)
        return logits

def _eval_loader(model, loader, device):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for ts, fc, lbl, site in loader:
            logits = model(ts.to(device), fc.to(device))
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            gts.extend(lbl.numpy())
            sites.extend(site.numpy())
    return np.array(probs), np.array(gts), np.array(sites)

def run_fbnetgen(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  FBNetGen Baseline | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    (ts_all, fc_corr_all, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, fc_corr_all, lbl_all, site_idx, train_idx, val_idx, test_idx,
        batch_size=args.batch_size
    )

    model = FBNetGen(n_roi=200, seq_len=ts_all.shape[1], dropout=0.4).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    print(f"\n[4/5] Training FBNetGen ({args.epochs} epochs max)...")

    best_sel = -1.
    best_ep = 0
    best_weights = None
    patience_cnt = 0

    for ep in range(args.epochs):
        model.train()
        for ts, fc, lbl, _ in train_loader:
            ts, fc, lbl = ts.to(DEVICE), fc.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(ts, fc)
            
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
    print_results(f"FBNetGen (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "FBNetGen",
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
    out_path = os.path.join(args.results_dir, f"fbnetgen_results_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
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
    run_fbnetgen(p.parse_args())

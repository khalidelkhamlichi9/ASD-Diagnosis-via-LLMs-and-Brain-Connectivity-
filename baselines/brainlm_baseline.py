"""
baselines/brainlm_baseline.py
BrainLM* (simplified, trained from scratch on ABIDE-I).
Note: No external pre-training (scratch only, not the original BrainLM).
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dataset import load_abide, inter_site_split
from baselines.shared_eval import evaluate_metrics, print_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCDataset(Dataset):
    def __init__(self, fc_all: np.ndarray, labels: np.ndarray, sites: np.ndarray):
        # We assume fc_all is (N, 200, 200)
        self.fc = torch.from_numpy(fc_all.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sites = torch.from_numpy(sites.astype(np.int64))
        
        # Pre-extract upper triangle (19900)
        triu_idx = torch.triu_indices(200, 200, offset=1)
        self.fc_triu = self.fc[:, triu_idx[0], triu_idx[1]] # (N, 19900)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.fc_triu[i], self.labels[i], self.sites[i]

def make_loaders(fc_all, lbl_all, site_idx, train_idx, val_idx, test_idx, batch_size):
    def _mk(idx, shuf):
        ds = FCDataset(fc_all[idx], lbl_all[idx], site_idx[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuf and len(idx) > 0))
    return _mk(train_idx, True), _mk(val_idx, False), _mk(test_idx, False)

class BrainLM(nn.Module):
    def __init__(self, input_dim=19900, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)
        return logits, mu, logvar

def _eval_loader(model, loader, device):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for x, lbl, site in loader:
            logits, _, _ = model(x.to(device))
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            gts.extend(lbl.numpy())
            sites.extend(site.numpy())
    return np.array(probs), np.array(gts), np.array(sites)

def run_brainlm(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  BrainLM (scratch) Baseline | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    (_, fc_corr_all, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_loader, val_loader, test_loader = make_loaders(
        fc_corr_all, lbl_all, site_idx, train_idx, val_idx, test_idx, batch_size=args.batch_size
    )

    model = BrainLM().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    best_sel = -1.
    best_ep = 0
    best_weights = None
    patience_cnt = 0

    beta = args.beta

    for ep in range(args.epochs):
        model.train()
        for x, lbl, _ in train_loader:
            x, lbl = x.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits, mu, logvar = model(x)
            
            loss_cls = F.cross_entropy(logits, lbl, label_smoothing=0.1)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = loss_cls + beta * loss_kl
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        vp, vg, _ = _eval_loader(model, val_loader, DEVICE)
        if len(vg) > 0:
            v_bacc = balanced_accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
            v_auc  = roc_auc_score(vg, vp) * 100
            sel    = 0.5 * v_bacc + 0.5 * v_auc
        else:
            sel = 0

        if sel > best_sel:
            best_sel = sel
            best_ep = ep
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            break

    print(f"\nEvaluation (Checkpoint Ep {best_ep+1})...")
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    tp, tg, ts_arr = _eval_loader(model, test_loader, DEVICE)
    metrics = evaluate_metrics(tp, tg, ts_arr, idx2site)
    print_results(f"BrainLM (scratch) (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "BrainLM (scratch)",
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
    out_path = os.path.join(args.results_dir, f"brainlm_results_seed{args.seed}.json")
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
    p.add_argument("--beta", type=float, default=0.001)
    run_brainlm(p.parse_args())

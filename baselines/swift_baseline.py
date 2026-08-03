import argparse
import json
import os
import sys
import copy
import math
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

class TSDataset(Dataset):
    def __init__(self, ts_all: np.ndarray, labels: np.ndarray, sites: np.ndarray):
        self.ts = torch.from_numpy(ts_all.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sites = torch.from_numpy(sites.astype(np.int64))

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.ts[i], self.labels[i], self.sites[i]

def make_loaders(ts_all, lbl_all, site_idx, train_idx, val_idx, test_idx, batch_size):
    def _mk(idx, shuf):
        ds = TSDataset(ts_all[idx], lbl_all[idx], site_idx[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuf and len(idx) > 0))
    return _mk(train_idx, True), _mk(val_idx, False), _mk(test_idx, False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SWiFT(nn.Module):
    def __init__(self, n_rois=200, window_size=20, stride=5, d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.2):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.proj = nn.Linear(n_rois, d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (B, T, N)
        B, T, N = x.shape
        
        # Extract windows
        windows = []
        for start in range(0, T - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = x[:, start:end, :] # (B, 20, 200)
            window_emb = self.proj(window).mean(dim=1) # (B, d_model)
            windows.append(window_emb)
            
        x = torch.stack(windows, dim=1) # (B, seq_len, d_model)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        cls_out = x[:, 0, :]
        logits = self.classifier(cls_out)
        return logits

def _eval_loader(model, loader, device):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for ts, lbl, site in loader:
            logits = model(ts.to(device))
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            gts.extend(lbl.numpy())
            sites.extend(site.numpy())
    return np.array(probs), np.array(gts), np.array(sites)

def run_swift(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  SWiFT Baseline | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    (ts_all, _, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, lbl_all, site_idx, train_idx, val_idx, test_idx, batch_size=args.batch_size
    )

    model = SWiFT().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    best_sel = -1.
    best_ep = 0
    best_weights = None
    patience_cnt = 0

    for ep in range(args.epochs):
        model.train()
        for ts, lbl, _ in train_loader:
            ts, lbl = ts.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(ts)
            
            loss = F.cross_entropy(logits, lbl, label_smoothing=0.1)
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
    print_results(f"SWiFT (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "SWiFT",
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
    out_path = os.path.join(args.results_dir, f"swift_results_seed{args.seed}.json")
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
    run_swift(p.parse_args())

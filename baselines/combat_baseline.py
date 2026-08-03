"""
baselines/combat_baseline.py
──────────────────────────
Baseline ComBat + MLP : Version optimisée pour maximiser l'Accuracy/AUC 
sur les sites invisibles (Test Set) tout en évitant le data leakage.

Recette magique :
  1. ComBat EB strict (fit sur train uniquement).
  2. Conservation de TOUTES les 19900 features (l'ANOVA causait un surapprentissage 
     aux sites d'entraînement).
  3. Réseau "Large-but-sparse" : Couche cachée large (256) mais avec un 
     Dropout massif (0.75) pour forcer la robustesse sans réduire l'expressivité.
  4. Label Smoothing (0.15) pour éviter la surconfiance.
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
# 1. ComBat Empirique Bayésien Interne (Strict)
# ─────────────────────────────────────────────────────────────────────────────

class EmpiricalBayesCombat:
    def __init__(self):
        self.gamma_star = {}
        self.delta_star = {}
        self.grand_mean = None
        self.var_pooled = None
        self.known_sites = []

    def fit(self, X: np.ndarray, site_labels: np.ndarray):
        sites = np.unique(site_labels)
        self.known_sites = sites.tolist()
        n_features = X.shape[1]

        self.grand_mean = np.mean(X, axis=0)
        X_centered = X - self.grand_mean
        
        var_pooled = np.zeros(n_features)
        for s in sites:
            mask = (site_labels == s)
            var_pooled += np.var(X_centered[mask], axis=0) * np.sum(mask)
        var_pooled /= len(site_labels)
        self.var_pooled = np.clip(var_pooled, 1e-8, None)

        X_stand = X_centered / np.sqrt(self.var_pooled)

        for s in sites:
            mask = (site_labels == s)
            Xs = X_stand[mask]
            
            gamma_hat = np.mean(Xs, axis=0)
            delta_hat = np.var(Xs, axis=0)
            
            n = np.sum(mask)
            gamma_bar = np.mean(gamma_hat)
            tau2 = np.var(gamma_hat)
            
            d_mean = np.mean(delta_hat)
            d_var = np.var(delta_hat)
            lambda_ = max((2 * d_mean**2) / (d_var + 1e-8) + 2, 2.1)
            theta_ = d_mean * (lambda_ - 1)
            
            self.gamma_star[s] = (n * tau2 * gamma_hat + d_mean * gamma_bar) / (n * tau2 + d_mean + 1e-8)
            self.delta_star[s] = (theta_ + 0.5 * n * delta_hat) / (lambda_ + 0.5 * n - 1 + 1e-8)

    def transform(self, X: np.ndarray, site_labels: np.ndarray) -> np.ndarray:
        X_trans = np.zeros_like(X)
        for s in np.unique(site_labels):
            mask = (site_labels == s)
            if s in self.known_sites:
                g = self.gamma_star[s]
                d = self.delta_star[s]
                X_stand = (X[mask] - self.grand_mean) / np.sqrt(self.var_pooled)
                X_trans[mask] = ((X_stand - g) / np.sqrt(d + 1e-8)) * np.sqrt(self.var_pooled) + self.grand_mean
            else:
                mean_s = np.mean(X[mask], axis=0)
                std_s = np.std(X[mask], axis=0) + 1e-8
                X_trans[mask] = ((X[mask] - mean_s) / std_s) * np.sqrt(self.var_pooled) + self.grand_mean
        return X_trans


# ─────────────────────────────────────────────────────────────────────────────
# 2. MLP Classifier Optimisé
# ─────────────────────────────────────────────────────────────────────────────

class RobustASDClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, s: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))
        self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            # Gaussian noise to simulate scanner variance
            x = x + torch.randn_like(x) * 0.01
        return x, self.y[i], self.s[i]


def get_upper_tri(fc: np.ndarray) -> np.ndarray:
    B, N, _ = fc.shape
    idx = np.triu_indices(N, k=1)
    return np.array([m[idx] for m in fc])

def _eval_loader(model, loader, device):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for x, y, s in loader:
            logits = model(x.to(device))
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            gts.extend(y.numpy())
            sites.extend(s.numpy())
    return np.array(probs), np.array(gts), np.array(sites)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Principal
# ─────────────────────────────────────────────────────────────────────────────

def run_combat(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"\n{'='*62}")
    print(f"  ComBat Baseline Optimisée | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    # 1. Chargement & Split
    (ts, fc_corr, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    # 2. Vectorisation
    X_all = get_upper_tri(fc_corr)
    
    X_tr, y_tr, s_tr = X_all[train_idx], lbl_all[train_idx], site_idx[train_idx]
    X_va, y_va, s_va = X_all[val_idx], lbl_all[val_idx], site_idx[val_idx]
    X_te, y_te, s_te = X_all[test_idx], lbl_all[test_idx], site_idx[test_idx]

    # 3. ComBat EB
    combat = EmpiricalBayesCombat()
    combat.fit(X_tr, s_tr)
    X_tr = combat.transform(X_tr, s_tr)
    X_va = combat.transform(X_va, s_va)
    X_te = combat.transform(X_te, s_te)

    tr_loader = DataLoader(VectorDataset(X_tr, y_tr, s_tr, augment=True), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(VectorDataset(X_va, y_va, s_va, augment=False), batch_size=args.batch_size)
    te_loader = DataLoader(VectorDataset(X_te, y_te, s_te, augment=False), batch_size=args.batch_size)

    # 4. Modèle
    model = RobustASDClassifier(in_features=19900, hidden_dim=64, dropout=0.5).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    
    print(f"\n[4/5] Entraînement MLP ({args.epochs} epochs max)...")

    best_sel = -1.
    best_ep = 0
    best_weights = None
    patience_cnt = 0

    for ep in range(args.epochs):
        model.train()
        for x, y, _ in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            # Label smoothing fort (0.15) pour éviter la surconfiance
            loss = F.cross_entropy(logits, y, label_smoothing=0.15)
            loss.backward()
            opt.step()
        sched.step()

        vp, vg, _ = _eval_loader(model, va_loader, DEVICE)
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

    # 5. Evaluation
    print(f"\n[5/5] Évaluation finale (Checkpoint Ep {best_ep+1})...")
    model.load_state_dict(best_weights)
    tp, tg, ts_arr = _eval_loader(model, te_loader, DEVICE)
    metrics = evaluate_metrics(tp, tg, ts_arr, idx2site)
    print_results(f"ComBat + Robust MLP (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "ComBat (EB) + Robust MLP",
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
    out_path = os.path.join(args.results_dir, f"combat_results_seed{args.seed}.json")
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
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--patience", type=int, default=15)
    run_combat(p.parse_args())

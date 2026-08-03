"""
baselines/brainjepa_baseline.py
────────────────────────────────
Brain-JEPA* (Joint Embedding Predictive Architecture on FC patches)
Trained from scratch on ABIDE-I — no external pre-training corpus.

NOTE: This is a simplified adaptation of JEPA principles to FC matrices.
      Not the original Brain-JEPA pre-trained on large-scale fMRI data.

Architecture:
  - Input: FC correlation matrix (200x200)
  - Split into 4x4 = 16 ROI-block patches of size 50x50 → flatten → 2500-d each
  - Context encoder: Linear(2500, 128) → LayerNorm → GELU (processes 12/16 patches)
  - Predictor: Transformer(d=128, nhead=4, nlayers=2) predicts target patch embeddings
  - Target encoder: EMA copy of context encoder (momentum=0.996)
  - JEPA loss: L2 between predicted and target embeddings (self-supervised)
  - Phase 1 (25 epochs): JEPA self-supervised pre-training
  - Phase 2 (25 epochs): Linear probe on frozen context encoder (avg pool patches → 128 → 2)
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

N_ROI = 200
PATCH_SIZE = 50          # 50x50 ROI block patches
N_PATCHES  = (N_ROI // PATCH_SIZE) ** 2   # = 16
PATCH_DIM  = PATCH_SIZE * PATCH_SIZE       # = 2500
D_MODEL    = 128
EMA_DECAY  = 0.996
N_CTX      = 12          # number of context patches (out of 16)
N_TGT      = 4           # number of target patches to predict


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FCPatchDataset(Dataset):
    def __init__(self, fc_all: np.ndarray, labels: np.ndarray, sites: np.ndarray):
        self.fc     = torch.from_numpy(fc_all.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.sites  = torch.from_numpy(sites.astype(np.int64))

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        fc = self.fc[i]  # (N, N)
        # Extract 4x4 patches of size 50x50
        patches = []
        for pr in range(N_ROI // PATCH_SIZE):
            for pc in range(N_ROI // PATCH_SIZE):
                patch = fc[pr*PATCH_SIZE:(pr+1)*PATCH_SIZE, pc*PATCH_SIZE:(pc+1)*PATCH_SIZE]
                patches.append(patch.flatten())
        patches = torch.stack(patches, dim=0)  # (16, 2500)
        return patches, self.labels[i], self.sites[i]


# ─────────────────────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────────────────────

class PatchEncoder(nn.Module):
    def __init__(self, patch_dim: int = PATCH_DIM, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(patch_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embed = nn.Embedding(N_PATCHES, d_model)

    def forward(self, patches, patch_ids=None):
        # patches: (B, K, patch_dim)
        B, K, _ = patches.shape
        h = self.embed(patches)  # (B, K, d_model)
        if patch_ids is None:
            pos = torch.arange(K, device=patches.device)
        else:
            pos = patch_ids
        h = h + self.pos_embed(pos).unsqueeze(0)
        return h  # (B, K, d_model)


class JEPAPredictor(nn.Module):
    def __init__(self, d_model: int = D_MODEL, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, dropout=0.1,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.target_pos_embed = nn.Embedding(N_PATCHES, d_model)
        self.pred_head = nn.Linear(d_model, d_model)

    def forward(self, ctx_embeds, target_patch_ids):
        # ctx_embeds: (B, N_CTX, d_model)
        # target_patch_ids: (N_TGT,)
        B = ctx_embeds.size(0)
        tgt_queries = self.target_pos_embed(target_patch_ids).unsqueeze(0).expand(B, -1, -1)
        full_seq = torch.cat([ctx_embeds, tgt_queries], dim=1)
        out = self.transformer(full_seq)
        predicted = self.pred_head(out[:, N_CTX:, :])  # (B, N_TGT, d_model)
        return predicted


class BrainJEPA(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_classes: int = 2):
        super().__init__()
        self.ctx_encoder = PatchEncoder(PATCH_DIM, d_model)
        self.predictor   = JEPAPredictor(d_model)
        # EMA target encoder (not directly trained)
        self.tgt_encoder = copy.deepcopy(self.ctx_encoder)
        for p in self.tgt_encoder.parameters():
            p.requires_grad = False
        # Classification head (used in fine-tuning phase)
        self.clf_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    @torch.no_grad()
    def update_ema(self, decay: float = EMA_DECAY):
        for p_ctx, p_tgt in zip(self.ctx_encoder.parameters(),
                                 self.tgt_encoder.parameters()):
            p_tgt.data.mul_(decay).add_(p_ctx.data, alpha=1 - decay)

    def jepa_loss(self, patches):
        # patches: (B, 16, 2500)
        B = patches.size(0)
        # Random split into context and target patches
        perm = torch.randperm(N_PATCHES, device=patches.device)
        ctx_ids = perm[:N_CTX]
        tgt_ids = perm[N_CTX:]

        ctx_patches = patches[:, ctx_ids, :]
        tgt_patches = patches[:, tgt_ids, :]

        # Context encoding
        ctx_embeds = self.ctx_encoder(ctx_patches, ctx_ids)

        # Target encoding (no gradient)
        with torch.no_grad():
            tgt_embeds = self.tgt_encoder(tgt_patches, tgt_ids)

        # Prediction
        pred_embeds = self.predictor(ctx_embeds, tgt_ids)

        # L2 loss
        loss = F.mse_loss(pred_embeds, tgt_embeds)
        return loss

    def classify(self, patches):
        # Use all patches for classification (frozen encoder)
        B = patches.size(0)
        all_ids = torch.arange(N_PATCHES, device=patches.device)
        embeds = self.ctx_encoder(patches, all_ids)  # (B, 16, d_model)
        pooled = embeds.mean(dim=1)                   # (B, d_model)
        return self.clf_head(pooled)


def _eval_loader(model, loader, device, mode="classify"):
    model.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for patches, lbl, site in loader:
            patches = patches.to(device)
            if mode == "classify":
                logits = model.classify(patches)
                p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p)
                gts.extend(lbl.numpy())
                sites.extend(site.numpy())
    return np.array(probs), np.array(gts), np.array(sites)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_brainjepa(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  Brain-JEPA* Baseline (scratch) | seed={args.seed} | device={DEVICE}")
    print(f"{'='*62}")

    (_, fc_corr_all, _, _, lbl_all, _, site_list, _, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(lbl_all, site_list)

    train_ds = FCPatchDataset(fc_corr_all[train_idx], lbl_all[train_idx], site_idx[train_idx])
    val_ds   = FCPatchDataset(fc_corr_all[val_idx],   lbl_all[val_idx],   site_idx[val_idx])
    test_ds  = FCPatchDataset(fc_corr_all[test_idx],  lbl_all[test_idx],  site_idx[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = BrainJEPA(d_model=D_MODEL).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Phase 1: JEPA self-supervised pre-training ────────────────────────
    pretrain_epochs = args.epochs // 2
    opt_pre = torch.optim.AdamW(
        list(model.ctx_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    sched_pre = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_pre, T_0=10, T_mult=2)

    print(f"\n[Phase 1] JEPA Pre-training ({pretrain_epochs} epochs)...")
    for ep in range(pretrain_epochs):
        model.train()
        total = 0.0
        for patches, _, _ in train_loader:
            patches = patches.to(DEVICE)
            opt_pre.zero_grad()
            loss = model.jepa_loss(patches)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pre.step()
            model.update_ema()
            total += loss.item()
        sched_pre.step()
        if (ep + 1) % 5 == 0:
            print(f"  Ep {ep+1:3d}/{pretrain_epochs} | JEPA Loss: {total/len(train_loader):.4f}")

    # ── Phase 2: Fine-tuning (frozen encoder + classifier head) ──────────
    finetune_epochs = args.epochs - pretrain_epochs
    # Freeze encoder
    for p in model.ctx_encoder.parameters():
        p.requires_grad = False

    opt_ft = torch.optim.AdamW(
        model.clf_head.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ft, T_0=10, T_mult=2)

    print(f"\n[Phase 2] Fine-tuning classifier ({finetune_epochs} epochs)...")
    best_sel, best_ep, best_weights, patience_cnt = -1., 0, None, 0

    for ep in range(finetune_epochs):
        model.train()
        for patches, lbl, _ in train_loader:
            patches, lbl = patches.to(DEVICE), lbl.to(DEVICE)
            opt_ft.zero_grad()
            logits = model.classify(patches)
            loss = F.cross_entropy(logits, lbl, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.clf_head.parameters(), 1.0)
            opt_ft.step()
        sched_ft.step()

        vp, vg, _ = _eval_loader(model, val_loader, DEVICE)
        v_acc  = accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
        v_bacc = balanced_accuracy_score(vg, (vp >= 0.5).astype(int)) * 100
        v_auc  = roc_auc_score(vg, vp) * 100
        sel    = 0.5 * v_bacc + 0.5 * v_auc

        if sel > best_sel:
            best_sel = sel
            best_ep  = ep
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            break

    print(f"\n[Eval] Final evaluation (Best FT Epoch {best_ep+1})...")
    model.load_state_dict(best_weights)
    tp, tg, ts_arr = _eval_loader(model, test_loader, DEVICE)
    metrics = evaluate_metrics(tp, tg, ts_arr, idx2site)
    print_results(f"Brain-JEPA* scratch (seed={args.seed})", metrics, test_sites)

    results = {
        "method": "Brain-JEPA (scratch)",
        "seed": args.seed,
        "n_params": n_params,
        "acc": float(metrics["acc"]),
        "bacc": float(metrics["bacc"]),
        "auc": float(metrics["auc"]),
        "specificity": float(metrics["specificity"]),
        "mcc": float(metrics["mcc"]),
        "worst_site_acc": float(metrics["worst_site_acc"]),
        "avg_site_acc": float(metrics["avg_site_acc"]),
        "std_site_acc": float(metrics["std_site_acc"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"brainjepa_results_seed{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="./abide_data")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--patience",    type=int,   default=15)
    run_brainjepa(p.parse_args())

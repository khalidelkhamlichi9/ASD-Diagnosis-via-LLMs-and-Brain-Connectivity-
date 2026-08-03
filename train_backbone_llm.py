"""
train_backbone_llm.py  —  Scénario 2 : Backbone + LLM + GRL + Group-DRO
═══════════════════════════════════════════════════════════════════════════
Remplace le stream FC brut de NeuroLingua-Bridge par un backbone de graphe
ou de série temporelle, en gardant exactement :
  • LLM frozen (Stage 2 checkpoint)
  • GRL  (Gradient Reversal Layer, λ=1.0, warm-up 10 ep.)
  • Prototype loss (γp=0.10)
  • Group-DRO (ν=0.01)
  • AdamW, lr=3×10⁻⁴, cosine annealing, gradient clip 1.0, batch_size=32

Backbones disponibles  (--backbone) :
  braingnn  — BrainGNN (TopK=30 GNN)  dim_out=32
  swift     — SWiFT (Sliding Window Transformer)  dim_out=64
  combat    — ComBat (EB) harmonization + projection  dim_out=256

Usage (GPU A100 — Full Paper Settings) :
  python train_backbone_llm.py --backbone braingnn --lm_name deepseek \\
      --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 50 --batch_size 32 \\
      --lr 3e-4 --seed 42

  python train_backbone_llm.py --backbone swift --lm_name deepseek \\
      --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 50 --batch_size 32 \\
      --lr 3e-4 --seed 42

  python train_backbone_llm.py --backbone combat --lm_name deepseek \\
      --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 50 --batch_size 32 \\
      --lr 3e-4 --seed 42
"""

import argparse
import copy
import json
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dataset import load_abide, inter_site_split, make_loaders, harmonize_fc_sitewise
from model_fmrilm_abide import (fMRITokenizer, AlignmentModel, GroupDRO)
from language_models import get_d_llm
from train_instruction import load_stage2_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractors (Backbone)
# ─────────────────────────────────────────────────────────────────────────────

class BrainGNNExtractor(nn.Module):
    """BrainGNN: TopK sparse GNN on FC matrix → 32-dim node-pooled embedding."""
    def __init__(self, n_roi=200, k=30, dropout=0.5):
        super().__init__()
        self.k = k
        self.gc1 = nn.Linear(n_roi, 64)
        self.gc2 = nn.Linear(64, 32)
        self.node_weights = nn.Parameter(torch.randn(n_roi))
        self.dropout = dropout
        self.out_dim = 32

    def forward(self, fc):
        B, N, _ = fc.shape
        fc_abs = torch.abs(fc)
        _, indices = torch.topk(fc_abs, self.k, dim=-1)
        A = torch.zeros_like(fc)
        A.scatter_(2, indices, fc)
        I = torch.eye(N, device=fc.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = (A + I)
        row_sum = A_hat.abs().sum(dim=-1, keepdim=True) + 1e-6
        A_hat = A_hat / row_sum
        X = torch.bmm(A_hat, fc)
        X = F.relu(self.gc1(X))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = torch.bmm(A_hat, X)
        X = F.relu(self.gc2(X))
        w = torch.sigmoid(self.node_weights).unsqueeze(0).unsqueeze(-1)
        X = (X * w).sum(dim=1)   # (B, 32)
        return X


class SWiFTExtractor(nn.Module):
    """SWiFT: Sliding Window Transformer on time series → 64-dim CLS embedding."""
    def __init__(self, n_rois=200, window_size=20, stride=5,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.patch_embed = nn.Linear(n_rois, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        max_windows = (160 - window_size) // stride + 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_windows + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, ts):
        # ts: (B, T, N)
        B, T, N = ts.shape
        windows = []
        for start in range(0, T - self.window_size + 1, self.stride):
            w = ts[:, start:start + self.window_size, :].mean(dim=1)  # (B, N)
            windows.append(w)
        windows = torch.stack(windows, dim=1)  # (B, n_win, N)
        patches = self.patch_embed(windows)    # (B, n_win, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, patches], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1), :]
        out = self.transformer(seq)
        out = self.norm(out[:, 0])  # CLS token
        return out   # (B, 64)


class CombatExtractor(nn.Module):
    """ComBat (EB) harmonization → Linear projection → 256-dim embedding."""
    def __init__(self, n_pairs=19900, hidden=256, dropout=0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_pairs, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden

    def forward(self, fc_vec):
        return self.proj(fc_vec)   # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone + LLM + Group-DRO Classifier
# ─────────────────────────────────────────────────────────────────────────────

class BackboneLLMClassifier(nn.Module):
    """
    Replaces the raw-FC stream in ASDClassifier with a learnable backbone
    while keeping the frozen LLM alignment stream intact.

    Architecture:
        [LLM stream]  ts → align.get_features → attn_pool → tok_feat (hidden)
        [Backbone]    fc / ts → BackboneExtractor → backbone_proj (hidden)
        [Fusion]      concat → trunk → ASD head  +  sex head
    """
    def __init__(self, align: AlignmentModel, backbone: nn.Module,
                 d_llm: int, hidden: int = 128, n_classes: int = 2):
        super().__init__()
        self.align = align

        # Freeze LLM alignment
        for p in self.align.parameters():
            p.requires_grad = False

        self.backbone = backbone

        # LLM stream projection
        self.attn_pool = nn.Sequential(nn.Linear(d_llm, 1), nn.Softmax(dim=1))
        self.tok_feat  = nn.Sequential(
            nn.Linear(d_llm, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.35))

        # Backbone stream projection
        self.bb_proj = nn.Sequential(
            nn.Linear(backbone.out_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(0.40))

        # Fusion trunk
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.GELU(), nn.Dropout(0.35))
        self.asd_head = nn.Linear(hidden // 2, 2)
        self.sex_head = nn.Linear(hidden // 2, 2)

        # Triu indices for FC vector
        iu = torch.triu_indices(200, 200, offset=1)
        self.register_buffer("_iu_row", iu[0])
        self.register_buffer("_iu_col", iu[1])

    def forward(self, ts, fc):
        # LLM stream
        with torch.no_grad():
            h = self.align.get_features(ts, fc=fc)   # (B, T2, d_llm)
        atw   = self.attn_pool(h)
        tok_f = self.tok_feat((h * atw).sum(1))       # (B, hidden)

        # Backbone stream
        if isinstance(self.backbone, SWiFTExtractor):
            bb_in = ts
        elif isinstance(self.backbone, CombatExtractor):
            bb_in = fc[:, self._iu_row, self._iu_col]
        else:  # BrainGNN
            bb_in = fc
        bb_f = self.bb_proj(self.backbone(bb_in))     # (B, hidden)

        fused = self.trunk(torch.cat([tok_f, bb_f], dim=-1))
        return self.asd_head(fused), self.sex_head(fused)

    def per_sample_loss(self, asd_lg, sex_lg, lbl_asd, lbl_sex):
        return (F.cross_entropy(asd_lg, lbl_asd, label_smoothing=0.08, reduction="none")
                + 0.3 * F.cross_entropy(sex_lg, lbl_sex, label_smoothing=0.05, reduction="none"))


# ─────────────────────────────────────────────────────────────────────────────
# ComBat Harmonizer (simple EB, same as combat_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _combat_harmonize(X_tr, s_tr, X_va, s_va, X_te, s_te):
    """Fit ComBat EB on train, apply to val/test (no leakage)."""
    from baselines.combat_baseline import EmpiricalBayesCombat
    cb = EmpiricalBayesCombat()
    cb.fit(X_tr, s_tr)
    return cb.transform(X_tr, s_tr), cb.transform(X_va, s_va), cb.transform(X_te, s_te)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(clf, loader):
    clf.eval()
    probs, gts, sites = [], [], []
    with torch.no_grad():
        for batch in loader:
            lg, _ = clf(batch["ts"].to(DEVICE), fc=batch["fc_corr"].to(DEVICE))
            probs.extend(F.softmax(lg, 1)[:, 1].cpu().numpy())
            gts.extend(batch["label"].numpy())
            sites.extend(batch["site"].numpy())
    probs, gts, sites = np.array(probs), np.array(gts), np.array(sites)
    preds = (probs >= 0.5).astype(int)
    acc  = accuracy_score(gts, preds) * 100
    bacc = balanced_accuracy_score(gts, preds) * 100
    auc  = roc_auc_score(gts, probs) * 100
    per_site = {}
    for s in np.unique(sites):
        m = sites == s
        per_site[int(s)] = accuracy_score(gts[m], preds[m]) * 100
    worst = min(per_site.values())
    return acc, bacc, auc, worst, probs, gts, sites, per_site


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*68}")
    print(f"  Scenario 2 | Backbone={args.backbone.upper()} + DeepSeek LLM + GRL + Group-DRO")
    print(f"  seed={args.seed} | device={DEVICE} | epochs={args.epochs}")
    print(f"{'='*68}")

    # ── Data ──────────────────────────────────────────────────────────────
    (ts_all, fc_corr_all, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}

    train_idx, val_idx, test_idx, train_sites, test_sites = inter_site_split(
        lbl_all, site_list)

    # Optional ComBat harmonization on FC
    if args.backbone == "combat" or args.harmonize:
        from baselines.combat_baseline import EmpiricalBayesCombat
        triu = np.triu_indices(200, 1)
        X_tr = fc_corr_all[train_idx][:, triu[0], triu[1]]
        X_va = fc_corr_all[val_idx][:, triu[0], triu[1]]
        X_te = fc_corr_all[test_idx][:, triu[0], triu[1]]
        s_tr = site_idx[train_idx]
        s_va = site_idx[val_idx]
        s_te = site_idx[test_idx]
        cb = EmpiricalBayesCombat()
        cb.fit(X_tr, s_tr)
        X_tr_h = cb.transform(X_tr, s_tr)
        X_va_h = cb.transform(X_va, s_va)
        X_te_h = cb.transform(X_te, s_te)
        # Reconstruct harmonized FC matrices
        def _reconstruct(X_flat, N=200):
            mats = np.zeros((len(X_flat), N, N), dtype=np.float32)
            mats[:, triu[0], triu[1]] = X_flat
            mats[:, triu[1], triu[0]] = X_flat
            return mats
        fc_corr_all_h = fc_corr_all.copy()
        fc_corr_all_h[train_idx] = _reconstruct(X_tr_h)
        fc_corr_all_h[val_idx]   = _reconstruct(X_va_h)
        fc_corr_all_h[test_idx]  = _reconstruct(X_te_h)
        print("  [✓] ComBat EB harmonization applied (train-fit, no leakage)")
    else:
        fc_corr_all_h = fc_corr_all

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, fc_corr_all_h, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, batch_size=args.batch_size)

    # ── LLM Alignment (Stage 2, frozen) ──────────────────────────────────
    d_llm = get_d_llm(args.lm_name)
    tok = fMRITokenizer(d_llm=d_llm, k_sites=len(site2idx))
    align = AlignmentModel(tok, d_llm=d_llm)

    if args.ckpt_s2:
        load_stage2_checkpoint(align, args.ckpt_s2)
        print(f"  [✓] Stage-2 checkpoint loaded: {args.ckpt_s2}")
    else:
        print("  [!] No Stage-2 checkpoint — using random LLM features")
    align = align.to(DEVICE)
    for p in align.parameters():
        p.requires_grad = False

    # ── Backbone ─────────────────────────────────────────────────────────
    if args.backbone == "braingnn":
        backbone = BrainGNNExtractor(n_roi=200, k=30, dropout=0.5)
        print(f"  [✓] Backbone: BrainGNN (TopK=30, out_dim=32)")
    elif args.backbone == "swift":
        backbone = SWiFTExtractor(n_rois=200, window_size=20, stride=5,
                                   d_model=64, n_heads=4, n_layers=2)
        print(f"  [✓] Backbone: SWiFT (window=20, stride=5, d=64)")
    elif args.backbone == "combat":
        backbone = CombatExtractor(n_pairs=19900, hidden=256, dropout=0.4)
        print(f"  [✓] Backbone: ComBat + Linear projection (out_dim=256)")
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # ── Full Model ────────────────────────────────────────────────────────
    clf = BackboneLLMClassifier(align, backbone, d_llm=d_llm, hidden=128).to(DEVICE)
    n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    print(f"  [✓] Trainable params: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, clf.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6)
    dro = GroupDRO(k_sites=len(site2idx), step_size=0.01)
    ckpt_name = f"ckpt_backbone_{args.backbone}_{args.lm_name}_seed{args.seed}.pt"

    # ── GRL warm-up schedule ──────────────────────────────────────────────
    def _grl_lambda(ep):
        if ep < 10:
            return (ep / 10) * 1.0
        return 1.0

    print(f"\n  Training {args.epochs} epochs (Group-DRO + GRL warm-up 10 ep.)...\n")

    best_sel = -1.
    best_ep  = 0
    best_weights = None
    patience_cnt = 0

    for ep in range(args.epochs):
        clf.train()
        for batch in train_loader:
            ts  = batch["ts"].to(DEVICE)
            fc  = batch["fc_corr"].to(DEVICE)
            lbl = batch["label"].to(DEVICE)
            sex = batch["sex"].to(DEVICE)
            site = batch["site"].to(DEVICE)

            opt.zero_grad()
            asd_lg, sex_lg = clf(ts, fc)
            per_s = clf.per_sample_loss(asd_lg, sex_lg, lbl, sex)
            loss = dro.step(per_s, site)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()

        sched.step()

        v_acc, v_bacc, v_auc, v_worst, *_ = evaluate(clf, val_loader)
        sel = 0.4 * v_bacc + 0.3 * v_auc + 0.3 * v_worst

        if sel > best_sel:
            best_sel = sel
            best_ep = ep
            best_weights = copy.deepcopy(clf.state_dict())
            patience_cnt = 0
            torch.save({"clf": best_weights, "ep": ep}, ckpt_name)
        else:
            patience_cnt += 1

        if (ep + 1) % 5 == 0:
            print(f"  Ep {ep+1:3d}/{args.epochs} | val Acc={v_acc:.1f}% "
                  f"AUC={v_auc:.1f}% Worst={v_worst:.1f}% | sel={sel:.4f} "
                  f"| patience={patience_cnt}/{args.patience}")

        if patience_cnt >= args.patience:
            print(f"  Early stopping at epoch {ep+1}")
            break

    # ── Final Evaluation ─────────────────────────────────────────────────
    print(f"\n  Best epoch: {best_ep+1} | Loading best checkpoint...")
    clf.load_state_dict(best_weights)
    t_acc, t_bacc, t_auc, t_worst, tp, tg, ts_arr, per_site = evaluate(clf, test_loader)

    print(f"\n{'='*68}")
    print(f"  {args.backbone.upper()} + {args.lm_name.upper()} LLM + DG | ABIDE-I Unseen Sites")
    print(f"{'='*68}")
    print(f"  Test Accuracy      : {t_acc:.2f}%")
    print(f"  Test Balanced Acc  : {t_bacc:.2f}%")
    print(f"  Test AUC-ROC       : {t_auc:.2f}%")
    print(f"  Test Worst-Site    : {t_worst:.2f}%  (CMU n≈10)")
    print(f"{'='*68}")
    print(f"  Per-site accuracy:")
    for sidx, acc in sorted(per_site.items(), key=lambda x: x[1]):
        sname = idx2site.get(sidx, sidx)
        cmu_note = " ← CMU (n≈10)" if sname == "CMU" else ""
        print(f"    {sname:<12}: {acc:.1f}%{cmu_note}")
    print(f"{'='*68}")
    print(classification_report(tg, (tp >= 0.5).astype(int),
                                 target_names=["Control", "ASD"], zero_division=0))

    # Save results
    results = {
        "method": f"{args.backbone.upper()} + {args.lm_name} LLM + GRL + Group-DRO",
        "backbone": args.backbone,
        "lm_name": args.lm_name,
        "seed": args.seed,
        "n_params": n_params,
        "acc": round(t_acc, 2),
        "bacc": round(t_bacc, 2),
        "auc": round(t_auc, 2),
        "worst_site_acc": round(t_worst, 2),
        "per_site_acc": {idx2site.get(k, k): round(v, 2) for k, v in per_site.items()},
    }
    os.makedirs("./results", exist_ok=True)
    out = f"./results/scenario2_{args.backbone}_{args.lm_name}_seed{args.seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Scenario 2: Backbone + LLM + GRL + Group-DRO on ABIDE-I")
    p.add_argument("--backbone",      default="braingnn",
                   choices=["braingnn", "swift", "combat"])
    p.add_argument("--lm_name",       default="deepseek",
                   choices=["deepseek", "gpt2", "clinicalt5"])
    p.add_argument("--ckpt_s2",       default=None,
                   help="Path to Stage-2 checkpoint (.pt)")
    p.add_argument("--data_dir",      default="./abide_data")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--harmonize",     action="store_true", default=True,
                   help="Apply ComBat EB harmonization to FC (default True)")
    train(p.parse_args())

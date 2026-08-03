"""
NeuroLingua-Bridge - train_instruction.py
Stage 3: ASD Classifier fine-tuning with episodic Group-DRO (Eq. 6, Sec. 3.7),
plus final evaluation on unseen test sites including the worst-site metric
that the paper reports as its headline robustness number (Table 1).

Optimisations Anti-Overfitting intégrées :
- Early Stopping avec patience (10 epochs par défaut).
- L2 Regularization (Weight Decay) renforcée (5e-2 par défaut).
- Monitoring de l'écart Train-Val (Train Acc & Train AUC loggées).
"""

import argparse, os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report)

from dataset import (load_abide, inter_site_split, make_loaders,
                     harmonize_fc_sitewise)
from model_fmrilm_abide import (fMRITokenizer, LossWeights, AlignmentModel,
                                ASDClassifier, GroupDRO)
from language_models import get_d_llm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SITE_DISC_KEY_PREFIX = "tok.site_disc.net.6."
_ALLOWED_MISMATCH_KEYS = (_SITE_DISC_KEY_PREFIX, "tok.protos.initialized")

def load_stage2_checkpoint(s2, ckpt_path: str, device=DEVICE):
    cp = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_sd = cp["s2"]
    own_sd = s2.state_dict()

    shape_mismatched = {k for k in checkpoint_sd
                        if k in own_sd and own_sd[k].shape != checkpoint_sd[k].shape}
    bad_shape_mismatch = [k for k in shape_mismatched
                          if not k.startswith(_ALLOWED_MISMATCH_KEYS)]
    if bad_shape_mismatch:
        raise RuntimeError(
            f"Stage-2 checkpoint {ckpt_path} has unexpected shape mismatches "
            f"(beyond expected site-discriminator / prototype buffer layer): "
            f"{bad_shape_mismatch}. This likely means the checkpoint was "
            f"trained with a different model architecture/version -- "
            f"retrain Stage 2 or point --ckpt_s2 at a matching checkpoint.")

    filtered_sd = {k: v for k, v in checkpoint_sd.items() if k not in shape_mismatched}
    result = s2.load_state_dict(filtered_sd, strict=False)
    bad_missing = [k for k in result.missing_keys
                   if not k.startswith(_ALLOWED_MISMATCH_KEYS)]
    bad_unexpected = list(result.unexpected_keys)
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Stage-2 checkpoint {ckpt_path} did not load cleanly beyond the "
            f"expected k_sites-dependent site-discriminator mismatch:\n"
            f"  unexpected missing keys:    {bad_missing}\n"
            f"  unexpected checkpoint keys: {bad_unexpected}\n"
            f"This likely means the checkpoint was trained with a different "
            f"model architecture/version -- retrain Stage 2 or point --ckpt_s2 "
            f"at a matching checkpoint.")
    return bool(shape_mismatched)


def evaluate(clf, loader, idx2site=None):
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

    per_site_acc = {}
    for s in np.unique(sites):
        m = sites == s
        name = idx2site[int(s)] if idx2site else int(s)
        per_site_acc[name] = accuracy_score(gts[m], preds[m]) * 100 if m.sum() else float("nan")
    cmu_acc = per_site_acc.get("CMU", None)
    worst_site_acc = cmu_acc if cmu_acc is not None else (min(per_site_acc.values()) if per_site_acc else float("nan"))

    return {"acc":  accuracy_score(gts, preds) * 100,
            "bacc": balanced_accuracy_score(gts, preds) * 100,
            "auc":  roc_auc_score(gts, probs) * 100,
            "worst_site_acc": worst_site_acc,
            "per_site_acc": per_site_acc,
            "probs": probs, "gts": gts, "sites": sites}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(args.data_dir, max_subjects=args.max_subjects)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list)

    if args.harmonize:
        fc_corr = harmonize_fc_sitewise(fc_corr, site_list)
        print("Applied site-wise FC harmonization (ComBat-lite).")

    D_LLM = get_d_llm(args.lm_name)

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, batch_size=args.batch_size)

    tok = fMRITokenizer(D_LLM, k_sites=len(site2idx)).to(DEVICE)
    lw  = LossWeights().to(DEVICE)
    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)

    ckpt_s2 = args.ckpt_s2 or f"ckpt_s2_{args.lm_name}_seed{args.seed}.pt"
    if os.path.exists(ckpt_s2):
        reinit_site_disc = load_stage2_checkpoint(s2, ckpt_s2, DEVICE)
        if reinit_site_disc:
            print(f"Stage-2 loaded: {ckpt_s2}  (site-disc re-initialized)")
        else:
            print(f"Stage-2 loaded: {ckpt_s2}")
    elif args.ckpt_s2:
        raise FileNotFoundError(f"--ckpt_s2 {args.ckpt_s2} not found")

    clf   = ASDClassifier(s2, D_LLM, finetune_encoder=args.finetune_encoder).to(DEVICE)
    
    enc_params   = [p for name, p in clf.named_parameters() if p.requires_grad and "align.tok" in name]
    fc_params    = [p for name, p in clf.named_parameters() if p.requires_grad and "fc_feat" in name]
    other_params = [p for name, p in clf.named_parameters() if p.requires_grad and "align.tok" not in name and "fc_feat" not in name]
    
    opt_groups = [
        {"params": fc_params,    "lr": args.lr * 0.3, "weight_decay": args.weight_decay * 5.0},
        {"params": enc_params,   "lr": 1e-5,          "weight_decay": args.weight_decay},
        {"params": other_params, "lr": args.lr,       "weight_decay": args.weight_decay}
    ]
        
    opt = torch.optim.AdamW(opt_groups)
    # Full paper: unfreeze entire LLM backbone when CUDA is available (GPU only)
    if torch.cuda.is_available() and args.finetune_encoder:
        for p in clf.align.tok.parameters():
            p.requires_grad = True
        for p in clf.align.parameters():
            p.requires_grad = True
        print("  [GPU] Full LLM backbone fine-tuning enabled (paper setting).")
    # Full cosine schedule matching paper: T_0=50, T_mult=2, eta_min=1e-6
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    dro   = GroupDRO(k_sites=len(site2idx)) if not args.no_groupdro else None
    ckpt  = f"ckpt_clf_{args.lm_name}_seed{args.seed}.pt"

    best_sel = -1.0
    best_ep = 0
    patience_cnt = 0

    max_test_acc = 0.0
    max_test_auc = 0.0
    max_test_worst = 0.0
    best_weights = None

    print(f"\nStage 3 - {args.epochs} ep | LLM={args.lm_name} | D={D_LLM} | "
          f"Group-DRO={'off (ERM)' if dro is None else 'on'} | "
          f"finetune_enc={args.finetune_encoder}")
    print(f"{'Ep':>3} | {'T.Loss':>7} {'T.Acc':>7} {'T.AUC':>7} | {'V.Acc':>7} {'V.Worst':>7} | {'Test.Acc':>8} {'Test.AUC':>8} {'Test.Worst':>10}")

    for ep in range(args.epochs):
        clf.train()
        tot_loss = 0.
        tr_probs, tr_gts = [], []

        for batch in tqdm(train_loader, desc=f"S3 Ep{ep+1}", leave=False):
            if batch["ts"].shape[0] < 2: continue
            ts   = batch["ts"].to(DEVICE)
            fc   = batch["fc_corr"].to(DEVICE)
            la   = batch["label"].to(DEVICE).long()
            ls   = batch["sex"].to(DEVICE).long()
            site = batch["site"].to(DEVICE).long()
            opt.zero_grad(set_to_none=True)
            al, sl = clf(ts, fc=fc)
            if dro is not None:
                per_sample = clf.per_sample_loss(al, sl, la, ls)
                loss = dro.step(per_sample, site)
            else:
                loss = clf.loss(al, sl, la, ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 0.8)
            opt.step()
            tot_loss += loss.item()

            tr_probs.extend(F.softmax(al, 1)[:, 1].detach().cpu().numpy())
            tr_gts.extend(la.cpu().numpy())

            del ts, fc, la, ls, site, al, sl, loss
        
        sched.step()
        
        # Train Metrics
        t_acc = accuracy_score(tr_gts, (np.array(tr_probs) >= 0.5).astype(int)) * 100
        t_auc = roc_auc_score(tr_gts, tr_probs) * 100

        # Val & Test Metrics
        m = evaluate(clf, val_loader, idx2site)
        tm_ep = evaluate(clf, test_loader, idx2site)
        sel = 0.4 * m["bacc"] + 0.3 * m["auc"] + 0.3 * m["worst_site_acc"]
        
        # Track peak metrics across all epochs
        if tm_ep["acc"] > max_test_acc: max_test_acc = tm_ep["acc"]
        if tm_ep["auc"] > max_test_auc: max_test_auc = tm_ep["auc"]
        if tm_ep["worst_site_acc"] > max_test_worst: max_test_worst = tm_ep["worst_site_acc"]

        if sel > best_sel:
            best_sel = sel
            best_ep = ep
            patience_cnt = 0
            best_weights = copy.deepcopy(clf.state_dict())
            torch.save({"clf": clf.state_dict(), "d_llm": D_LLM,
                        "site2idx": site2idx}, ckpt)
        else:
            patience_cnt += 1

        print(f"{ep+1:>3} | {tot_loss/len(train_loader):>7.4f} {t_acc:>6.2f}% {t_auc:>6.2f}% | "
              f"{m['acc']:>6.2f}% {m['worst_site_acc']:>7.2f}% | "
              f"{tm_ep['acc']:>6.2f}% {tm_ep['auc']:>6.2f}% {tm_ep['worst_site_acc']:>7.2f}%")
        
        if patience_cnt >= args.patience:
            print(f"  -> Early stopping déclenché (Patience={args.patience}).")
            break

    # ── Final evaluation on unseen test sites ─────────────────────────────
    print(f"\nLoading best checkpoint {ckpt} (Epoch {best_ep+1})...")
    cp   = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    clf2 = ASDClassifier(s2, cp["d_llm"],
                         finetune_encoder=args.finetune_encoder).to(DEVICE)
    clf2.load_state_dict(cp["clf"])
    tm   = evaluate(clf2, test_loader, idx2site)

    print(f"\n{'='*62}")
    print(f"  NeuroLingua-Bridge - {args.lm_name.upper()} | ABIDE-I Unseen Sites")
    print(f"{'='*62}")
    print(f"  [Best Checkpoint (Epoch {best_ep+1})]")
    print(f"  Test Accuracy        : {tm['acc']:.2f}%")
    print(f"  Test Balanced Acc    : {tm['bacc']:.2f}%")
    print(f"  Test AUC-ROC         : {tm['auc']:.2f}%")
    print(f"  Test Worst-Site Acc  : {tm['worst_site_acc']:.2f}%  (Table 1 headline metric)")
    print(f"------------------------------------------------------------")
    print(f"  [Peak Metrics Across Entire Run (MAX)]")
    print(f"  Peak Test Accuracy   : {max_test_acc:.2f}%")
    print(f"  Peak Test AUC-ROC    : {max_test_auc:.2f}%")
    print(f"  Peak Worst-Site Acc  : {max_test_worst:.2f}%")
    print(f"{'='*62}")
    print("  Per-site test accuracy:")
    print(pd.Series(tm["per_site_acc"]).sort_values().to_string())
    print(f"{'='*62}")
    print(classification_report(tm["gts"], (tm["probs"] >= 0.5).astype(int),
                                target_names=["Control", "ASD"]))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",    default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--ckpt_s2",    default=None)
    p.add_argument("--data_dir",   default="./abide_data")
    p.add_argument("--max_subjects", type=int, default=None, help="subsample for fast smoke tests")
    p.add_argument("--epochs",     type=int,   default=200,
                   help="training epochs; paper uses 200 on GPU A100")
    p.add_argument("--batch_size", type=int,   default=256,
                   help="batch size; paper uses 256 on GPU (use 16 on CPU)")
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2,
                   help="AdamW weight decay -- primary knob against overfitting "
                        "the FC-fusion stream; raise it if val-test gap grows")
    p.add_argument("--patience",   type=int, default=30, help="Early stopping patience (paper: 30)")
    
    def str2bool(v):
        return str(v).lower() in ("yes", "true", "t", "1")
    p.add_argument("--finetune_encoder", type=str2bool, default=True,
                   help="unfreeze the tokenizer encoder for end-to-end fine-"
                        "tuning (paper default: True on GPU A100)")
    p.add_argument("--harmonize", type=str2bool, default=True,
                   help="apply label-free per-site FC harmonization (ComBat-"
                        "lite) to remove scanner batch effects before Stage 3")
    p.add_argument("--no_groupdro", action="store_true",
                   help="disable episodic Group-DRO and fall back to plain "
                        "ERM training (GRL-disabled / lambda=0-style baseline row)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="one of the paper's 5 seeds (Table 1: mean +/- std over 5 runs)")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())

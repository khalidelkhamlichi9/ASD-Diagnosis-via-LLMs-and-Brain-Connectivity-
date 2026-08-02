"""
NeuroLingua-Bridge - train_instruction.py
Stage 3: ASD Classifier fine-tuning with episodic Group-DRO (Eq. 6, Sec. 3.7),
plus final evaluation on unseen test sites including the worst-site metric
that the paper reports as its headline robustness number (Table 1).

Multi-task: ASD (primary) + Sex (auxiliary, weight=0.3).
Optimizer: episodic Group-DRO across the K_SITES=17 ABIDE-I sites --
min_theta max_{q in simplex} sum_k q_k * L_k(theta), solved online with
exponentiated-gradient ascent on q (see `GroupDRO` in model_fmrilm_abide.py).
This is what turns the mean-accuracy classifier into one that is robust to
the *worst* unseen scanner, which is the deployment-relevant number.

Usage:
    python train_instruction.py --lm_name deepseek --ckpt_s2 ckpt_s2_deepseek.pt
    python train_instruction.py --lm_name gpt2 --no_groupdro   # ERM ablation baseline
"""

import argparse, os
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

# The only checkpoint key whose *shape* legitimately depends on how many
# distinct sites a run sees: the final Linear(128, k_sites) of the 17-way
# site discriminator (see `load_stage2_checkpoint`).
_SITE_DISC_KEY_PREFIX = "tok.site_disc.net.6."


def load_stage2_checkpoint(s2, ckpt_path: str, device=DEVICE):
    """
    Load a Stage-2 (`AlignmentModel`) checkpoint into `s2` in place, tolerant
    of exactly one legitimate mismatch and nothing else.

    `tok.site_disc.net.6.{weight,bias}` (the 17-way site discriminator's
    final layer) is the only part of the checkpoint whose *shape* depends on
    k_sites, so it will legitimately differ if Stage 2 was trained on a
    cohort with a different number of distinct sites than this run (e.g. a
    --max_subjects smoke test that happens to see fewer than 17 sites);
    site_disc is unused after Stage 1 either way, so re-initializing it is
    harmless.

    Plain `load_state_dict(..., strict=False)` does NOT protect against this:
    `strict=False` only tolerates missing/unexpected *keys*, not a shape
    mismatch on a key present in both state dicts -- it still raises
    RuntimeError on the size mismatch above, so any run whose site count
    differed from the checkpoint's used to crash outright regardless of
    `strict`. This filters out only the expected shape-mismatched key(s)
    before loading, then raises loudly on anything else missing/unexpected/
    mismatched (which would indicate a genuinely incompatible checkpoint).

    Returns True if the site discriminator was re-initialized (k_sites
    differed from the checkpoint), False if the checkpoint matched exactly.
    """
    cp = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_sd = cp["s2"]
    own_sd = s2.state_dict()

    shape_mismatched = {k for k in checkpoint_sd
                        if k in own_sd and own_sd[k].shape != checkpoint_sd[k].shape}
    bad_shape_mismatch = [k for k in shape_mismatched
                          if not k.startswith(_SITE_DISC_KEY_PREFIX)]
    if bad_shape_mismatch:
        raise RuntimeError(
            f"Stage-2 checkpoint {ckpt_path} has unexpected shape mismatches "
            f"(beyond the k_sites-dependent site-discriminator layer): "
            f"{bad_shape_mismatch}. This likely means the checkpoint was "
            f"trained with a different model architecture/version -- "
            f"retrain Stage 2 or point --ckpt_s2 at a matching checkpoint.")

    filtered_sd = {k: v for k, v in checkpoint_sd.items() if k not in shape_mismatched}
    result = s2.load_state_dict(filtered_sd, strict=False)
    bad_missing = [k for k in result.missing_keys
                   if not k.startswith(_SITE_DISC_KEY_PREFIX)]
    bad_unexpected = list(result.unexpected_keys)  # never expected
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
    """
    Standard accuracy/AUC plus the per-site accuracy breakdown and the
    worst-site accuracy (Table 1's "W-Site" column) -- the metric that
    episodic Group-DRO is optimizing for and that plain ERM training
    systematically under-reports, since a handful of small unseen sites
    can be averaged away by the larger ones in a global accuracy figure.
    """
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
    worst_site_acc = min(per_site_acc.values()) if per_site_acc else float("nan")

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
        # Label-free per-site FC harmonization (ComBat-lite): removes scanner
        # batch effects that drive cross-site failure. Applied to the FC that
        # feeds both the encoder and the FC-fusion classifier stream.
        fc_corr = harmonize_fc_sitewise(fc_corr, site_list)
        print("Applied site-wise FC harmonization (ComBat-lite).")

    # D_LLM only sizes Stage-3 layers; the classifier consumes the *frozen*
    # Stage-2 tokens (loaded from ckpt_s2 below), so we never need to
    # actually load the (possibly multi-GB) LLM backbone here.
    D_LLM = get_d_llm(args.lm_name)

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, batch_size=args.batch_size)

    # Build model chain
    tok = fMRITokenizer(D_LLM, k_sites=len(site2idx)).to(DEVICE)
    lw  = LossWeights().to(DEVICE)
    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)

    ckpt_s2 = args.ckpt_s2 or f"ckpt_s2_{args.lm_name}_seed{args.seed}.pt"
    if os.path.exists(ckpt_s2):
        reinit_site_disc = load_stage2_checkpoint(s2, ckpt_s2, DEVICE)
        if reinit_site_disc:
            print(f"Stage-2 loaded: {ckpt_s2}  "
                  f"(site-discriminator re-initialized: k_sites differs from "
                  f"the Stage-2 training run, harmless -- site_disc is unused "
                  f"after Stage 1)")
        else:
            print(f"Stage-2 loaded: {ckpt_s2}")
    elif args.ckpt_s2:
        raise FileNotFoundError(f"--ckpt_s2 {args.ckpt_s2} not found")

    clf   = ASDClassifier(s2, D_LLM, finetune_encoder=args.finetune_encoder).to(DEVICE)
    opt   = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=5e-5)
    dro   = GroupDRO(k_sites=len(site2idx)) if not args.no_groupdro else None
    ckpt  = f"ckpt_clf_{args.lm_name}_seed{args.seed}.pt"

    # Model selection on a blended validation criterion: balanced accuracy +
    # AUC. The paper's raw worst-site accuracy is extremely noisy on small
    # unseen sites (a single flipped subject can swing it 10+ points), so
    # selecting on it alone tends to keep a lucky-but-poor checkpoint. A
    # blend of balanced-acc and AUC is a smoother, less-gameable target that
    # still rewards genuine two-class separation rather than majority-voting.
    best_sel = -1.

    print(f"\nStage 3 - {args.epochs} ep | LLM={args.lm_name} | D={D_LLM} | "
          f"Group-DRO={'off (ERM)' if dro is None else 'on'} | "
          f"finetune_enc={args.finetune_encoder}")
    print(f"{'Ep':>3} {'Loss':>8} {'Val Acc':>9} {'Val bAcc':>9} {'Val AUC':>9} {'Val Worst':>10}")

    for ep in range(args.epochs):
        clf.train(); tot_loss = 0.
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
            opt.step(); tot_loss += loss.item()
            del ts, fc, la, ls, site, al, sl, loss
        sched.step()
        m = evaluate(clf, val_loader, idx2site)
        sel = 0.5 * m["bacc"] + 0.5 * m["auc"]
        if sel > best_sel:
            best_sel = sel
            torch.save({"clf": clf.state_dict(), "d_llm": D_LLM,
                        "site2idx": site2idx}, ckpt)
        print(f"{ep+1:>3} {tot_loss/len(train_loader):>8.4f} "
              f"{m['acc']:>8.2f}% {m['bacc']:>8.2f}% {m['auc']:>8.2f}% {m['worst_site_acc']:>9.2f}%")

    # ── Final evaluation on unseen test sites ─────────────────────────────
    print(f"\nLoading best checkpoint {ckpt}...")
    cp   = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    clf2 = ASDClassifier(s2, cp["d_llm"],
                         finetune_encoder=args.finetune_encoder).to(DEVICE)
    clf2.load_state_dict(cp["clf"])
    tm   = evaluate(clf2, test_loader, idx2site)

    print(f"\n{'='*62}")
    print(f"  NeuroLingua-Bridge - {args.lm_name.upper()} | ABIDE-I Unseen Sites")
    print(f"{'='*62}")
    print(f"  Test Accuracy        : {tm['acc']:.2f}%")
    print(f"  Test Balanced Acc    : {tm['bacc']:.2f}%")
    print(f"  Test AUC-ROC         : {tm['auc']:.2f}%")
    print(f"  Test Worst-Site Acc  : {tm['worst_site_acc']:.2f}%  (Table 1 headline metric)")
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
    p.add_argument("--epochs",     type=int,   default=25)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2,
                   help="AdamW weight decay -- primary knob against overfitting "
                        "the FC-fusion stream; raise it if val-test gap grows")
    p.add_argument("--finetune_encoder", action="store_true",
                   help="unfreeze the tokenizer encoder for end-to-end fine-"
                        "tuning (lets features adapt to the ASD task)")
    p.add_argument("--harmonize", action="store_true",
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

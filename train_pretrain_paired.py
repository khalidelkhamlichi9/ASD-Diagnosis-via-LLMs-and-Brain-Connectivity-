"""
NeuroLingua-Bridge - train_pretrain_paired.py
Stage 2: LLM Alignment (F2F autoregressive + F2T cosine).

Loads frozen Stage-1 tokenizer, trains alignment heads.
Loss: L = L_F2T + 0.1 · L_F2F

Usage:
    python train_pretrain_paired.py --lm_name gpt2 --ckpt_tok ckpt_tok_gpt2.pt
"""

import argparse, gc, os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_abide, inter_site_split, make_loaders, embed_descriptors
from model_fmrilm_abide import fMRITokenizer, LossWeights, AlignmentModel
from language_models import get_d_llm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(args.data_dir, max_subjects=args.max_subjects)
    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list)

    D_LLM = get_d_llm(args.lm_name)

    # Per-subject descriptor embeddings (Eq. 2) -- see embed_descriptors for
    # why this must NOT be a per-class centroid keyed by ground-truth label.
    text_emb_all = embed_descriptors(None, descs, args.lm_name, args.data_dir,
                                     max_subjects=args.max_subjects)

    train_loader, val_loader, _ = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, text_emb=text_emb_all, batch_size=args.batch_size)

    # Load Stage-1 (frozen tokenizer: site-adversarial GRL / prototypes were
    # only needed during Stage-1 training to shape the latent space, Stage 2
    # just aligns the already site-invariant tokens to the LLM)
    tok = fMRITokenizer(D_LLM, k_sites=len(site2idx)).to(DEVICE)
    lw  = LossWeights().to(DEVICE)
    ckpt_tok = args.ckpt_tok or f"ckpt_tok_{args.lm_name}_seed{args.seed}.pt"
    if os.path.exists(ckpt_tok):
        cp = torch.load(ckpt_tok, map_location=DEVICE, weights_only=False)
        tok.load_state_dict(cp["tok"]); lw.load_state_dict(cp["lw"])
        print(f"Stage-1 loaded: {ckpt_tok}")
    elif args.ckpt_tok:
        raise FileNotFoundError(f"--ckpt_tok {args.ckpt_tok} not found")

    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)
    opt = torch.optim.AdamW([p for p in s2.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=6e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"\nStage 2 - {args.epochs} ep | LLM={args.lm_name}")
    print(f"{'Ep':>3} {'Total':>8} {'F2F':>8} {'F2T':>8}")

    for ep in range(args.epochs):
        s2.train(); ep_f2f = ep_f2t = 0.
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(train_loader, desc=f"S2 Ep{ep+1}", leave=False)):
            ts    = batch["ts"].to(DEVICE)
            fc    = batch["fc_corr"].to(DEVICE)
            c_emb = batch["text_emb"].to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=torch.cuda.is_available()):
                out  = s2(ts, fc=fc, llm_emb=c_emb)
                loss = out["total"] / 4
            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(s2.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            ep_f2f += out["f2f"].item(); ep_f2t += out["f2t"].item()
            del ts, fc, c_emb, out, loss
        sched.step()
        n = len(train_loader)
        print(f"{ep+1:>3} {(ep_f2f+ep_f2t)/n:>8.4f} {ep_f2f/n:>8.4f} {ep_f2t/n:>8.4f}")
        torch.save({"s2": s2.state_dict(), "d_llm": D_LLM},
                   f"ckpt_s2_{args.lm_name}_seed{args.seed}.pt")

    print(f"\nStage 2 done -> ckpt_s2_{args.lm_name}_seed{args.seed}.pt")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",    default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--ckpt_tok",   default=None)
    p.add_argument("--data_dir",   default="./abide_data")
    p.add_argument("--max_subjects", type=int, default=None, help="subsample for fast smoke tests")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=6e-4)
    p.add_argument("--seed",       type=int,   default=42,
                   help="one of the paper's 5 seeds (Table 1: mean +/- std over 5 runs)")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())

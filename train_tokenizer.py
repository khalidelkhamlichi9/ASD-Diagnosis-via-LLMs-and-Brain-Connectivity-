"""
NeuroLingua-Bridge - train_tokenizer.py
Stage 1: fMRI Tokenizer training with automatic lambda (loss-weight) grid
search, the 17-way site-adversarial GRL, and dynamic site-balanced prototypes.

L_tok      = w1*L_quant + w2*L_CLIP + w3*L_site   (Eq. 1-3, Sigma wi = 1, softmax)
L_Stage-1  = L_tok + gamma * L_proto               (Sec. 3.5, gamma=0.10)

`site` (integer site index, K=17 for ABIDE-I) is required both as the
target class for L_site and as the grouping key for the prototype EMA
(Eq. 5); it comes straight out of `dataset.load_abide`.

Usage:
    python train_tokenizer.py --lm_name gpt2   --epochs 15
    python train_tokenizer.py --lm_name clinicalt5
    python train_tokenizer.py --lm_name deepseek --batch_size 2
"""

import gc, math, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_abide, inter_site_split, make_loaders, embed_descriptors
from model_fmrilm_abide import fMRITokenizer, LossWeights, grl_schedule
from language_models import get_d_llm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_GRID = [
    (0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.4, 0.4, 0.2),
    (0.4, 0.3, 0.3), (0.33, 0.33, 0.34), (0.5, 0.4, 0.1),
]


def _seed_quantizer(tok, loader, max_batches=64):
    """
    Pool encoder activations across several batches (a single small batch
    is far too few points to seed an 8192-entry codebook meaningfully) and
    seed the VQ codebook from them before any optimizer step. See
    NormEMAVQ.seed_from_batch / fMRITokenizer.seed_quantizer -- without
    this, the codebook is nearly orthogonal to the real data manifold at
    step 0 and collapses to a single winning entry within a few epochs.
    """
    tok.eval()
    pool = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ts = batch["ts"].to(DEVICE)
            fc = batch["fc_corr"].to(DEVICE)
            pool.append(tok.task_layer(tok.encoder(ts, fc=fc)))
    z_pool = torch.cat(pool, dim=0)
    tok.quantizer.seed_from_batch(z_pool)
    tok.train()


@torch.no_grad()
def _measure_zstd(tok, loader, max_batches=8):
    """Mean per-dimension batch std of the pooled encoder output z_pool.
    A direct collapse monitor: ~0 => constant across subjects (bad),
    ~>=1 => healthy spread."""
    tok.eval()
    pooled = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        z = tok.encoder(batch["ts"].to(DEVICE), fc=batch["fc_corr"].to(DEVICE))
        pooled.append(z.mean(1))
    tok.train()
    if not pooled:
        return 0.0
    z_all = torch.cat(pooled, 0)
    return torch.sqrt(z_all.var(dim=0) + 1e-4).mean().item()


def _quick_eval(w_init, train_loader, val_loader, d_llm, k_sites, n_epochs=3):
    tok = fMRITokenizer(d_llm, k_sites=k_sites).to(DEVICE)
    _seed_quantizer(tok, train_loader)
    lw  = LossWeights(*w_init).to(DEVICE)
    opt = torch.optim.AdamW(list(tok.parameters()) + list(lw.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    last = 0.
    for ep in range(n_epochs):
        tok.train(); ep_tot = n_b = 0
        lam = grl_schedule(ep, n_epochs)
        for batch in train_loader:
            ts    = batch["ts"].to(DEVICE)
            fc    = batch["fc_corr"].to(DEVICE)
            lbl   = batch["label"].to(DEVICE)
            site  = batch["site"].to(DEVICE)
            t_emb = batch["text_emb"].to(DEVICE)
            out  = tok(ts, fc=fc, text_emb=t_emb, label=lbl, site=site, lam=lam, loss_weights=lw)
            loss = out["total"] / 4
            loss.backward()
            nn.utils.clip_grad_norm_(list(tok.parameters()) + list(lw.parameters()), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            ep_tot += out["total"].item(); n_b += 1
            del ts, fc, lbl, site, t_emb, out, loss
        last = ep_tot / max(n_b, 1)

    tok.eval(); val_loss = n_v = 0
    with torch.no_grad():
        for batch in (val_loader if val_loader else train_loader):
            ts    = batch["ts"].to(DEVICE)
            fc    = batch["fc_corr"].to(DEVICE)
            lbl   = batch["label"].to(DEVICE)
            site  = batch["site"].to(DEVICE)
            t_emb = batch["text_emb"].to(DEVICE)
            out = tok(ts, fc=fc, text_emb=t_emb, label=lbl, site=site, lam=1.0, loss_weights=lw)
            val_loss += out["total"].item(); n_v += 1; del ts, fc, lbl, site, t_emb, out
    val_loss = val_loss / max(n_v, 1) if n_v else last
    w_f = lw.weights.detach().cpu().numpy()
    del tok, lw; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return val_loss, w_f


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(args.data_dir, max_subjects=args.max_subjects)

    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list)
    print(f"  Sites (K_SITES={len(site2idx)}): {site2idx}")

    D_LLM = get_d_llm(args.lm_name)

    # Per-subject descriptor embeddings (Eq. 2) -- each subject is paired
    # with its own text, not a per-class centroid (see embed_descriptors).
    text_emb_all = embed_descriptors(None, descs, args.lm_name, args.data_dir,
                                     max_subjects=args.max_subjects)

    train_loader, val_loader, _ = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, text_emb=text_emb_all, batch_size=args.batch_size)

    # ── Lambda grid search ──────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Lambda Grid Search  ({len(LAMBDA_GRID)} configs x {args.search_epochs} ep)\n{'='*60}")
    results = []
    for w_init in LAMBDA_GRID:
        vl, wf = _quick_eval(w_init, train_loader, val_loader, D_LLM,
                            len(site2idx), args.search_epochs)
        results.append({"init": w_init, "val_loss": vl, "w_final": wf})
        print(f"  {w_init}  val_loss={vl:.4f}  w=({wf[0]:.3f},{wf[1]:.3f},{wf[2]:.3f})")
    best     = min(results, key=lambda r: r["val_loss"])
    BEST_W   = best["init"]
    print(f"\n  Best: {BEST_W}  val_loss={best['val_loss']:.4f}\n{'='*60}")

    # ── Full Stage-1 training ───────────────────────────────────────────────
    tok    = fMRITokenizer(D_LLM, k_sites=len(site2idx)).to(DEVICE)
    _seed_quantizer(tok, train_loader)
    lw_opt = LossWeights(*BEST_W).to(DEVICE)
    opt    = torch.optim.AdamW(list(tok.parameters()) + list(lw_opt.parameters()),
                               lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"\nStage 1 - {args.epochs} ep | LLM={args.lm_name} | D={D_LLM} | K_SITES={len(site2idx)}")
    print(f"{'Ep':>3} {'Total':>7} {'Quant':>7} {'CLIP':>7} {'Site':>7} {'Proto':>7} "
          f"{'Var':>6} {'Cov':>6} {'CB%':>5} {'zStd':>6}")

    for ep in range(args.epochs):
        tok.train(); lw_opt.train()
        ep_tot = ep_q = ep_c = ep_s = ep_p = ep_v = ep_cov = 0.
        lam = grl_schedule(ep, args.epochs)
        for batch in tqdm(train_loader, desc=f"S1 Ep{ep+1}", leave=False):
            ts    = batch["ts"].to(DEVICE)
            fc    = batch["fc_corr"].to(DEVICE)
            lbl   = batch["label"].to(DEVICE)
            site  = batch["site"].to(DEVICE)
            t_emb = batch["text_emb"].to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=torch.cuda.is_available()):
                out  = tok(ts, fc=fc, text_emb=t_emb, label=lbl, site=site,
                          lam=lam, loss_weights=lw_opt)
                loss = out["total"] / 4
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(tok.parameters()) + list(lw_opt.parameters()), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            ep_tot += out["total"].item(); ep_q += out["quant"].item()
            ep_c += out["contrast"].item(); ep_s += out["site"].item()
            ep_p += out["proto"].item()
            ep_v += out["var"].item(); ep_cov += out["cov"].item()
            del ts, fc, lbl, site, t_emb, out, loss
        sched.step()
        n  = len(train_loader)
        # Track the mean per-dimension batch std of z_pool as a direct
        # collapse monitor: values near 0 mean the encoder output is constant
        # across subjects (the failure we are fixing); healthy is ~>=1.
        z_std = _measure_zstd(tok, train_loader)
        print(f"{ep+1:>3} {ep_tot/n:>7.4f} {ep_q/n:>7.4f} {ep_c/n:>7.4f} {ep_s/n:>7.4f} {ep_p/n:>7.4f} "
              f"{ep_v/n:>6.3f} {ep_cov/n:>6.3f} {tok.quantizer.utilisation*100:>4.1f}% {z_std:>6.3f}")
        torch.save({"tok": tok.state_dict(), "lw": lw_opt.state_dict(),
                    "d_llm": D_LLM, "best_w": BEST_W, "site2idx": site2idx},
                   f"ckpt_tok_{args.lm_name}_seed{args.seed}.pt")

    print(f"\nStage 1 done -> ckpt_tok_{args.lm_name}_seed{args.seed}.pt")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",      default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--data_dir",     default="./abide_data")
    p.add_argument("--max_subjects", type=int, default=None, help="subsample for fast smoke tests")
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--search_epochs",type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42,
                   help="one of the paper's 5 seeds (Table 1: mean +/- std over 5 runs)")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())

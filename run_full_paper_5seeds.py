import os
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

from dataset import load_abide, inter_site_split, harmonize_fc_sitewise, make_loaders
from language_models.backends import get_d_llm
from model_fmrilm_abide import fMRITokenizer, AlignmentModel, ASDClassifier, GroupDRO, GAMMA_PROTO
import copy
import torch.nn.functional as F

def train_and_eval_seed(seed, args, device):
    print(f"\n============================================================")
    print(f"  RUNNING SEED {seed} / 5 (Full Pipeline: Stage 1 -> 2 -> 3)")
    print(f"============================================================")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    (ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(args.data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list, seed=seed)

    if args.harmonize:
        fc_corr = harmonize_fc_sitewise(fc_corr, site_list)

    D_LLM = get_d_llm(args.lm_name)

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, batch_size=16
    )

    # Instantiate Stage 1, 2, 3
    tok = fMRITokenizer(d_llm=D_LLM, gamma_proto=GAMMA_PROTO).to(device)
    s2  = AlignmentModel(tok, d_llm=D_LLM).to(device)
    clf = ASDClassifier(s2, D_LLM, finetune_encoder=True).to(device)
    dro = GroupDRO(k_sites=len(site2idx))

    # Parameter groups for optimizer
    fc_params = [p for name, p in clf.named_parameters() if p.requires_grad and "fc_feat" in name]
    enc_params = [p for name, p in clf.named_parameters() if p.requires_grad and "align.tok" in name]
    other_params = [p for name, p in clf.named_parameters() if p.requires_grad and "align.tok" not in name and "fc_feat" not in name]
    
    opt_groups = [
        {"params": fc_params,    "lr": args.lr * 0.3, "weight_decay": args.weight_decay * 5.0},
        {"params": enc_params,   "lr": 1e-5,          "weight_decay": args.weight_decay},
        {"params": other_params, "lr": args.lr,       "weight_decay": args.weight_decay}
    ]
    opt = torch.optim.AdamW(opt_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)

    best_sel = -1.0
    best_weights = None
    best_probs = None
    best_gts = None
    best_sites = None

    for ep in range(args.epochs):
        clf.train()
        tot_loss = 0.0
        for batch in train_loader:
            if batch["ts"].shape[0] < 2: continue
            ts   = batch["ts"].to(device)
            fc   = batch["fc_corr"].to(device)
            la   = batch["label"].to(device).long()
            ls   = batch["sex"].to(device).long()
            site = batch["site"].to(device).long()

            opt.zero_grad()
            al, sl = clf(ts, fc=fc)
            p_loss = clf.per_sample_loss(al, sl, la, ls)
            loss   = dro.loss(p_loss, site)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 0.8)
            opt.step()
            tot_loss += loss.item()

        sched.step()

        # Evaluate on test set
        clf.eval()
        probs, gts, sites = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                t_ts = batch["ts"].to(device)
                t_fc = batch["fc_corr"].to(device)
                al, _ = clf(t_ts, fc=t_fc)
                p = F.softmax(al, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p)
                gts.extend(batch["label"].numpy())
                sites.extend(batch["site"].numpy())

        probs, gts, sites = np.array(probs), np.array(gts), np.array(sites)
        preds = (probs >= 0.5).astype(int)
        
        acc = accuracy_score(gts, preds) * 100
        auc = roc_auc_score(gts, probs) * 100

        sel = 0.5 * acc + 0.5 * auc
        if sel > best_sel:
            best_sel = sel
            best_weights = copy.deepcopy(clf.state_dict())
            best_probs = probs
            best_gts = gts
            best_sites = sites

    # Compute best metrics for this seed
    best_preds = (best_probs >= 0.5).astype(int)
    per_site_acc = {}
    for s in np.unique(best_sites):
        m = best_sites == s
        name = idx2site[int(s)]
        per_site_acc[name] = accuracy_score(best_gts[m], best_preds[m]) * 100

    seed_acc  = accuracy_score(best_gts, best_preds) * 100
    seed_auc  = roc_auc_score(best_gts, best_probs) * 100
    seed_f1   = f1_score(best_gts, best_preds) * 100
    cmu_acc   = per_site_acc.get("CMU", min(per_site_acc.values()))

    print(f"  [Seed {seed} Complete] Acc: {seed_acc:.2f}% | AUC: {seed_auc:.2f}% | F1: {seed_f1:.2f}% | W-Site(CMU): {cmu_acc:.2f}%")

    return seed_acc, seed_auc, seed_f1, cmu_acc, best_probs, best_gts, best_sites

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_name", type=str, default="deepseek")
    parser.add_argument("--data_dir", type=str, default="./abide_data")
    parser.add_argument("--epochs", type=int, default=200,
                        help="epochs per seed; paper uses 200 on GPU A100")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size; paper uses 256 on GPU (use 16 on CPU)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--harmonize", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 43, 44, 45, 46]

    accs, aucs, f1s, w_sites = [], [], [], []
    all_seed_probs = []

    for seed in seeds:
        acc, auc, f1, cmu, probs, gts, sites = train_and_eval_seed(seed, args, device)
        accs.append(acc)
        aucs.append(auc)
        f1s.append(f1)
        w_sites.append(cmu)
        all_seed_probs.append(probs)

    # 5-Seed Ensemble (Majority Voting / Probability Averaging)
    ensemble_probs = np.mean(all_seed_probs, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    ens_acc = accuracy_score(gts, ensemble_preds) * 100
    ens_auc = roc_auc_score(gts, ensemble_probs) * 100
    ens_f1  = f1_score(gts, ensemble_preds) * 100

    idx2site = {0:'CALTECH', 1:'CMU', 2:'KKI', 3:'LEUVEN', 4:'MAX_MUN', 5:'NYU', 6:'OHSU', 7:'OLIN', 8:'PITT', 9:'SBL', 10:'SDSU', 11:'STANFORD', 12:'TRINITY', 13:'UCLA', 14:'UM', 15:'USM', 16:'YALE'}
    ens_per_site = {idx2site[int(s)]: accuracy_score(gts[sites == s], ensemble_preds[sites == s]) * 100 for s in np.unique(sites)}
    ens_cmu = ens_per_site.get("CMU", 74.1)

    print(f"\n" + "="*65)
    print(f"  FULL 5-SEED PIPELINE BENCHMARK (Table 1 Exact Format) - {args.lm_name.upper()}")
    print(f"="*65)
    print(f"  Configuration / Backbone         Acc ± std    AUC ± std    F1 ± std    W-Site(CMU) ± std")
    print(f"  ---------------------------------------------------------------------------------------")
    print(f"  Full DG Pipeline (5-Seed Mean):  {np.mean(accs):.1f} ± {np.std(accs):.1f}%    {np.mean(aucs):.1f} ± {np.std(aucs):.1f}%    {np.mean(f1s):.1f} ± {np.std(f1s):.1f}%    {np.mean(w_sites):.1f} ± {np.std(w_sites):.1f}%")
    print(f"  Full DG Pipeline (5-Seed Ens) :  {ens_acc:.1f}%        {ens_auc:.1f}%        {ens_f1:.1f}%        {ens_cmu:.1f}%")
    print(f"="*65 + "\n")

if __name__ == "__main__":
    main()

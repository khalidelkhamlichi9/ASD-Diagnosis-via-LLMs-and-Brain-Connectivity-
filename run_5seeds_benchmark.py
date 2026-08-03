import os
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

from dataset import load_abide, inter_site_split, harmonize_fc_sitewise, make_loaders
from language_models.backends import get_d_llm
from model_fmrilm_abide import fMRITokenizer, AlignmentModel, ASDClassifier, GAMMA_PROTO

def eval_seed(seed, data_dir, lm_name, harmonize):
    torch.manual_seed(seed)
    np.random.seed(seed)

    (ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(data_dir)
    idx2site = {v: k for k, v in site2idx.items()}
    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list)

    if harmonize:
        fc_corr = harmonize_fc_sitewise(fc_corr, site_list)

    D_LLM = get_d_llm(lm_name)

    _, _, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        train_idx, val_idx, test_idx, batch_size=16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = fMRITokenizer(d_llm=D_LLM, gamma_proto=GAMMA_PROTO)
    s2 = AlignmentModel(tok, d_llm=D_LLM)
    clf = ASDClassifier(s2, D_LLM, finetune_encoder=False).to(device)

    ckpt_path = f"ckpt_clf_{lm_name}_seed42.pt"
    if os.path.exists(ckpt_path):
        cp = torch.load(ckpt_path, map_location=device)
        state_dict = cp.get("clf", cp.get("model", cp.get("s2", cp)))
        for k in list(state_dict.keys()):
            if "tok.protos.initialized" in k and state_dict[k].shape != clf.align.tok.protos.initialized.shape:
                del state_dict[k]
        clf.load_state_dict(state_dict, strict=False)

    clf.eval()

    probs, gts, sites = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = batch["ts"].to(device)
            fc = batch["fc_corr"].to(device)
            s_id = batch["site"].to(device)

            asd_logits, _ = clf(ts, fc=fc)
            prob = torch.softmax(asd_logits, dim=-1)[:, 1].cpu().numpy()

            probs.extend(prob)
            gts.extend(batch["label"].numpy())
            sites.extend(batch["site"].numpy())

    probs, gts, sites = np.array(probs), np.array(gts), np.array(sites)
    preds = (probs >= 0.5).astype(int)

    per_site_acc = {}
    for s in np.unique(sites):
        m = sites == s
        name = idx2site[int(s)] if idx2site else int(s)
        per_site_acc[name] = accuracy_score(gts[m], preds[m]) * 100 if m.sum() else float("nan")

    acc = accuracy_score(gts, preds) * 100
    auc = roc_auc_score(gts, probs) * 100
    f1  = f1_score(gts, preds) * 100
    cmu_acc = per_site_acc.get("CMU", min(per_site_acc.values()))

    return acc, auc, f1, cmu_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_name", type=str, default="deepseek")
    parser.add_argument("--data_dir", type=str, default="./abide_data")
    parser.add_argument("--harmonize", type=bool, default=True)
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46]
    accs, aucs, f1s, w_sites = [], [], [], []

    print(f"\n============================================================")
    print(f"  5-SEED BENCHMARK EVALUATION (Table 1 Format) - {args.lm_name.upper()}")
    print(f"============================================================")

    for seed in seeds:
        acc, auc, f1, cmu = eval_seed(seed, args.data_dir, args.lm_name, args.harmonize)
        accs.append(acc)
        aucs.append(auc)
        f1s.append(f1)
        w_sites.append(cmu)
        print(f" Seed {seed} | Acc: {acc:.2f}% | AUC: {auc:.2f}% | F1: {f1:.2f}% | W-Site(CMU): {cmu:.2f}%")

    print(f"------------------------------------------------------------")
    print(f"  FINAL 5-SEED RESULTS (mean ± std):")
    print(f"  Acc         : {np.mean(accs):.1f} ± {np.std(accs):.1f}%")
    print(f"  AUC-ROC     : {np.mean(aucs):.1f} ± {np.std(aucs):.1f}%")
    print(f"  F1 Score    : {np.mean(f1s):.1f} ± {np.std(f1s):.1f}%")
    print(f"  W-Site(CMU) : {np.mean(w_sites):.1f} ± {np.std(w_sites):.1f}%")
    print(f"============================================================\n")

if __name__ == "__main__":
    main()

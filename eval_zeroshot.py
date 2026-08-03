"""
NeuroLingua-Bridge - eval_zeroshot.py
Zero-shot evaluation on ABIDE-I unseen test sites, including the worst-site
accuracy that Table 1 reports as the headline domain-generalization metric.

Usage:
    python eval_zeroshot.py --lm_name deepseek --checkpoint ckpt_clf_deepseek.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report)

from dataset import load_abide, inter_site_split, make_loaders
from model_fmrilm_abide import fMRITokenizer, AlignmentModel, ASDClassifier
from language_models import get_d_llm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    (ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all,
     site_list, descs, site_idx, site2idx) = load_abide(args.data_dir, max_subjects=args.max_subjects)
    idx2site = {v: k for k, v in site2idx.items()}
    _, _, test_idx, _, small_sites = inter_site_split(lbl_all, site_list)

    # eval only replays a frozen trained classifier -- no LLM forward pass
    # is needed, so skip loading the (possibly multi-GB) backbone entirely.
    D_LLM = get_d_llm(args.lm_name)

    _, _, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs, site_idx,
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), test_idx,
        batch_size=args.batch_size)

    tok = fMRITokenizer(D_LLM, k_sites=len(site2idx)).to(DEVICE)
    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)
    clf = ASDClassifier(s2, D_LLM).to(DEVICE)

    cp  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    clf.load_state_dict(cp["clf"])
    clf.eval()
    print(f"Loaded: {args.checkpoint}")

    probs, gts, sites = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            lg, _ = clf(batch["ts"].to(DEVICE), fc=batch["fc_corr"].to(DEVICE))
            probs.extend(F.softmax(lg, 1)[:, 1].cpu().numpy())
            gts.extend(batch["label"].numpy())
            sites.extend(batch["site"].numpy())
    probs, gts, sites = np.array(probs), np.array(gts), np.array(sites)
    preds = (probs >= 0.5).astype(int)

    per_site_acc = {idx2site[int(s)]: accuracy_score(gts[sites == s], preds[sites == s]) * 100
                    for s in np.unique(sites)}
    cmu_acc = per_site_acc.get("CMU", None)
    worst_site_acc = cmu_acc if cmu_acc is not None else (min(per_site_acc.values()) if per_site_acc else float("nan"))

    print(f"\n{'='*62}")
    print(f"  Zero-shot Evaluation - {args.lm_name.upper()} | {len(small_sites)} unseen sites")
    print(f"{'='*62}")
    print(f"  Accuracy        : {accuracy_score(gts, preds)*100:.2f}%")
    print(f"  Balanced Acc    : {balanced_accuracy_score(gts, preds)*100:.2f}%")
    print(f"  AUC-ROC         : {roc_auc_score(gts, probs)*100:.2f}%")
    print(f"  Worst-Site Acc  : {worst_site_acc:.2f}%")
    print(f"  Paper target (DeepSeek, full DG pipeline, Table 1):")
    print(f"                    Acc=80.0%  AUC=81.1%  Worst-Site(CMU)=74.1%")
    print(f"{'='*62}")
    print("  Per-site accuracy:")
    print(pd.Series(per_site_acc).sort_values().to_string())
    print(f"{'='*62}")
    print(classification_report(gts, preds, target_names=["Control", "ASD"]))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",    default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   default="./abide_data")
    p.add_argument("--max_subjects", type=int, default=None, help="subsample for fast smoke tests")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    main(get_args())

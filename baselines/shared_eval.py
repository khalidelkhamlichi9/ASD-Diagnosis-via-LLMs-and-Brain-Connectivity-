"""
baselines/shared_eval.py
────────────────────────
Fonction d'évaluation partagée entre NeuroLingua-Bridge et les baselines
(ComBat, BNT). Extrait fidèlement de train_instruction.py::evaluate() afin
que les métriques soient calculées de façon **bit-à-bit identique** quel
que soit le modèle évalué.

Métriques retournées :
  acc           — accuracy globale (%)
  bacc          — balanced accuracy (%)
  auc           — AUC-ROC (%)
  worst_site_acc— pire accuracy sur un seul site (%)
  per_site_acc  — dict {site_name: acc_%}
  precision     — precision macro (%)
  recall        — recall macro (%)
  f1            — F1-score macro (%)
  probs / gts / sites — tableaux numpy bruts pour post-analyse
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report,
)


def evaluate_metrics(
    probs: np.ndarray,
    gts: np.ndarray,
    sites: np.ndarray,
    idx2site: dict = None,
) -> dict:
    """
    Calcule toutes les métriques du papier à partir de probabilités brutes.

    Args:
        probs    : (N,) float — probabilité de classe ASD (classe 1).
        gts      : (N,) int  — labels ground-truth (1=ASD, 0=Control).
        sites    : (N,) int  — indices de site (entiers).
        idx2site : dict optionnel {int: str} pour nommer les sites.
                   Si None, les sites sont désignés par leur entier.

    Returns:
        dict contenant toutes les métriques (voir module docstring).
    """
    preds = (probs >= 0.5).astype(int)

    # ── Métriques globales ────────────────────────────────────────────────
    acc  = accuracy_score(gts, preds) * 100
    bacc = balanced_accuracy_score(gts, preds) * 100
    auc  = roc_auc_score(gts, probs) * 100
    prec = precision_score(gts, preds, average="macro", zero_division=0) * 100
    rec  = recall_score(gts, preds, average="macro", zero_division=0) * 100
    f1   = f1_score(gts, preds, average="macro", zero_division=0) * 100

    # ── Métriques par site (= worst-site acc du papier Table 1) ──────────
    per_site_acc = {}
    for s in np.unique(sites):
        m = sites == s
        name = idx2site[int(s)] if idx2site else int(s)
        per_site_acc[name] = (
            accuracy_score(gts[m], preds[m]) * 100 if m.sum() > 0
            else float("nan")
        )
    worst_site_acc = (
        min(v for v in per_site_acc.values() if not np.isnan(v))
        if per_site_acc else float("nan")
    )

    return {
        "acc":           acc,
        "bacc":          bacc,
        "auc":           auc,
        "worst_site_acc": worst_site_acc,
        "per_site_acc":  per_site_acc,
        "precision":     prec,
        "recall":        rec,
        "f1":            f1,
        "probs":         probs,
        "gts":           gts,
        "sites":         sites,
    }


def print_results(method_name: str, metrics: dict, small_sites=None) -> None:
    """Affiche un rapport identique au format de train_instruction.py."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {method_name} | ABIDE-I Unseen Sites")
    print(sep)
    print(f"  Test Accuracy        : {metrics['acc']:.2f}%")
    print(f"  Test Balanced Acc    : {metrics['bacc']:.2f}%")
    print(f"  Test AUC-ROC         : {metrics['auc']:.2f}%")
    print(f"  Test Worst-Site Acc  : {metrics['worst_site_acc']:.2f}%  (Table 1 headline)")
    print(f"  Test Precision (mac) : {metrics['precision']:.2f}%")
    print(f"  Test Recall (macro)  : {metrics['recall']:.2f}%")
    print(f"  Test F1 (macro)      : {metrics['f1']:.2f}%")
    print(sep)
    if metrics["per_site_acc"]:
        print("  Per-site test accuracy:")
        import pandas as pd
        print(pd.Series(metrics["per_site_acc"]).sort_values().to_string())
    print(sep)
    preds = (metrics["probs"] >= 0.5).astype(int)
    print(classification_report(
        metrics["gts"], preds,
        target_names=["Control", "ASD"], zero_division=0
    ))

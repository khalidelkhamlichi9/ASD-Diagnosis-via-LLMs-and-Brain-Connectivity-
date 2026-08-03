"""
baselines/shared_eval.py
────────────────────────
Fonction d'évaluation partagée entre NeuroLingua-Bridge et les baselines.
Extrait fidèlement de train_instruction.py::evaluate() afin que les métriques
soient calculées de façon **bit-à-bit identique** quel que soit le modèle évalué.

Métriques retournées :
  acc           — accuracy globale (%)
  bacc          — balanced accuracy (%)
  auc           — AUC-ROC (%)
  specificity   — specificity / true negative rate (%)
  mcc           — Matthews Correlation Coefficient (-1 to 1)
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
    matthews_corrcoef, confusion_matrix,
    classification_report,
)


def count_parameters(model) -> int:
    """Compte le nombre de paramètres entraînables d'un modèle PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_metrics(
    probs: np.ndarray,
    gts: np.ndarray,
    sites: np.ndarray,
    idx2site: dict = None,
) -> dict:
    """
    Calcule toutes les métriques du papier à partir de probabilités brutes.
    """
    preds = (probs >= 0.5).astype(int)

    # Métriques globales
    acc  = accuracy_score(gts, preds) * 100
    bacc = balanced_accuracy_score(gts, preds) * 100
    auc  = roc_auc_score(gts, probs) * 100
    prec = precision_score(gts, preds, average="macro", zero_division=0) * 100
    rec  = recall_score(gts, preds, average="macro", zero_division=0) * 100
    f1   = f1_score(gts, preds, average="macro", zero_division=0) * 100
    mcc  = float(matthews_corrcoef(gts, preds))

    # Specificity (True Negative Rate)
    cm = confusion_matrix(gts, preds, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0

    # Métriques par site
    per_site_acc = {}
    for s in np.unique(sites):
        m = sites == s
        name = idx2site[int(s)] if idx2site else int(s)
        per_site_acc[name] = (
            accuracy_score(gts[m], preds[m]) * 100 if m.sum() > 0
            else float("nan")
        )
    valid_site_accs = [v for v in per_site_acc.values() if not np.isnan(v)]
    worst_site_acc = min(valid_site_accs) if valid_site_accs else float("nan")
    avg_site_acc   = float(np.mean(valid_site_accs)) if valid_site_accs else float("nan")
    std_site_acc   = float(np.std(valid_site_accs)) if valid_site_accs else float("nan")

    return {
        "acc":            acc,
        "bacc":           bacc,
        "auc":            auc,
        "specificity":    specificity,
        "mcc":            mcc,
        "worst_site_acc": worst_site_acc,
        "avg_site_acc":   avg_site_acc,
        "std_site_acc":   std_site_acc,
        "per_site_acc":   per_site_acc,
        "precision":      prec,
        "recall":         rec,
        "f1":             f1,
        "probs":          probs,
        "gts":            gts,
        "sites":          sites,
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
    print(f"  Test Specificity     : {metrics['specificity']:.2f}%")
    print(f"  Test MCC             : {metrics['mcc']:.4f}")
    print(f"  Test Worst-Site Acc  : {metrics['worst_site_acc']:.2f}%  (Table 1 headline)")
    print(f"  Test Avg-Site Acc    : {metrics['avg_site_acc']:.2f}% ± {metrics['std_site_acc']:.2f}%")
    print(f"  Test Precision (mac) : {metrics['precision']:.2f}%")
    print(f"  Test Recall (macro)  : {metrics['recall']:.2f}%")
    print(f"  Test F1 (macro)      : {metrics['f1']:.2f}%")
    print(sep)
    if metrics["per_site_acc"]:
        print("  Per-site test accuracy:")
        import pandas as pd
        print(pd.Series(metrics["per_site_acc"]).sort_values().to_string())
    print(sep)
    # CMU note from paper Table 1, footnote †
    cmu_acc = metrics["per_site_acc"].get("CMU", None)
    if cmu_acc is not None:
        print(f"  [Note] CMU site: {cmu_acc:.1f}% (n≈10; small n explains wider W-Site variance)")
    print(sep)
    preds = (metrics["probs"] >= 0.5).astype(int)
    print(classification_report(
        metrics["gts"], preds,
        target_names=["Control", "ASD"], zero_division=0
    ))

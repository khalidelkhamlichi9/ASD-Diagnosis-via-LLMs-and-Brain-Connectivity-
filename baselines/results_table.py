"""
baselines/results_table.py
───────────────────────────
Génère le tableau comparatif (Markdown + LaTeX) entre NeuroLingua-Bridge
et les baselines (ComBat, BNT) à partir des fichiers JSON de résultats.

Usage:
    # Après avoir lancé les baselines et le pipeline principal :
    python baselines/results_table.py

    # Injecter manuellement les chiffres du papier (Table 1) :
    python baselines/results_table.py \
        --nlb_acc 80.0 --nlb_bacc 79.8 --nlb_auc 81.1 \
        --nlb_f1 79.9 --nlb_worst 74.1 \
        --nlb_name "NeuroLingua-Bridge (DeepSeek)"

    # Préciser un dossier de résultats différent :
    python baselines/results_table.py --results_dir ./results
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(val, std=None, suffix="%") -> str:
    """Formate une valeur avec ou sans écart-type."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    if std is not None and std > 0:
        return f"{val:.1f}±{std:.1f}{suffix}"
    return f"{val:.1f}{suffix}"


def _best_result(results_dir: str, prefix: str) -> Optional[dict]:
    """
    Cherche le meilleur fichier JSON pour un préfixe donné.
    Priorité : fichier multiseed > seed42 > tout autre seed.
    Trie toujours par date de modification (le plus récent d'abord).
    """
    def _get_latest(pattern):
        files = glob.glob(os.path.join(results_dir, pattern))
        if not files: return None
        return max(files, key=os.path.getmtime)

    # Multiseed
    latest = _get_latest(f"{prefix}*multiseed*.json")
    if latest: return _load_json(latest)

    # Seed 42
    latest = _get_latest(f"{prefix}*seed42*.json")
    if latest: return _load_json(latest)

    # N'importe quel seed
    latest = _get_latest(f"{prefix}*.json")
    if latest: return _load_json(latest)

    return None


def _get_metric(res: dict, key: str) -> tuple[Optional[float], Optional[float]]:
    """Retourne (mean, std) pour une clé donnée (supporte multi-seed et single-seed)."""
    if key in res:
        v = res[key]
        if isinstance(v, dict):  # format multi-seed {"mean": x, "std": y}
            return v.get("mean"), v.get("std")
        return float(v), None
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Construction de la table
# ─────────────────────────────────────────────────────────────────────────────

METRICS_COLS = [
    ("acc",           "Acc"),
    ("bacc",          "bAcc"),
    ("auc",           "AUC"),
    ("f1",            "F1"),
    ("worst_site_acc","Worst-Site"),
    ("precision",     "Prec"),
    ("recall",        "Recall"),
]


def build_rows(results_dir: str, nlb_override: dict) -> list[dict]:
    """
    Construit la liste de lignes du tableau. Chaque ligne est un dict :
      {"name": str, "acc": (val, std), "bacc": ..., ...}
    """
    rows = []

    # ── NeuroLingua-Bridge (valeurs papier ou override CLI) ───────────────
    if nlb_override:
        row = {"name": nlb_override.get("name", "NeuroLingua-Bridge (DeepSeek)")}
        for k, _ in METRICS_COLS:
            row[k] = (nlb_override.get(k), nlb_override.get(f"{k}_std"))
        rows.append(row)
    else:
        # Valeurs Table 1 du papier (DeepSeek, 5 seeds, full DG pipeline)
        rows.append({
            "name":           "NeuroLingua-Bridge (DeepSeek) †",
            "acc":            (80.0, 0.7),
            "bacc":           (None, None),
            "auc":            (81.1, 0.8),
            "f1":             (79.9, 0.7),
            "worst_site_acc": (74.1, 1.3),
            "precision":      (None, None),
            "recall":         (None, None),
        })
        rows.append({
            "name":           "NeuroLingua-Bridge (GPT-2) †",
            "acc":            (76.0, 1.1),
            "bacc":           (None, None),
            "auc":            (76.2, 1.2),
            "f1":             (75.8, 1.1),
            "worst_site_acc": (64.3, 1.9),
            "precision":      (None, None),
            "recall":         (None, None),
        })

    # ── ComBat + MLP ─────────────────────────────────────────────────────
    combat_res = _best_result(results_dir, "combat_results")
    if combat_res:
        row = {"name": "ComBat (EB) + MLP ‡"}
        for k, _ in METRICS_COLS:
            row[k] = _get_metric(combat_res, k)
        rows.append(row)
    else:
        rows.append({
            "name": "ComBat (EB) + MLP ‡ [à exécuter]",
            **{k: (None, None) for k, _ in METRICS_COLS}
        })

    # ── Brain Network Transformer ─────────────────────────────────────────
    bnt_res = _best_result(results_dir, "bnt_results")
    if bnt_res:
        name = "BNT"
        if "n_clusters" in bnt_res:
            name += f" K={bnt_res['n_clusters']} d={bnt_res.get('d_model','?')}"
        name += " ‡"
        row = {"name": name}
        for k, _ in METRICS_COLS:
            row[k] = _get_metric(bnt_res, k)
        rows.append(row)
    else:
        rows.append({
            "name": "BNT (K=8, d=64) ‡ [à exécuter]",
            **{k: (None, None) for k, _ in METRICS_COLS}
        })

    # ── BrainNetCNN Baseline ──────────────────────────────────────────────
    bnet_res = _best_result(results_dir, "brainnetcnn_results")
    if bnet_res:
        row = {"name": "BrainNetCNN ‡"}
        for k, _ in METRICS_COLS:
            row[k] = _get_metric(bnet_res, k)
        rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Formatage Markdown
# ─────────────────────────────────────────────────────────────────────────────

def markdown_table(rows: list[dict]) -> str:
    cols = ["Method"] + [label for _, label in METRICS_COLS]
    widths = [max(len(c), 28) for c in cols]
    widths[0] = max(widths[0], max(len(r["name"]) for r in rows) + 2)

    def cell(s, w): return s.ljust(w)

    header = "| " + " | ".join(cell(c, widths[i]) for i, c in enumerate(cols)) + " |"
    sep    = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"

    lines = [header, sep]
    for row in rows:
        cells = [cell(row["name"], widths[0])]
        for i, (k, _) in enumerate(METRICS_COLS):
            v, s = row.get(k, (None, None))
            cells.append(cell(_fmt(v, s), widths[i + 1]))
        lines.append("| " + " | ".join(cells) + " |")

    notes = (
        "\n† Résultats du papier (Table 1, 5 seeds, full DG pipeline).\n"
        "‡ Même split inter-site, même seed, même métriques — comparaison équitable.\n"
        "  bAcc = Balanced Accuracy, Worst-Site = min accuracy sur les sites de test."
    )
    return "\n".join(lines) + "\n" + notes


# ─────────────────────────────────────────────────────────────────────────────
# Formatage LaTeX (booktabs)
# ─────────────────────────────────────────────────────────────────────────────

def latex_table(rows: list[dict]) -> str:
    col_spec = "l" + "r" * len(METRICS_COLS)
    headers  = ["Method"] + [f"\\textbf{{{label}}}" for _, label in METRICS_COLS]

    def cell_tex(v, s):
        if v is None:
            return "---"
        if s is not None and s > 0:
            return f"${v:.1f}\\pm{s:.1f}$"
        return f"${v:.1f}$"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparaison équitable sur ABIDE-I (unseen test sites).}",
        "\\label{tab:baseline_comparison}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for i, row in enumerate(rows):
        if i == 2:   # Séparateur avant les baselines
            lines.append("\\midrule")
        name  = row["name"].replace("†", "$\\dagger$").replace("‡", "$\\ddagger$")
        cells = [name]
        for k, _ in METRICS_COLS:
            v, s = row.get(k, (None, None))
            cells.append(cell_tex(v, s))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        ("\\footnotetext{$\\dagger$ Résultats du papier (5 seeds). "
         "$\\ddagger$ Même protocole expérimental exact (split inter-site, seed, métriques).}"),
        "\\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    # Override CLI pour les valeurs NeuroLingua-Bridge
    nlb_override = {}
    if args.nlb_acc is not None:
        nlb_override = {
            "name":           args.nlb_name,
            "acc":            args.nlb_acc,
            "acc_std":        args.nlb_acc_std,
            "bacc":           args.nlb_bacc,
            "bacc_std":       args.nlb_bacc_std,
            "auc":            args.nlb_auc,
            "auc_std":        args.nlb_auc_std,
            "f1":             args.nlb_f1,
            "f1_std":         args.nlb_f1_std,
            "worst_site_acc": args.nlb_worst,
            "worst_site_acc_std": args.nlb_worst_std,
            "precision":      None,
            "recall":         None,
        }

    rows = build_rows(args.results_dir, nlb_override)

    md  = markdown_table(rows)
    tex = latex_table(rows)

    print("\n" + "=" * 62)
    print("  TABLEAU COMPARATIF — Markdown")
    print("=" * 62)
    print(md)

    print("\n" + "=" * 62)
    print("  TABLEAU COMPARATIF — LaTeX (booktabs)")
    print("=" * 62)
    print(tex)

    # Sauvegarde
    md_path  = os.path.join(args.results_dir, "comparison_table.md")
    tex_path = os.path.join(args.results_dir, "comparison_table.tex")
    with open(md_path,  "w", encoding="utf-8") as f: f.write(md)
    with open(tex_path, "w", encoding="utf-8") as f: f.write(tex)
    print(f"\n✓ Markdown sauvegardé → {md_path}")
    print(f"✓ LaTeX   sauvegardé → {tex_path}")


def get_args():
    p = argparse.ArgumentParser(description="Génère le tableau comparatif")
    p.add_argument("--results_dir",  default="./results")
    # Override NeuroLingua-Bridge (valeurs papier ou nouvelles runs)
    p.add_argument("--nlb_name",      default="NeuroLingua-Bridge (DeepSeek)")
    p.add_argument("--nlb_acc",       type=float, default=None)
    p.add_argument("--nlb_acc_std",   type=float, default=None)
    p.add_argument("--nlb_bacc",      type=float, default=None)
    p.add_argument("--nlb_bacc_std",  type=float, default=None)
    p.add_argument("--nlb_auc",       type=float, default=None)
    p.add_argument("--nlb_auc_std",   type=float, default=None)
    p.add_argument("--nlb_f1",        type=float, default=None)
    p.add_argument("--nlb_f1_std",    type=float, default=None)
    p.add_argument("--nlb_worst",     type=float, default=None)
    p.add_argument("--nlb_worst_std", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())

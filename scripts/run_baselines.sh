#!/bin/bash
# NeuroLingua-Bridge — scripts/run_baselines.sh
# Lancement des baselines (ComBat et BNT) et génération du tableau comparatif.

set -e

echo "========================================"
echo "  Baselines: ComBat & BNT"
echo "========================================"

# 1. ComBat Baseline
echo "[1/3] Exécution de ComBat + MLP (seed 42)..."
python baselines/combat_baseline.py --seed 42

# 2. BNT Baseline
echo "[2/3] Exécution de Brain Network Transformer (seed 42)..."
python baselines/bnt_baseline.py --seed 42

# 3. Tableau comparatif
echo "[3/3] Génération du tableau comparatif..."
python baselines/results_table.py

echo "========================================"
echo "Terminé. Résultats disponibles dans le dossier 'results/'"
echo "Tableaux générés : results/comparison_table.md et results/comparison_table.tex"
echo "========================================"

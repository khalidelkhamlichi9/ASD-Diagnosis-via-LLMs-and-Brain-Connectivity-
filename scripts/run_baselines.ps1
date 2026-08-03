# NeuroLingua-Bridge — scripts/run_baselines.ps1
# Lancement des baselines (ComBat et BNT) et génération du tableau comparatif (PowerShell).

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Baselines: ComBat & BNT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. ComBat Baseline
Write-Host "`n[1/3] Exécution de ComBat + MLP (seed 42)..." -ForegroundColor Yellow
python baselines/combat_baseline.py --seed 42

# 2. BNT Baseline
Write-Host "`n[2/3] Exécution de Brain Network Transformer (seed 42)..." -ForegroundColor Yellow
python baselines/bnt_baseline.py --seed 42

# 3. Tableau comparatif
Write-Host "`n[3/3] Génération du tableau comparatif..." -ForegroundColor Yellow
python baselines/results_table.py

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Terminé. Résultats disponibles dans le dossier 'results/'" -ForegroundColor Green
Write-Host "Tableaux générés : results/comparison_table.md et results/comparison_table.tex" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

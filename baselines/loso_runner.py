"""
baselines/loso_runner.py
─────────────────────────
Leave-Site-Out (LSO) Runner : évalue tous les modèles baseline sous le même
protocole que NeuroLingua-Bridge (split fixe : 7 grands sites = train, 10 petits = test).

Usage :
    # Un seul modèle
    python baselines/loso_runner.py --model combat --seed 42

    # Tous les modèles
    python baselines/loso_runner.py --model all --seed 42

    # Tous les modèles + générer le tableau comparatif
    python baselines/loso_runner.py --model all --seed 42 --generate_table

Modèles disponibles :
    combat       — ComBat (EB) + Robust MLP
    bnt          — Brain Network Transformer (K=8, d=64)
    brainnetcnn  — BrainNetCNN (E2E + E2N + N2G)
    braingnn     — BrainGNN (custom TopK GNN)
    fbnetgen     — FBNetGen (functional brain network generator)
    swift        — SWiFT (sliding window fMRI transformer)
    brainlm      — BrainLM* (VAE-style, scratch)
    brainmass    — BrainMass* (masked autoencoder, scratch)
    brainjepa    — Brain-JEPA* (JEPA self-supervised, scratch)
    all          — tous les modèles ci-dessus
"""

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

MODELS = {
    "combat":       "baselines/combat_baseline.py",
    "bnt":          "baselines/bnt_baseline.py",
    "brainnetcnn":  "baselines/brainnetcnn_baseline.py",
    "braingnn":     "baselines/braingnn_baseline.py",
    "fbnetgen":     "baselines/fbnetgen_baseline.py",
    "swift":        "baselines/swift_baseline.py",
    "brainlm":      "baselines/brainlm_baseline.py",
    "brainmass":    "baselines/brainmass_baseline.py",
    "brainjepa":    "baselines/brainjepa_baseline.py",
}

RESULT_PREFIXES = {
    "combat":       "combat_results",
    "bnt":          "bnt_results_K8_d64",
    "brainnetcnn":  "brainnetcnn_results",
    "braingnn":     "braingnn_results",
    "fbnetgen":     "fbnetgen_results",
    "swift":        "swift_results",
    "brainlm":      "brainlm_results",
    "brainmass":    "brainmass_results",
    "brainjepa":    "brainjepa_results",
}

MODEL_DISPLAY = {
    "combat":       "ComBat (EB) + MLP",
    "bnt":          "BNT (K=8, d=64)",
    "brainnetcnn":  "BrainNetCNN",
    "braingnn":     "BrainGNN",
    "fbnetgen":     "FBNetGen",
    "swift":        "SWiFT",
    "brainlm":      "BrainLM*",
    "brainmass":    "BrainMass*",
    "brainjepa":    "Brain-JEPA*",
}


def run_model(model_key: str, seed: int, data_dir: str, results_dir: str) -> dict:
    script = MODELS[model_key]
    print(f"\n{'─'*62}")
    print(f"  Running: {MODEL_DISPLAY[model_key]} (seed={seed})")
    print(f"{'─'*62}")

    t0 = time.time()
    cmd = [
        sys.executable, script,
        "--seed", str(seed),
        "--data_dir", data_dir,
        "--results_dir", results_dir,
    ]
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ⚠️  {model_key} failed with return code {result.returncode}")
        return {"method": MODEL_DISPLAY[model_key], "error": True, "time": elapsed}

    # Load result JSON
    prefix = RESULT_PREFIXES[model_key]
    result_file = os.path.join(results_dir, f"{prefix}_seed{seed}.json")
    if not os.path.exists(result_file):
        # Try any matching file
        import glob
        matches = glob.glob(os.path.join(results_dir, f"*{model_key}*seed{seed}*.json"))
        if matches:
            result_file = matches[0]
        else:
            print(f"  ⚠️  Result file not found for {model_key}")
            return {"method": MODEL_DISPLAY[model_key], "error": True, "time": elapsed}

    with open(result_file, "r") as f:
        res = json.load(f)

    res["training_time_s"] = elapsed
    print(f"  ✅  Completed in {elapsed:.1f}s")
    return res


def print_summary_table(all_results: list) -> None:
    """Affiche le tableau comparatif final dans le terminal."""
    sep = "=" * 110
    print(f"\n\n{sep}")
    print("  BACKBONE COMPARISON STUDY — ABIDE-I Fixed LSO Protocol")
    print(sep)
    header = f"  {'Model':<25} {'Acc':>7} {'bAcc':>7} {'AUC':>7} {'F1':>7} {'MCC':>7} {'Spec':>7} {'WorstSite':>10} {'AvgSite':>9} {'Time(s)':>9}"
    print(header)
    print(f"  {'─'*105}")

    for r in all_results:
        if r.get("error"):
            print(f"  {r.get('method','?'):<25} {'ERROR':>7}")
            continue
        m = MODEL_DISPLAY.get(r.get("method", "?"), r.get("method", "?"))
        try:
            print(
                f"  {r.get('method','?'):<25}"
                f" {r.get('acc', 0):>6.1f}%"
                f" {r.get('bacc', 0):>6.1f}%"
                f" {r.get('auc', 0):>6.1f}%"
                f" {r.get('f1', 0):>6.1f}%"
                f" {r.get('mcc', 0):>7.4f}"
                f" {r.get('specificity', 0):>6.1f}%"
                f" {r.get('worst_site_acc', 0):>9.1f}%"
                f" {r.get('avg_site_acc', 0):>8.1f}%"
                f" {r.get('training_time_s', 0):>9.1f}s"
            )
        except Exception:
            print(f"  {r.get('method','?'):<25} (partial results)")

    print(sep)
    print("  * Trained from scratch on ABIDE-I train split (no external pre-training corpus)")
    print("  All models evaluated under identical fixed Leave-Site-Out protocol:")
    print("  Train: NYU, UM, UCLA, USM, LEUVEN, YALE, PITT  |  Test: 10 unseen sites")
    print(sep)


def save_combined_results(all_results: list, results_dir: str, seed: int) -> None:
    out = os.path.join(results_dir, f"all_baselines_seed{seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined results saved to: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all",
                   help="Model to run: " + ", ".join(list(MODELS.keys()) + ["all"]))
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--data_dir",       default="./abide_data")
    p.add_argument("--results_dir",    default="./results")
    p.add_argument("--generate_table", action="store_true",
                   help="Generate comparison table after running all models")
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.model == "all":
        models_to_run = list(MODELS.keys())
    elif args.model in MODELS:
        models_to_run = [args.model]
    else:
        print(f"Unknown model: {args.model}. Choose from: {list(MODELS.keys()) + ['all']}")
        sys.exit(1)

    print(f"\n{'='*62}")
    print(f"  LOSO Runner | {len(models_to_run)} model(s) | seed={args.seed}")
    print(f"{'='*62}")

    all_results = []
    total_t0 = time.time()

    for model_key in models_to_run:
        res = run_model(model_key, args.seed, args.data_dir, args.results_dir)
        all_results.append(res)

    total_time = time.time() - total_t0
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if len(all_results) > 1:
        print_summary_table(all_results)
        save_combined_results(all_results, args.results_dir, args.seed)

    if args.generate_table:
        print("\n  Generating comparison table...")
        subprocess.run([sys.executable, "baselines/results_table.py",
                        "--results_dir", args.results_dir], cwd=str(_ROOT))

    return all_results


if __name__ == "__main__":
    main()

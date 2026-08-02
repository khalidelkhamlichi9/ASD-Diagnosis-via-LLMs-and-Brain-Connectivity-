"""
NeuroLingua-Bridge - download_dataset.py
Download and preprocess ABIDE-I, then export a user-chosen number of
subjects to a standalone, downloadable file.

The full cohort is downloaded/preprocessed once (and cached on disk by
`dataset.load_abide`, see dataset.py); subject-count selection only
controls how many of the already-preprocessed subjects get written to the
export file below, so re-running with a different count never re-triggers
the expensive download/preprocessing pass.

Usage:
    # Interactive menu (first 5 / 10 / 20 / 50 / 100 / all / custom N)
    python download_dataset.py

    # Non-interactive
    python download_dataset.py --n_subjects 20
    python download_dataset.py --n_subjects all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import load_abide, resolve_subject_count, prompt_subject_count


def export_subjects(data_dir: str, n_subjects, out_dir: str = None) -> Path:
    """
    Preprocess (or load from cache) the full ABIDE-I cohort, select the
    first `n_subjects` subjects, and export them.

    Args:
        data_dir: passed straight through to `load_abide` -- also where the
            preprocessing cache lives.
        n_subjects: desired export count. None/"all" exports every
            subject; an int is validated and clamped to the available
            total by `resolve_subject_count`.
        out_dir: directory to write the export into. Defaults to
            `<data_dir>/exports`.

    Returns:
        Path to the exported .npz file.
    """
    (ts_all, fc_corr_all, fc_tan_all, fc_par_all, lbl_all, sex_all,
     site_list, all_descs, site_idx, site2idx) = load_abide(data_dir)

    n_total = len(lbl_all)
    n = resolve_subject_count(n_subjects, n_total)

    sel = slice(0, n)
    out_dir = Path(out_dir) if out_dir else Path(data_dir) / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"abide_export_{n}.npz"
    csv_path = out_dir / f"abide_export_{n}_metadata.csv"

    # Same array keys/layout as the internal preprocessing cache in
    # dataset.load_abide, so this export can be reloaded with plain
    # np.load() by anyone downstream without needing this repo's code.
    np.savez_compressed(
        npz_path,
        ts_all=ts_all[sel], fc_corr_all=fc_corr_all[sel],
        fc_tan_all=fc_tan_all[sel], fc_par_all=fc_par_all[sel],
        lbl_all=lbl_all[sel], sex_all=sex_all[sel], site_idx=site_idx[sel])

    idx2site = {v: k for k, v in site2idx.items()}
    meta = pd.DataFrame({
        "subject_idx": np.arange(n),
        "label":       lbl_all[sel],
        "diagnosis":   ["ASD" if l == 1 else "Control" for l in lbl_all[sel]],
        "sex":         sex_all[sel],
        "site":        [site_list[i] for i in range(n)],
        "descriptor":  all_descs[:n],
    })
    meta.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  Exported {n}/{n_total} subjects")
    print(f"{'='*60}")
    print(f"  Arrays   : {npz_path}")
    print(f"  Metadata : {csv_path}")
    print(f"  ASD={int((lbl_all[sel] == 1).sum())}  "
          f"Control={int((lbl_all[sel] == 0).sum())}  "
          f"Sites={len(set(site_list[:n]))}")
    print(f"{'='*60}")
    return npz_path


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="./abide_data",
                   help="ABIDE-I download/cache/export root")
    p.add_argument("--n_subjects", default=None,
                   help="number of subjects to export (positive integer, "
                        "or 'all'); omit for an interactive menu "
                        "(presets: 5/10/20/50/100/all/custom)")
    p.add_argument("--out_dir", default=None,
                   help="export directory (default: <data_dir>/exports)")
    return p.parse_args()


def main():
    args = get_args()

    if args.n_subjects is None:
        # Need the true total before the menu can hide presets >= it and
        # accept a validated custom entry, so preprocess/load first.
        (_, _, _, _, lbl_all, *_ ) = load_abide(args.data_dir)
        n_subjects = prompt_subject_count(len(lbl_all))
    else:
        n_subjects = args.n_subjects

    export_subjects(args.data_dir, n_subjects, args.out_dir)


if __name__ == "__main__":
    main()

"""
NeuroLingua-Bridge - test_subject_selection.py
Standalone smoke test for the subject-count export feature
(dataset.resolve_subject_count / prompt_subject_count / download_dataset.export_subjects).

Uses synthetic in-memory data -- no ABIDE-I download, no GPU, no network --
so it runs in a couple seconds anywhere the repo's dependencies are installed.

Usage:
    python test_subject_selection.py
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset import resolve_subject_count, prompt_subject_count
import download_dataset


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def test_resolve_subject_count():
    print("test_resolve_subject_count")
    _check(resolve_subject_count(None, 100) == 100, "None -> all")
    _check(resolve_subject_count("all", 100) == 100, "'all' -> all")
    _check(resolve_subject_count("ALL", 100) == 100, "case-insensitive 'all'")
    _check(resolve_subject_count(10, 100) == 10, "10 of 100 -> 10")
    _check(resolve_subject_count(100, 100) == 100, "exactly total -> total")
    _check(resolve_subject_count(500, 100) == 100, "N > total -> clamped to total")
    _check(resolve_subject_count("20", 100) == 20, "numeric string -> int")

    for bad in (0, -5, "-1"):
        try:
            resolve_subject_count(bad, 100)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            print(f"  OK: {bad!r} correctly rejected (non-positive)")

    for bad in ("abc", "", 3.7):
        try:
            resolve_subject_count(bad, 100)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            print(f"  OK: {bad!r} correctly rejected (invalid)")


def test_prompt_subject_count_presets():
    print("test_prompt_subject_count_presets")
    with patch("builtins.input", side_effect=["2"]):
        result = prompt_subject_count(1000)  # option 2 = "First 10"
    _check(result == 10, f"preset menu option 2 -> 10, got {result}")

    with patch("builtins.input", side_effect=["99", "6"]):  # invalid then "All"
        result = prompt_subject_count(1000)
    _check(result == 1000, f"'all' option -> total, got {result}")

    with patch("builtins.input", side_effect=["7", "37"]):  # custom -> 37
        result = prompt_subject_count(1000)
    _check(result == 37, f"custom entry -> 37, got {result}")

    with patch("builtins.input", side_effect=["7", "-3", "6"]):  # invalid custom -> back to menu -> all
        result = prompt_subject_count(1000)
    _check(result == 1000, f"invalid custom re-prompts, got {result}")

    # small n_total hides presets >= it (n_total=8 hides 10/20/50/100)
    with patch("builtins.input", side_effect=["1"]):
        result = prompt_subject_count(8)  # only preset "First 5" -> option 1
    _check(result == 5, f"presets filtered by n_total, got {result}")


def _synthetic_abide(n=30, n_sites=3):
    rng = np.random.RandomState(0)
    ts_all      = rng.randn(n, 160, 200).astype(np.float32)
    fc_corr_all = rng.randn(n, 200, 200).astype(np.float32)
    fc_tan_all  = fc_corr_all.copy()
    fc_par_all  = fc_corr_all.copy()
    lbl_all     = rng.randint(0, 2, n).astype(np.int64)
    sex_all     = rng.randint(1, 3, n).astype(np.int64)
    site_list   = [f"SITE{i % n_sites}" for i in range(n)]
    all_descs   = [f"descriptor for subject {i}" for i in range(n)]
    site2idx    = {f"SITE{i}": i for i in range(n_sites)}
    site_idx    = np.array([site2idx[s] for s in site_list], dtype=np.int64)
    return (ts_all, fc_corr_all, fc_tan_all, fc_par_all, lbl_all, sex_all,
            site_list, all_descs, site_idx, site2idx)


def test_export_subjects():
    print("test_export_subjects")
    tmp = Path(tempfile.mkdtemp())
    try:
        synthetic = _synthetic_abide(n=30)
        with patch("download_dataset.load_abide", return_value=synthetic):
            path = download_dataset.export_subjects(str(tmp), n_subjects=10)
        _check(path.exists(), "export .npz written")
        z = np.load(path)
        _check(z["lbl_all"].shape[0] == 10, f"exported exactly 10 subjects, got {z['lbl_all'].shape[0]}")
        csv_path = path.with_name(path.stem + "_metadata.csv")
        _check(csv_path.exists(), "metadata CSV written")

        # N > total clamps to total (30)
        with patch("download_dataset.load_abide", return_value=synthetic):
            path_all = download_dataset.export_subjects(str(tmp), n_subjects=999)
        z_all = np.load(path_all)
        _check(z_all["lbl_all"].shape[0] == 30, f"N > total clamps to 30, got {z_all['lbl_all'].shape[0]}")

        # 'all' exports everything
        with patch("download_dataset.load_abide", return_value=synthetic):
            path_str_all = download_dataset.export_subjects(str(tmp), n_subjects="all")
        z_str_all = np.load(path_str_all)
        _check(z_str_all["lbl_all"].shape[0] == 30, "'all' exports every subject")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_resolve_subject_count()
    test_prompt_subject_count_presets()
    test_export_subjects()
    print("\nAll subject-selection tests passed.")

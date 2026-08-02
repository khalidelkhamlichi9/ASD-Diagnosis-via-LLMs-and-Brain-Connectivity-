"""
NeuroLingua-Bridge - test_stage2_checkpoint_loading.py
Standalone smoke test for train_instruction.load_stage2_checkpoint.

Regression test for a real bug: plain `load_state_dict(..., strict=False)`
does NOT tolerate a *shape* mismatch on a key present in both state dicts
(only missing/unexpected keys) -- so any Stage-3 run whose number of
distinct sites differed from the Stage-2 checkpoint's training run used to
crash with a RuntimeError regardless of `strict`. Verifies the fix
(pre-filtering the one legitimate k_sites-dependent mismatch) works, and
that a genuinely incompatible checkpoint is still rejected loudly instead of
silently producing a half-loaded model.

Usage:
    python test_stage2_checkpoint_loading.py
"""

import shutil
import tempfile
from pathlib import Path

import torch

from model_fmrilm_abide import fMRITokenizer, AlignmentModel
from train_instruction import load_stage2_checkpoint

D_LLM = 64


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _save_ckpt(s2, path):
    torch.save({"s2": s2.state_dict(), "d_llm": D_LLM}, path)


def test_matching_k_sites_loads_cleanly():
    print("test_matching_k_sites_loads_cleanly")
    tmp = Path(tempfile.mkdtemp())
    try:
        tok = fMRITokenizer(D_LLM, k_sites=17)
        s2_src = AlignmentModel(tok, D_LLM)
        ckpt = tmp / "ckpt_s2.pt"
        _save_ckpt(s2_src, ckpt)

        tok_dst = fMRITokenizer(D_LLM, k_sites=17)
        s2_dst = AlignmentModel(tok_dst, D_LLM)
        reinit = load_stage2_checkpoint(s2_dst, str(ckpt))
        _check(reinit is False, "identical k_sites -> no re-init reported")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mismatched_k_sites_does_not_crash():
    print("test_mismatched_k_sites_does_not_crash")
    tmp = Path(tempfile.mkdtemp())
    try:
        # Stage 2 trained on the full cohort (17 sites)...
        tok17 = fMRITokenizer(D_LLM, k_sites=17)
        s2_17 = AlignmentModel(tok17, D_LLM)
        ckpt = tmp / "ckpt_s2.pt"
        _save_ckpt(s2_17, ckpt)

        # ...loaded into a Stage-3 run that only saw 5 sites (e.g. a
        # --max_subjects smoke-test subsample). Before the fix this raised
        # RuntimeError: size mismatch for tok.site_disc.net.6.weight.
        tok5 = fMRITokenizer(D_LLM, k_sites=5)
        s2_5 = AlignmentModel(tok5, D_LLM)
        reinit = load_stage2_checkpoint(s2_5, str(ckpt))
        _check(reinit is True, "mismatched k_sites -> re-init reported, no crash")

        # Everything except site_disc.net.6 must have actually loaded (not
        # just skipped), e.g. the encoder/projector weights should now match
        # the checkpoint's, not the fresh random init.
        loaded = dict(s2_5.named_parameters())
        src = dict(s2_17.named_parameters())
        same = torch.allclose(loaded["projector.0.weight"], src["projector.0.weight"])
        _check(same, "non-site_disc weights (e.g. projector) actually transferred")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_genuinely_incompatible_checkpoint_raises():
    print("test_genuinely_incompatible_checkpoint_raises")
    tmp = Path(tempfile.mkdtemp())
    try:
        tok = fMRITokenizer(D_LLM, k_sites=17)
        s2_src = AlignmentModel(tok, D_LLM)
        sd = s2_src.state_dict()
        # Drop an unrelated module entirely -- simulates a checkpoint from
        # an incompatible/older architecture, which must NOT be silently
        # accepted.
        corrupted = {k: v for k, v in sd.items() if not k.startswith("projector")}
        ckpt = tmp / "ckpt_s2_bad.pt"
        torch.save({"s2": corrupted, "d_llm": D_LLM}, ckpt)

        tok_dst = fMRITokenizer(D_LLM, k_sites=17)
        s2_dst = AlignmentModel(tok_dst, D_LLM)
        try:
            load_stage2_checkpoint(s2_dst, str(ckpt))
            raise AssertionError("expected RuntimeError for missing projector keys")
        except RuntimeError as e:
            _check("projector" in str(e), f"missing-module checkpoint correctly rejected: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_genuinely_incompatible_shape_raises():
    print("test_genuinely_incompatible_shape_raises")
    tmp = Path(tempfile.mkdtemp())
    try:
        tok = fMRITokenizer(D_LLM, k_sites=17)
        s2_src = AlignmentModel(tok, D_LLM)
        sd = dict(s2_src.state_dict())
        # Corrupt a non-site_disc weight's shape directly -- must be
        # rejected, not silently dropped like the expected site_disc case.
        sd["projector.0.weight"] = sd["projector.0.weight"][:1]
        ckpt = tmp / "ckpt_s2_bad_shape.pt"
        torch.save({"s2": sd, "d_llm": D_LLM}, ckpt)

        tok_dst = fMRITokenizer(D_LLM, k_sites=17)
        s2_dst = AlignmentModel(tok_dst, D_LLM)
        try:
            load_stage2_checkpoint(s2_dst, str(ckpt))
            raise AssertionError("expected RuntimeError for corrupted projector shape")
        except RuntimeError as e:
            _check("projector" in str(e), f"corrupted-shape checkpoint correctly rejected: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_matching_k_sites_loads_cleanly()
    test_mismatched_k_sites_does_not_crash()
    test_genuinely_incompatible_checkpoint_raises()
    test_genuinely_incompatible_shape_raises()
    print("\nAll Stage-2 checkpoint-loading tests passed.")

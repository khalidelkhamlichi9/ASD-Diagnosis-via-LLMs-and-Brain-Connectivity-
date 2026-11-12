# NeuroLingua-Bridge

## Diagnosis of Autism Spectrum Disorder using LLMs and Multimodal Brain Connectivity Analysis

---

## Overview

**NeuroLingua-Bridge** maps resting-state fMRI into the token space of a large
language model while remaining invariant to the acquisition site. On ABIDE-I,
scanner and protocol differences across the 17 sites dominate the diagnostic
signal and cause language-aligned fMRI representations to degrade sharply
across sites.

The framework trains a text-aligned tokenizer jointly with three
domain-generalization (DG) components:

1. **Multi-class site-adversarial GRL** — a 17-way site discriminator behind a
   gradient-reversal layer. At any saddle point of the minimax objective the
   latent representation satisfies I(z̄; s) = 0, generalizing the two-domain
   DANN setting to K = 17.
2. **Dynamic latent prototypes** — two learnable class anchors in the invariant
   embedding space, updated by an EMA of *site-balanced* class means so that no
   single site dominates a prototype.
3. **Episodic Group-DRO** — site-as-task meta-learning that optimizes the
   worst-site risk instead of the average, since ERM lets large sites dominate.

**Main backbone:** DeepSeek-Coder-1.3B (2048-d embedding space)
**Also evaluated:** GPT-2 Medium, ClinicalT5-large, Grok, Gemini

---

## Data & Preprocessing

- **Dataset:** ABIDE-I (17 acquisition sites)
- **Pipeline:** CPAC
- **Atlas:** CC200 - 200 ROIs, used natively (no re-parcellation)
- **Temporal:** interpolated to TR = 2.0 s, T = 160 time points
- **Normalization:** z-scored, then site-wise variance-normalized

---

## Evaluation Protocol

Strict **leave-site-out**: target sites are held out and unseen at test time.
Training *and* checkpoint selection use source-site data only. All results are
mean ± std over **5 seeds**. The worst site is **CMU** (n ≈ 10) in every
configuration; its small n explains the wider variance on the worst-site metric.

---

## Repository Structure

```
NeuroLingua-Bridge/
├── dataset.py                        # ABIDE-I loader (CC200, 200 ROIs, 17 sites, inter-site split)
├── download_dataset.py               # Download/preprocess + export a chosen subject count
├── test_subject_selection.py         # Smoke tests for the subject-count export feature
├── model_fmrilm_abide.py             # Encoder + VQ + 17-way site-adversarial GRL + prototypes
│                                      # (self-contained: encoder, VQ codebook, classifier all live here)
├── train_tokenizer.py                # Stage 1: tokenizer (L_quant + L_CLIP + L_site + L_proto)
├── train_pretrain_paired.py          # Stage 2: frozen-LLM alignment (F2F + F2T)
├── train_instruction.py              # Stage 3: ASD classifier + episodic Group-DRO
├── eval_zeroshot.py                  # Evaluation on held-out unseen sites
├── utils_loss.py                     # Shared loss terms (siglip_loss)
├── language_models/
│   └── backends.py                   # 5 LLM backends (gpt2, clinicalt5, deepseek, grok, gemini)
├── chatbot/
│   └── fmri_chatbot.py               # 3-paradigm clinical chatbot interface (standalone demo, Fig. 3)
├── configs/                          # dataset_config.yaml, vit_base_p32/p160.yaml, deepspeed_zero2.json
├── scripts/                          # run_deepseek.sh, run_gpt2.sh, run_clinicalt5.sh, run_grok.sh, run_gemini.sh
├── nbs_data/                         # preprocessing_fmri.py, get_fmri_discriptor.py, prompt_expansion.py
├── abide_data/                       # ABIDE-I data root (--data_dir)
├── NeuroLingua_Bridge_Colab.ipynb    # Colab-ready runnable notebook (start here)
├── NeuroLingua_Bridge_DomainGen.ipynb# Domain-generalization focused notebook
├── requirements.txt
└── paper/                            # LaTeX source + compiled PDF
```

---

## Quick Start (Google Colab)

1. Open `NeuroLingua_Bridge_Colab.ipynb` in Google Colab
2. `Runtime → Change runtime type → GPU (T4)`
3. `Runtime → Run all`

Set `QUICK_TEST = True` (default) for a fast smoke-run on a subset.
Set `QUICK_TEST = False` for the full experiment on ABIDE-I.

---

## Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/NeuroLingua-Bridge.git
cd NeuroLingua-Bridge
pip install -r requirements.txt
```

---

## Downloading & Exporting a Subset of Subjects

`download_dataset.py` downloads/preprocesses ABIDE-I once (cached on disk by
`dataset.load_abide`, so re-running is instant) and lets you export just the
first *N* preprocessed subjects instead of the whole cohort — useful for
quick local inspection or sharing a small sample.

```bash
# Interactive menu: first 5 / 10 / 20 / 50 / 100 / all / custom N
python download_dataset.py

# Non-interactive
python download_dataset.py --n_subjects 20
python download_dataset.py --n_subjects all
```

Input is validated: non-positive or non-numeric values are rejected, and a
count larger than the number of available subjects is automatically clamped
to the full cohort (with a printed notice) rather than erroring out. The
export writes two files to `<data_dir>/exports/`:

- `abide_export_<N>.npz` — `ts_all`, `fc_corr_all`, `fc_tan_all`,
  `fc_par_all`, `lbl_all`, `sex_all`, `site_idx` for the selected subjects,
  in the same array layout as the internal preprocessing cache.
- `abide_export_<N>_metadata.csv` — one row per subject (index, label,
  sex, site, fMRI-text descriptor).

This is purely a post-processing export step: it never changes what
`train_tokenizer.py` / `train_pretrain_paired.py` / `train_instruction.py`
/ `eval_zeroshot.py` do with `--max_subjects` (their own, separate,
*random* subsampling flag used for fast smoke tests — see below).

---

## Running the Pipeline

The three stages are sequential: each one consumes the checkpoint produced by
the previous stage. `--lm_name` selects the backbone and accepts
`deepseek`, `gpt2`, `clinicalt5`, `grok`, or `gemini`.

---

---

## ⚡ High-Performance GPU Execution (NVIDIA A100 / V100)

To reproduce the exact **State-of-the-Art (SOTA)** paper results reported in **Table 1**, execute the pipeline on an NVIDIA A100 or V100 GPU.

> **Expected SOTA Metrics (Table 1 Headline Results):**
> - **Accuracy:** **80.0% ± 0.7%**
> - **AUC-ROC:** **81.1% ± 0.8%**
> - **Worst-Site (CMU):** **74.1% ± 1.3%**
> - *Total GPU Runtime: ~1 hour (A100)*

---

### 🏆 1. NeuroLingua-Bridge SOTA Pipeline (DeepSeek-Coder 1.3B)

```bash
# Stage 1 — fMRI Tokenizer (GRL + Dynamic Prototypes + VICReg)
python train_tokenizer.py --lm_name deepseek --data_dir ./abide_data --epochs 50 --batch_size 256

# Stage 2 — Frozen-LLM Alignment (SigLIP Contrastive fMRI↔Text)
python train_pretrain_paired.py --lm_name deepseek --data_dir ./abide_data --ckpt_tok ckpt_tok_deepseek_seed42.pt --epochs 30 --batch_size 256

# Stage 3 — ASD Classifier + Full LLM Unfreezing + Group-DRO (SOTA)
python train_instruction.py --lm_name deepseek --data_dir ./abide_data --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 200 --batch_size 256 --lr 5e-4 --weight_decay 5e-2 --harmonize True --finetune_encoder True --patience 30

# Evaluate on Unseen Test Sites
python eval_zeroshot.py --lm_name deepseek --data_dir ./abide_data --checkpoint ckpt_clf_deepseek_seed42.pt
```

---

### 📊 2. Full 5-Seed Paper Benchmark (Table 1 Reproducibility)

Run the full 5-seed statistical evaluation in a single command:

```bash
python run_full_paper_5seeds.py --lm_name deepseek --epochs 200 --batch_size 256 --lr 5e-4 --weight_decay 5e-2
```

---

### 🔬 3. Scénario 2 — Backbone (BrainGNN / SWiFT / ComBat) + DeepSeek LLM + DG

Evaluates alternative neuroimaging backbones coupled with the DeepSeek LLM alignment and Domain Generalization modules on GPU A100:

```bash
# BrainGNN + DeepSeek LLM + GRL + Group-DRO
python train_backbone_llm.py --backbone braingnn --lm_name deepseek --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 50 --batch_size 32 --lr 3e-4 --seed 42

# SWiFT (Sliding Window Transformer) + DeepSeek LLM + GRL + Group-DRO
python train_backbone_llm.py --backbone swift --lm_name deepseek --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 50 --batch_size 32 --lr 3e-4 --seed 42

# ComBat (EB) + DeepSeek LLM + GRL + Group-DRO
python train_backbone_llm.py --backbone combat --lm_name deepseek --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 50 --batch_size 32 --lr 3e-4 --seed 42
```

---

### 📉 4. Full Baseline Benchmark (9 Models under LSO)

Run all 9 baseline models under identical Leave-Site-Out (LSO) protocol:

```bash
# Run ALL 9 baseline models in sequence + generate unified Markdown/LaTeX table
python baselines/loso_runner.py --model all --seed 42 --generate_table
```

---

### 💻 Fast Local Test Run (CPU Mode)

If running locally on CPU for code verification or smoke testing:

```bash
# Quick Stage 1-3 local test
python train_tokenizer.py --lm_name deepseek --data_dir ./abide_data --epochs 5 --batch_size 4
python train_pretrain_paired.py --lm_name deepseek --data_dir ./abide_data --ckpt_tok ckpt_tok_deepseek_seed42.pt --epochs 5 --batch_size 4
python train_instruction.py --lm_name deepseek --data_dir ./abide_data --ckpt_s2 ckpt_s2_deepseek_seed42.pt --epochs 10 --batch_size 8 --lr 5e-4 --harmonize True --patience 5
```

---

### 📊 Backbone Comparison Study (Full — 9 Models)

All models run under the **exact same fixed Leave-Site-Out (LSO) protocol**:
- **Train**: NYU, UM, UCLA, USM, LEUVEN, YALE, PITT (7 large sites)
- **Test**: MAX_MUN, KKI, TRINITY, STANFORD, OLIN, SDSU, SBL, OHSU, CMU, CALTECH (10 unseen sites)

#### 🚀 One Command — Run All Baselines

```bash
# Run ALL 9 baseline models in sequence + print unified comparison table
python baselines/loso_runner.py --model all --seed 42 --generate_table
```

#### Run Individual Baselines

```bash
# Tier 1 — Classic Models (no pre-training needed)
python baselines/combat_baseline.py    --seed 42   # ComBat (EB) + Robust MLP
python baselines/bnt_baseline.py       --seed 42   # Brain Network Transformer
python baselines/brainnetcnn_baseline.py --seed 42 # BrainNetCNN
python baselines/braingnn_baseline.py  --seed 42   # BrainGNN (custom TopK GNN)
python baselines/fbnetgen_baseline.py  --seed 42   # FBNetGen (FC generator)
python baselines/swift_baseline.py     --seed 42   # SWiFT (fMRI time series transformer)

# Tier 2 — Foundation Models (scratch, no external pre-training)
python baselines/brainlm_baseline.py   --seed 42   # BrainLM* (VAE-style)
python baselines/brainmass_baseline.py --seed 42   # BrainMass* (masked autoencoder)
python baselines/brainjepa_baseline.py --seed 42   # Brain-JEPA* (JEPA self-supervised)

# Generate unified Markdown + LaTeX comparison table
python baselines/results_table.py
```

#### Expected Metrics per Model

| Model | Acc | AUC | F1 | MCC | Worst-Site† | Time (CPU) |
|:------|:---:|:---:|:---:|:---:|:---:|:---:|
| ComBat + MLP | ~59% | ~62% | ~59% | — | ~43% | ~2 min |
| BNT (K=8) | ~54% | ~57% | ~54% | — | ~40% | ~5 min |
| BrainNetCNN | ~57% | ~58% | ~56% | — | ~40% | ~8 min |
| BrainGNN | TBD | TBD | TBD | TBD | TBD | ~10 min |
| FBNetGen | TBD | TBD | TBD | TBD | TBD | ~15 min |
| SWiFT | TBD | TBD | TBD | TBD | TBD | ~20 min |
| BrainLM\* | TBD | TBD | TBD | TBD | TBD | ~25 min |
| BrainMass\* | TBD | TBD | TBD | TBD | TBD | ~30 min |
| Brain-JEPA\* | TBD | TBD | TBD | TBD | TBD | ~30 min |
| **NeuroLingua-Bridge** | **~80%** | **~81%** | **~80%** | — | **74.1%** | ~60 min (GPU) |

> `*` = Trained from scratch on ABIDE-I (no external pre-training corpus)
>
> `†` **The worst site is CMU (n ≈ 10) in all configurations**; the small n explains the wider
> variance on W-Site. This is consistent with Table 1, footnote † of the paper.
> W-Site = min accuracy across all 10 test sites.

---



## Key Scientific Contribution

This work addresses specific limitations of **fMRI-LM**
(Wei et al., arXiv:2511.21760):

| fMRI-LM | NeuroLingua-Bridge |
|---------|--------------------|
| Binary fMRI-vs-text adversary (no site label) | **17-way site adversary** → I(z̄; s) = 0 |
| No geometric structure in embedding space | **Dynamic latent prototypes** (EMA, site-balanced) |
| ERM training (dominated by large sites) | **Episodic Group-DRO** (worst-site risk) |
| Within-site / mixed-site evaluation | **Strict inter-site protocol** (unseen test sites) |
| Schaefer-400 + Tian (450 ROIs) | **CC200 atlas (200 ROIs, native)** |

---

## Results

### Ablation of DG modules (DeepSeek-Coder-1.3B, 5 seeds)

| Configuration | Acc (%) | AUC (%) | F1 (%) | Worst-site (%) |
|---------------|---------|---------|--------|----------------|
| GRL disabled (λ = 0) | 68.4 ± 1.2 | 69.0 ± 1.3 | 68.3 ± 1.1 | 58.2 ± 2.1 |
| + GRL | 74.5 ± 1.0 | 75.8 ± 1.1 | 74.4 ± 1.0 | 66.8 ± 1.8 |
| + GRL + Prototypes | 77.2 ± 0.9 | 78.6 ± 0.9 | 77.1 ± 0.8 | 70.9 ± 1.5 |
| **+ GRL + Proto + G-DRO** | **80.0 ± 0.7** | **81.1 ± 0.8** | **79.9 ± 0.7** | **74.1 ± 1.3** |

Each module contributes: GRL adds +6.1 mean / +8.6 worst-site; prototypes add a
further +2.7 / +4.1; Group-DRO brings the total to **+11.6 mean / +15.9
worst-site** over the no-DG baseline — the largest gains exactly where
deployment is hardest.

### Full DG pipeline across backbones (5 seeds)

| Backbone | Acc (%) | AUC (%) | F1 (%) | Worst-site (%) |
|----------|---------|---------|--------|----------------|
| GPT-2 Medium | 76.0 ± 1.1 | 76.2 ± 1.2 | 75.8 ± 1.1 | 64.3 ± 1.9 |
| ClinicalT5-large | 77.0 ± 1.0 | 77.9 ± 1.1 | 77.0 ± 1.0 | 66.1 ± 1.7 |
| **DeepSeek-1.3B (main)** | **80.0 ± 0.7** | **81.1 ± 0.8** | 79.9 ± 0.7 | **74.1 ± 1.3** |
| Grok (frozen emb.)¹ | 79.0 ± 0.9 | 79.6 ± 0.9 | 79.0 ± 0.8 | 69.8 ± 1.6 |
| Gemini (frozen emb.)¹ | 80.0 ± 0.8 | 80.9 ± 0.8 | **80.0 ± 0.8** | 72.5 ± 1.4 |

¹ Grok-1 and Gemini 1.5 Flash are API-only; we align to their frozen text
embeddings, not the models themselves.

All five backbones improve over the no-DG baseline, and the top three
(DeepSeek, Gemini, Grok) are close — the gains come from the DG paradigm, not
from one specific backbone.

### Comparison with state of the art (ABIDE-I)

| Type | Method | Acc (%) | AUC (%) | Worst-site (%) |
|------|--------|---------|---------|----------------|
| Supervised | BrainNetCNN | 66.8 | 68.1 | N/A |
| Supervised | BrainGNN | 67.3 | 69.0 | N/A |
| Supervised | BNT | 68.5 | 70.2 | N/A |
| Supervised | FBNetGen | 65.9 | 67.4 | N/A |
| Supervised | SwiFT | 66.2 | 68.0 | N/A |
| Foundation | BrainLM | 71.2 | 72.6 | N/A |
| Foundation | BrainMass | 70.4 | 72.1 | N/A |
| Foundation | Brain-JEPA | 72.0 | 73.3 | N/A |
| **Ours** | **NeuroLingua-Bridge** | **80.0** | **81.1** | **74.1** |

Baselines use within-site or mixed-site protocols, so the worst-site metric is
undefined for them (N/A) and the comparison is not fully controlled.
NeuroLingua-Bridge reaches the highest accuracy and AUC (+8.0 / +7.8 over
Brain-JEPA) *despite* the stricter inter-site protocol with unseen scanners at
test time, and is the only method reporting a worst-site figure.

---

## Paper & Preprint

> **Diagnosis of Autism Spectrum Disorder using LLMs and Multimodal Brain Connectivity Analysis**
> Khalid El Khamlichi, Fatima Ez-Zahraa Bazay, Ahmed Drissi El Maliani
> *arXiv preprint arXiv:2511.21760 (2026)*

```
@article{elkhamlichi2026neurolinguabridge,
  title   = {Diagnosis of Autism Spectrum Disorder using LLMs
             and Multimodal Brain Connectivity Analysis},
  author  = {El Khamlichi, Khalid and Bazay, Fatima Ez-Zahraa
             and Drissi El Maliani, Ahmed},
  journal = {arXiv preprint arXiv:2511.21760},
  year    = {2026}
}
```

---

## Compiling the Paper (LaTeX)

Both `main.tex` and `supplementary.tex` use the Springer LNCS template (`llncs.cls`).
Run the following sequence from the project root:

```bash
# Main paper (8 pages)
pdflatex  main.tex
bibtex    main
pdflatex  main.tex
pdflatex  main.tex

# Supplementary material
pdflatex  supplementary.tex
bibtex    supplementary
pdflatex  supplementary.tex
pdflatex  supplementary.tex
```

> **Dependencies:** `llncs.cls`, `splncs04.bst`, `booktabs`, `multirow`,
> `subcaption`, `hyperref`, `cleveref`, `amsmath`, `amssymb`, `amsthm`,
> `graphicx`, `caption`, `etoolbox`, `float`, `placeins`.
> All are included in a standard TeX Live / MiKTeX installation.

---

## Publishing to GitHub

```bash
# First-time setup
git init
git remote add origin https://github.com/khalidelkhamlichi9/ASD-Diagnosis-via-LLMs-and-Brain-Connectivity-.git

# Stage, commit, and push
git add .
git commit -m "feat: NeuroLingua-Bridge — final release (arXiv:2511.21760)"
git branch -M main
git push -u origin main

# Subsequent updates
git add .
git commit -m "fix: <description of change>"
git push
```

> **Note:** checkpoint files (`*.pt`) are large (50–111 MB each).
> Add them to Git LFS or `.gitignore` if you do not want to push them:
>
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> ```

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## Contact

**Khalid El Khamlichi** — [khalid_elkhamlichi@um5.ac.ma](mailto:khalid_elkhamlichi@um5.ac.ma)
Faculty of Sciences, Mohammed V University, Rabat, Morocco.

Issues and pull requests are welcome.

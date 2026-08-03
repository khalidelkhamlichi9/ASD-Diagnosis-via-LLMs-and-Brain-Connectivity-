"""
NeuroLingua-Bridge - dataset.py
ABIDE-I dataset: download via Nilearn, preprocessing, fMRI-text descriptors,
integer site indexing, inter-site split and PyTorch DataLoaders.

Preprocessing (fMRI-LM baseline, Wei et al. arXiv:2511.21760v3 (2026)):
  - CPAC pipeline, CC200 atlas (Craddock 200, 200 ROIs, native)
  - Resample to T=160 timepoints
  - 3 FC matrices: Correlation (Fisher-Z), Tangent, Partial Correlation
  - Robust IQR z-score per site
  - 4 text descriptor types: FC, FG, ICA, Graph

NeuroLingua-Bridge domain-generalization additions (Sec. 3.1, 3.4, 3.7):
  - Integer `site_idx` per subject (0..K-1, K=17 for ABIDE-I): required as
    the target class for the multi-class site-adversarial GRL (Eq. 3), the
    grouping key for the site-balanced prototype EMA (Eq. 5), and the group
    label for episodic Group-DRO (Eq. 6).
  - Inter-site split: large sites (top 70 % subjects) -> Train | small -> Test
    (unseen-scanner generalization protocol).
"""

import os, math, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.linalg import eigh as sp_eigh
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import networkx as nx

try:
    from nilearn.datasets import fetch_abide_pcp
    from nilearn.connectome import ConnectivityMeasure
    NILEARN_OK = True
except ImportError:
    NILEARN_OK = False

SEED        = 42
N_ROIS      = 200
T_TARGET    = 160
MIN_TP      = 50
TRAIN_RATIO = 0.70

# ABIDE-I's raw SITE_ID has 20 distinct values because three institutions
# scanned subjects on two visits/scanners (UM_1/UM_2, UCLA_1/UCLA_2,
# LEUVEN_1/LEUVEN_2), which the literature (and this paper's "17 sites of
# ABIDE-I") treats as a single acquisition site each. Left unmerged, the
# site-adversarial GRL/prototypes/Group-DRO would train a spurious 20-way
# split instead of the paper's K=17.
_SITE_MERGE = {"UM_1": "UM", "UM_2": "UM",
               "UCLA_1": "UCLA", "UCLA_2": "UCLA",
               "LEUVEN_1": "LEUVEN", "LEUVEN_2": "LEUVEN"}

def _canonical_site(raw_site: str) -> str:
    return _SITE_MERGE.get(raw_site, raw_site)

NET_NAMES = ["Visual", "SomMot", "DorsAttn", "SalVentAttn",
             "Limbic", "Cont", "Default", "Subcort"]
# CC200 has ~200 ROIs; partition into 8 contiguous functional blocks of 25.
_BLK = N_ROIS // 8
NET_ROIS  = [list(range(i * _BLK, (i + 1) * _BLK)) for i in range(7)] + \
            [list(range(7 * _BLK, N_ROIS))]

# ICA networks: 7 overlapping blocks across the 200 ROIs.
ICA_ROIS  = [list(range(i * _BLK, (i + 1) * _BLK)) for i in range(7)]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample(ts, T=T_TARGET):
    if ts.shape[0] == T: return ts
    f = interp1d(np.linspace(0, 1, ts.shape[0]), ts, axis=0, kind="linear")
    return f(np.linspace(0, 1, T)).astype(np.float32)

def _fit_rois(ts, N=N_ROIS):
    """CC200 is natively ~200 ROIs. Pad or truncate to exactly N without
    fabricating a cross-region interpolation (the AAL->450 hack is removed)."""
    T, n = ts.shape
    if n == N:
        return ts.astype(np.float32)
    if n > N:
        return ts[:, :N].astype(np.float32)
    out = np.zeros((T, N), np.float32)
    out[:, :n] = ts
    return out

def _robust_zscore(ts):
    q75, q25 = np.percentile(ts, 75, axis=0), np.percentile(ts, 25, axis=0)
    return (ts - np.median(ts, axis=0)) / (q75 - q25 + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Text descriptor builders  (Table 1, Wei et al. 2026)
# ─────────────────────────────────────────────────────────────────────────────

def _fc_descriptor(fc):
    pv = {f"{n1}-{n2}": float(fc[np.ix_(r1, r2)].mean())
          for i, (n1, r1) in enumerate(zip(NET_NAMES, NET_ROIS))
          for j, (n2, r2) in enumerate(zip(NET_NAMES, NET_ROIS)) if i < j}
    top3 = sorted(pv, key=pv.get, reverse=True)[:3]
    bot3 = sorted(pv, key=pv.get)[:3]
    return (f"FC top=[{','.join(f'{k}({pv[k]:+.2f})' for k in top3)}] "
            f"bot=[{','.join(f'{k}({pv[k]:+.2f})' for k in bot3)}].")

def _fg_descriptor(fc):
    fc_pos = np.maximum(fc, 0.); np.fill_diagonal(fc_pos, 0.)
    Ln = np.eye(fc.shape[0]) - fc_pos / (fc_pos.sum(1, keepdims=True) + 1e-8)
    try:
        _, V = sp_eigh(Ln, subset_by_index=[1, 4])
        g = [float(V[:, k].max() - V[:, k].min()) for k in range(min(3, V.shape[1]))]
        while len(g) < 3: g.append(0.)
    except Exception:
        g = [0., 0., 0.]
    return f"FG g1={g[0]:.3f} g2={g[1]:.3f} g3={g[2]:.3f}."

def _ica_descriptor(ts):
    sigs = [ts[:, [c for c in r if c < ts.shape[1]]].mean(1)
            if any(c < ts.shape[1] for c in r) else np.zeros(ts.shape[0])
            for r in ICA_ROIS]
    amp = [float(np.abs(s).mean()) for s in sigs]
    var = [float(s.std()) for s in sigs]
    fnc = float(np.corrcoef(np.array(sigs))[np.triu_indices(len(sigs), k=1)].mean())
    return f"ICA amp={np.mean(amp):.3f} var={np.mean(var):.3f} FNC={fnc:.3f}."

def _graph_descriptor(fc):
    np.fill_diagonal(fc, 0.)
    cut = np.percentile(np.abs(fc[np.triu_indices(fc.shape[0], k=1)]), 90)
    A   = np.where(np.abs(fc) >= cut, np.abs(fc), 0.)
    G   = nx.from_numpy_array(A)
    try:
        eff   = nx.global_efficiency(G)
        clust = nx.average_clustering(G, weight="weight")
        parts = nx.community.louvain_communities(G, weight="weight", seed=SEED)
        mod   = nx.community.modularity(G, parts, weight="weight")
    except Exception:
        eff = clust = mod = 0.
    return f"Graph mod={mod:.3f} eff={eff:.3f} clust={clust:.3f}."

def build_descriptor(ts, fc):
    return " ".join([_fc_descriptor(fc), _fg_descriptor(fc),
                     _ica_descriptor(ts), _graph_descriptor(fc)])


# ─────────────────────────────────────────────────────────────────────────────
# Subject-count selection (export / download-time subsetting)
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_COUNT_PRESETS = [5, 10, 20, 50, 100]


def resolve_subject_count(requested, n_total: int) -> int:
    """
    Validate and clamp a user-requested subject export count against the
    number of subjects actually available after preprocessing.

    Args:
        requested: desired subject count. `None` (or the string "all")
            means "export every available subject". Otherwise must be a
            positive integer.
        n_total: number of subjects available to export from.

    Returns:
        The number of subjects to actually export: `n_total` if `requested`
        is None/"all" or exceeds `n_total`, otherwise `requested`.

    Raises:
        ValueError: if `requested` is not a positive integer and not
            None/"all".
    """
    if requested is None or (isinstance(requested, str) and requested.strip().lower() == "all"):
        return n_total
    if isinstance(requested, bool):
        raise ValueError(f"Number of subjects must be a positive integer or 'all', got {requested!r}.")
    if isinstance(requested, float):
        if not requested.is_integer():
            raise ValueError(f"Number of subjects must be a whole number, got {requested!r}.")
        n = int(requested)
    elif isinstance(requested, (int, np.integer)):
        n = int(requested)
    elif isinstance(requested, str):
        s = requested.strip()
        if not (s.lstrip("-").isdigit()):
            raise ValueError(f"Number of subjects must be a positive integer or 'all', got {requested!r}.")
        n = int(s)
    else:
        raise ValueError(f"Number of subjects must be a positive integer or 'all', got {requested!r}.")
    if n <= 0:
        raise ValueError(f"Number of subjects must be a positive integer, got {n}.")
    if n > n_total:
        print(f"Requested {n} subjects but only {n_total} are available -- exporting all {n_total}.")
        return n_total
    return n


def prompt_subject_count(n_total: int) -> int:
    """
    Interactive menu for choosing how many (preprocessed) subjects to
    export, offered as fixed presets plus a manual entry option. Presets
    larger than `n_total` are hidden since they would just export everything.

    Returns the resolved, validated subject count (never raises on bad
    input -- re-prompts instead, since this is meant for interactive use).
    """
    options = [p for p in SUBJECT_COUNT_PRESETS if p < n_total]
    print(f"\n{len(options) + 2} export options ({n_total} subjects available):")
    for i, p in enumerate(options, start=1):
        print(f"  {i}. First {p} subjects")
    print(f"  {len(options) + 1}. All {n_total} subjects")
    print(f"  {len(options) + 2}. Enter a custom number")

    while True:
        choice = input(f"Choose an option [1-{len(options) + 2}]: ").strip()
        if choice == str(len(options) + 1):
            return n_total
        if choice == str(len(options) + 2):
            custom = input("Enter number of subjects: ").strip()
            try:
                return resolve_subject_count(custom, n_total)
            except ValueError as e:
                print(f"  Invalid input: {e}")
                continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  Invalid choice, please try again.")


# ─────────────────────────────────────────────────────────────────────────────
# ABIDE-I download + full preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_abide(data_dir: str = "./abide_data", max_subjects: int = None,
               use_cache: bool = True):
    """
    Download ABIDE-I (via Nilearn) and return fully preprocessed arrays.

    Every field returned here is deterministic given (data_dir, max_subjects)
    -- preprocessing does not depend on the training seed -- so this wraps
    `_preprocess_abide` with an on-disk .npz/.json cache. Without it, each of
    train_tokenizer.py / train_pretrain_paired.py / train_instruction.py /
    eval_zeroshot.py (and every seed of a multi-seed sweep) would redo the
    full FC + graph-descriptor pipeline over the entire cohort from scratch.

    Args:
        max_subjects: if set, subsample to this many subjects *before*
            running the (expensive) per-subject FC/descriptor pipeline.
            Intended for fast smoke tests / CI on a laptop CPU -- the
            paper's reported numbers use the full ABIDE-I cohort.
        use_cache: set False to force a fresh preprocessing pass.

    Returns
    -------
    ts_all      : (N, T, ROI) float32
    fc_corr_all : (N, ROI, ROI) float32
    fc_tan_all  : (N, ROI, ROI) float32
    fc_par_all  : (N, ROI, ROI) float32
    lbl_all     : (N,) int64  - 1=ASD, 0=Control
    sex_all     : (N,) int64
    site_list   : list[str]        - raw ABIDE SITE_ID strings (kept for the
                                      inter-site split, which reasons about
                                      subject counts per raw site name)
    all_descs   : list[str]
    site_idx    : (N,) int64       - integer site id in [0, K-1], K=17 for
                                      ABIDE-I. Target class for the 17-way
                                      site-adversarial GRL (Eq. 3), grouping
                                      key for the prototype EMA (Eq. 5) and
                                      the Group-DRO group label (Eq. 6).
    site2idx    : dict[str, int]   - mapping raw SITE_ID -> site_idx, sorted
                                      alphabetically for reproducibility.
    """
    Path(data_dir).mkdir(exist_ok=True)
    tag       = str(max_subjects) if max_subjects is not None else "full"
    cache_npz = Path(data_dir) / f"preprocessed_cache_{tag}.npz"
    cache_json = Path(data_dir) / f"preprocessed_cache_{tag}.json"

    if use_cache and cache_npz.exists() and cache_json.exists():
        print(f"Loading cached preprocessed ABIDE-I ({tag}) from {cache_npz} ...")
        z = np.load(cache_npz, allow_pickle=True)
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
        return (z["ts_all"], z["fc_corr_all"], z["fc_tan_all"], z["fc_par_all"],
                z["lbl_all"], z["sex_all"], meta["site_list"], meta["all_descs"],
                z["site_idx"], {k: int(v) for k, v in meta["site2idx"].items()})

    result = _preprocess_abide(data_dir, max_subjects)
    if use_cache:
        (ts_all, fc_corr_all, fc_tan_all, fc_par_all, lbl_all, sex_all,
         site_list, all_descs, site_idx, site2idx) = result
        np.savez_compressed(cache_npz, ts_all=ts_all, fc_corr_all=fc_corr_all,
                           fc_tan_all=fc_tan_all, fc_par_all=fc_par_all,
                           lbl_all=lbl_all, sex_all=sex_all, site_idx=site_idx)
        cache_json.write_text(json.dumps({"site_list": list(site_list),
                                          "all_descs": all_descs,
                                          "site2idx": site2idx}), encoding="utf-8")
        print(f"Cached preprocessed ABIDE-I ({tag}) -> {cache_npz}")
    return result


def _preprocess_abide(data_dir: str, max_subjects: int = None):
    assert NILEARN_OK, "pip install nilearn"
    Path(data_dir).mkdir(exist_ok=True)
    print("Downloading ABIDE-I (pipeline=cpac, rois_cc200, 200 ROIs)...")
    abide    = fetch_abide_pcp(data_dir=data_dir, pipeline="cpac",
                               band_pass_filtering=True,
                               global_signal_regression=False,
                               derivatives=["rois_cc200"],
                               quality_checked=False, verbose=0)
    pheno    = pd.DataFrame(abide.phenotypic)
    qc_mask  = pheno.get("qc_rater_1", pd.Series([""] * len(pheno))).astype(str).str.upper() == "OK"
    qc_idx   = np.where(qc_mask)[0]
    pheno    = pheno.iloc[qc_idx].reset_index(drop=True)
    rois_qc  = [abide.rois_cc200[i] for i in qc_idx]

    dx_arr   = pd.to_numeric(pheno["DX_GROUP"],  errors="coerce").fillna(0).values
    sex_raw  = pd.to_numeric(pheno["SEX"], errors="coerce").fillna(2).astype(int).values
    site_raw = np.array([_canonical_site(s) for s in
                         pheno.get("SITE_ID", pd.Series(["UNK"] * len(pheno))).values])

    valid_idx = [i for i, ts in enumerate(rois_qc)
                 if np.array(ts).ndim == 2 and np.array(ts).shape[0] > 10]

    if max_subjects is not None and max_subjects < len(valid_idx):
        rng = np.random.RandomState(SEED)
        valid_idx = list(rng.choice(valid_idx, size=max_subjects, replace=False))
        print(f"  (smoke-test subsample: {max_subjects}/{len(rois_qc)} subjects)")

    roi_all  = [rois_qc[i] for i in valid_idx]
    sex_v    = sex_raw[valid_idx]
    site_v   = site_raw[valid_idx]

    conn_c = ConnectivityMeasure(kind="correlation",         vectorize=False)
    conn_p = ConnectivityMeasure(kind="partial correlation", vectorize=False)
    buf    = {"ts": [], "fc_c": [], "fc_p": [], "lbl": [], "sex": [], "site": []}

    for i, roi_data in enumerate(tqdm(roi_all, desc="Preprocessing")):
        try:
            ts = np.array(roi_data, dtype=np.float32)
            if ts.shape[0] < ts.shape[1]: ts = ts.T
            if ts.shape[0] < MIN_TP: continue
            ts = _resample(ts)
            ts = _fit_rois(ts)
            ts = _robust_zscore(ts).astype(np.float32)

            fc_c = conn_c.fit_transform([ts])[0]
            np.fill_diagonal(fc_c, 0.)
            fc_c = np.arctanh(np.clip(fc_c, -0.999, 0.999)).astype(np.float32)

            try:
                fc_p = conn_p.fit_transform([ts])[0]
                np.fill_diagonal(fc_p, 0.)
                fc_p = fc_p.astype(np.float32)
            except Exception:
                fc_p = fc_c.copy()

            lbl = 1 if int(dx_arr[valid_idx[i]]) == 1 else 0
            buf["ts"].append(ts); buf["fc_c"].append(fc_c)
            buf["fc_p"].append(fc_p); buf["lbl"].append(lbl)
            buf["sex"].append(int(sex_v[i])); buf["site"].append(site_v[i])
        except Exception:
            continue

    # Tangent FC (batch)
    conn_t = ConnectivityMeasure(kind="tangent", vectorize=False)
    try:
        fc_t_list = conn_t.fit_transform(buf["ts"])
        fc_t_list = [f.astype(np.float32) for f in fc_t_list]
        for f in fc_t_list: np.fill_diagonal(f, 0.)
    except Exception:
        fc_t_list = [f.copy() for f in buf["fc_c"]]

    ts_all      = np.array(buf["ts"],   dtype=np.float32)
    fc_corr_all = np.array(buf["fc_c"], dtype=np.float32)
    fc_tan_all  = np.array(fc_t_list,   dtype=np.float32)
    fc_par_all  = np.array(buf["fc_p"], dtype=np.float32)
    lbl_all     = np.array(buf["lbl"],  dtype=np.int64)
    sex_all     = np.array(buf["sex"],  dtype=np.int64)
    site_list   = buf["site"]

    # Site-wise variance normalisation
    for s in np.unique(site_list):
        mask = np.array(site_list) == s
        if mask.sum() >= 2:
            ts_all[mask] /= (ts_all[mask].std(axis=0, keepdims=True) + 1e-8)

    print("Building fMRI-text descriptors...")
    all_descs = [build_descriptor(ts_all[i], fc_corr_all[i])
                 for i in tqdm(range(len(ts_all)), desc="Descriptors")]

    # Integer site index (Sec. 3.4/3.5/3.7): sorted alphabetically so the
    # mapping is deterministic across runs/checkpoints.
    sites_sorted = sorted(set(site_list))
    site2idx     = {s: k for k, s in enumerate(sites_sorted)}
    site_idx     = np.array([site2idx[s] for s in site_list], dtype=np.int64)

    print(f"Done: {len(ts_all)} subjects  ASD={lbl_all.sum()}  "
          f"CTL={(lbl_all == 0).sum()}  sites={len(sites_sorted)}")
    return (ts_all, fc_corr_all, fc_tan_all, fc_par_all, lbl_all, sex_all,
            site_list, all_descs, site_idx, site2idx)


# ─────────────────────────────────────────────────────────────────────────────
# Site harmonization (ComBat-lite; Fortin et al. 2018 [9], Pomponio 2020 [23])
# ─────────────────────────────────────────────────────────────────────────────

def harmonize_fc_sitewise(fc_all: np.ndarray, site_list) -> np.ndarray:
    """
    Remove first-order acquisition-site batch effects from FC matrices by
    standardizing every connectivity edge *within each site* to zero mean
    and unit variance (a label-free ComBat-lite).

    Cross-site ASD failure is driven largely by scanner/site batch effects:
    each site's FC edges sit at a site-specific mean/scale, so a classifier
    trained on some sites sees a shifted distribution on unseen ones. This
    normalizes each site's edge distribution to a common location/scale,
    which is exactly the shift the site-adversarial GRL tries to learn away
    but can address directly on the raw FC that feeds the classifier.

    Label-free and applied independently per site (train sites and unseen
    test sites each use their own subjects' statistics), so there is no
    train->test label leakage. This is the standard harmonization setting
    used by ComBat (the paper cites both Fortin et al. and Pomponio et al.).
    Sites with <2 subjects are left unchanged (no variance to estimate).
    """
    out = fc_all.copy()
    site_arr = np.array(site_list)
    for s in np.unique(site_arr):
        m = site_arr == s
        if m.sum() >= 2:
            mu = out[m].mean(axis=0, keepdims=True)
            sd = out[m].std(axis=0, keepdims=True) + 1e-6
            out[m] = (out[m] - mu) / sd
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject text-descriptor embeddings (Sec. 3.2, Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

def embed_descriptors(llm, all_descs, lm_name: str, data_dir: str = "./abide_data",
                      max_subjects: int = None, use_cache: bool = True) -> np.ndarray:
    tag = str(max_subjects) if max_subjects is not None else "full"
    cache_path = Path(data_dir) / f"text_emb_{lm_name}_{tag}.npy"
    if use_cache and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(all_descs):
            print(f"Loading cached {lm_name} text embeddings ({tag}) from {cache_path} ...")
            return arr
        print(f"Cache {cache_path} has wrong subject count, recomputing.")

    if llm is None:
        from language_models import load_backend
        llm = load_backend(lm_name)

    print(f"Embedding {len(all_descs)} per-subject descriptors via {lm_name} "
         f"(cached for all future Stage-1/Stage-2 runs/seeds)...")
    embs = np.stack([llm.get_embedding(d).numpy()
                     for d in tqdm(all_descs, desc="Text embeddings")])
    if use_cache:
        np.save(cache_path, embs)
        print(f"Cached text embeddings -> {cache_path}")
    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Inter-site split  (Wei et al. 2026 protocol)
# ─────────────────────────────────────────────────────────────────────────────

def inter_site_split(lbl_all, site_list, train_ratio: float = TRAIN_RATIO, seed: int = SEED):
    """
    Adaptive inter-site split.
    Large sites (top `train_ratio` of subjects) -> Train + Val.
    Small sites -> Test  (unseen domain, domain-shift evaluation).
    """
    site_arr    = np.array(site_list)
    site_counts = pd.Series(site_list).value_counts().sort_values(ascending=False)
    cum_pct     = site_counts.cumsum() / len(site_list)

    big_sites   = site_counts[cum_pct <= train_ratio].index.tolist()
    small_sites = site_counts[cum_pct >  train_ratio].index.tolist()

    if len(small_sites) < 2:
        n = max(2, len(site_counts) // 4)
        small_sites = site_counts.index[-n:].tolist()
        big_sites   = site_counts.index[:-n].tolist()

    train_full = np.where(np.isin(site_arr, big_sites))[0]
    test_idx   = np.where(np.isin(site_arr, small_sites))[0]

    cls, cnt = np.unique(lbl_all[train_full], return_counts=True)
    strat = len(cls) >= 2 and cnt.min() >= 2 and int(len(train_full) * 0.15) >= 2
    if strat:
        train_idx, val_idx = train_test_split(
            train_full, test_size=0.15,
            stratify=lbl_all[train_full], random_state=seed)
    else:
        sz = max(2, int(len(train_full) * 0.15))
        train_idx, val_idx = train_full[sz:], train_full[:sz]

    print(f"Split - Train:{len(train_idx)}  Val:{len(val_idx)}  Test:{len(test_idx)}")
    print(f"  Train sites ({len(big_sites)}): {big_sites}")
    print(f"  Test  sites ({len(small_sites)}): {small_sites}")
    return train_idx, val_idx, test_idx, big_sites, small_sites


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset + DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

class ABIDEDataset(Dataset):
    def __init__(self, ts, fc_corr, fc_tan, fc_par, labels, sex, site_idx,
                 descs, text_emb=None, tokenizer=None, max_len: int = 256,
                 augment: bool = False):
        self.ts, self.fc_corr = ts.astype(np.float32), fc_corr.astype(np.float32)
        self.fc_tan, self.fc_par = fc_tan.astype(np.float32), fc_par.astype(np.float32)
        self.labels   = np.array(labels, dtype=np.int64)
        self.sex      = np.clip(np.array(sex) - 1, 0, 1).astype(np.int64)
        self.site_idx = np.array(site_idx, dtype=np.int64)
        self.descs    = descs
        self.augment  = augment
        # Per-subject descriptor embedding (Eq. 2) -- NOT a per-class lookup.
        # See dataset.embed_descriptors for why that distinction matters.
        self.text_emb = (text_emb.astype(np.float32) if text_emb is not None
                         else np.zeros((len(labels), 1), dtype=np.float32))
        if tokenizer is not None:
            self.tok_ids = np.array([
                tokenizer(d, max_length=max_len, truncation=True,
                          padding="max_length", return_tensors="np")["input_ids"][0]
                for d in descs], dtype=np.int64)
        else:
            self.tok_ids = np.zeros((len(labels), max_len), dtype=np.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        ts = torch.from_numpy(self.ts[i]).clone()
        fc_corr = torch.from_numpy(self.fc_corr[i]).clone()
        if self.augment:
            ts = ts + torch.randn_like(ts) * 0.02
            if torch.rand(1) > 0.5: ts = ts.flip(0)
            # Add small symmetric noise to FC matrix during training to prevent exact-value memorization
            noise = torch.randn_like(fc_corr) * 0.01
            noise = (noise + noise.T) / 2.0
            fc_corr = fc_corr + noise
            fc_corr.fill_diagonal_(0.0)
        return {"ts":       ts,
                "fc_corr":  fc_corr,
                "fc_tan":   torch.from_numpy(self.fc_tan[i]),
                "fc_par":   torch.from_numpy(self.fc_par[i]),
                "label":    torch.tensor(self.labels[i]),
                "sex":      torch.tensor(self.sex[i]),
                "site":     torch.tensor(self.site_idx[i]),
                "text_emb": torch.from_numpy(self.text_emb[i]),
                "tok_ids":  torch.from_numpy(self.tok_ids[i])}


def make_loaders(ts_all, fc_corr_all, fc_tan_all, fc_par_all,
                 lbl_all, sex_all, all_descs, site_idx,
                 train_idx, val_idx, test_idx, text_emb=None,
                 tokenizer=None, batch_size: int = 4, max_len: int = 256):
    def _mk(idx, aug, shuf):
        ds = ABIDEDataset(ts_all[idx], fc_corr_all[idx], fc_tan_all[idx], fc_par_all[idx],
                          lbl_all[idx], sex_all[idx], site_idx[idx],
                          [all_descs[j] for j in idx],
                          text_emb=(text_emb[idx] if text_emb is not None else None),
                          tokenizer=tokenizer, max_len=max_len, augment=aug)
        # RandomSampler raises on an empty dataset, which happens whenever a
        # caller only needs the test split (e.g. eval_zeroshot.py passes
        # empty train/val index arrays) -- shuffling is meaningless there anyway.
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuf and len(idx) > 0),
                          drop_last=False, pin_memory=torch.cuda.is_available())
    return _mk(train_idx, True, True), _mk(val_idx, False, False), _mk(test_idx, False, False)

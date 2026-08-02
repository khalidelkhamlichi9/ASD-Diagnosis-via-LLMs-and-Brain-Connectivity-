"""
NeuroLingua-Bridge — model_fmrilm_abide.py
Site-invariant, language-aligned fMRI model for cross-site ASD diagnosis on
ABIDE-I, built on top of the fMRI-LM tokenizer/alignment paradigm (Wei et al.,
arXiv:2511.21760v3, 2026) with pluggable LLM backends.

fMRI-LM's binary fMRI-vs-text domain loss is blind to acquisition site: it
does not know which of the 17 ABIDE-I scanners a subject came from, so its
tokens still carry site-discriminative structure that the classifier can
piggy-back on -- and that structure evaporates on unseen scanners (Sec. 1).
NeuroLingua-Bridge keeps fMRI-LM's tokenizer/backbone architecture unchanged
and instead adds three complementary domain-generalization (DG) modules on
top of Stage 1 and Stage 3 (Sec. 3.4-3.7 of the paper):

  (1) Multi-class site-adversarial GRL          -- Eq. (3), Sec. 3.4
  (2) Dynamic latent prototypes (site-balanced EMA) -- Eq. (4)-(5), Sec. 3.5
  (3) Episodic Group-DRO over sites             -- Eq. (6), Sec. 3.7
      (training loop lives in train_instruction.py; the classifier itself
      is architecture-agnostic to which optimizer drives it)

3-stage pipeline:
  Stage 1 — fMRI Tokenizer  (Encoder -> VQ -> Decoder)
             L_tok      = w1*L_quant + w2*L_CLIP + w3*L_site   (Eq. 1-3, Sigma wi = 1)
             L_Stage-1  = L_tok + gamma * L_proto               (Eq. 4, gamma=0.10)
  Stage 2 — LLM Alignment   (F2F + F2T)
  Stage 3 — ASD Classifier  (attention pool + multi-task head), trained with
             episodic Group-DRO across the 17 sites (Eq. 6)

Reference (backbone/tokenizer paradigm): Wei et al. arXiv:2511.21760v3 (2026)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_loss import siglip_loss

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (Appendix C, Table 6)
# ──────────────────────────────────────────────────────────────────────────────
D_ENC    = 256   # encoder hidden dim
D_VQ     = 128   # VQ codebook dim
K_VQ     = 8192  # codebook size
K_SITES  = 17    # ABIDE-I acquisition sites (site-adversarial GRL, Eq. 3)
N_CLASSES = 2    # ASD / Control (dynamic prototypes, Eq. 4-5)
GAMMA_PROTO = 0.10  # weight of L_proto in L_Stage-1 (paper: gamma = 0.10)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — fMRI Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

class GradReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin et al., 2016) for domain-adversarial training.

    Identity on the forward pass; on the backward pass it negates and scales
    the incoming gradient by `alpha`. Used to drive min_theta max_psi
    (L_task - lambda * L_site): the site discriminator D_psi learns to predict
    the acquisition site from the pooled latent z_bar, while the encoder
    E_theta is pushed (via the reversed gradient) to erase whatever made that
    prediction possible. At any saddle point of this minimax game the latent
    code satisfies I(z_bar; s) = 0 (Prop. S1, extending Ganin et al.'s
    Theorem 1 from K=2 to the K=17 multi-class case used here).
    """
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return grad.neg() * ctx.alpha, None

def grl_schedule(ep: int, total: int, gamma: float = 10.0) -> float:
    """
    Standard DANN sigmoid ramp-up for the GRL strength lambda: 0 -> 1 over
    training so the adversary doesn't destabilize the tokenizer before the
    reconstruction/contrastive losses have found a reasonable representation.
    """
    p = ep / max(total - 1, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class SiteDiscriminator(nn.Module):
    """
    D_psi in Eq. (3): a K=17-way site classifier over the pooled latent
    z_bar in R^{D_ENC}. Trained with ordinary cross-entropy on genuine
    (non-reversed) activations; only the encoder-facing gradient is reversed
    (via GradReversal, applied by the caller before this module sees the
    tensor). This is what distinguishes NeuroLingua-Bridge's adversary from
    fMRI-LM's domain loss: fMRI-LM's D is a binary fMRI-vs-text classifier
    with a constant target and carries no notion of acquisition site at all.
    """
    def __init__(self, d_in: int = D_ENC, k_sites: int = K_SITES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, k_sites))

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        return self.net(z_bar)   # (B, K_SITES) logits


class DynamicPrototypes(nn.Module):
    """
    Dynamic latent prototypes {mu_c} (Eq. 4-5, Sec. 3.5): one learnable
    geometric anchor per class (C=2: Control/ASD) in the site-invariant
    latent space. Each pooled token z_bar is pulled toward its own class
    prototype and pushed away from the nearest competing prototype
    (Eq. 4), which sharpens class clusters on top of the site-invariant
    subspace carved out by the GRL.

    The prototypes are NOT optimizer parameters (no gradient flows into
    `mu` from L_proto): they evolve by an EMA of the *site-balanced* class
    mean (Eq. 5) --

        mu_c <- eta * mu_c + (1 - eta) * (1/K) * sum_k mean_{i: c_i=c, s_i=k} z_bar_i

    i.e. the inner mean is taken *within* each site and the outer mean
    averages those K site-conditional means with equal weight, so that a
    single large site cannot pull a prototype onto its own manifold. This
    is what makes the anchors "site-balanced" rather than plain running
    class centroids.
    """
    def __init__(self, dim: int = D_ENC, n_classes: int = N_CLASSES,
                 k_sites: int = K_SITES, eta: float = 0.9, beta: float = 0.5):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_classes, dim))
        self.register_buffer("initialized", torch.zeros(1))
        self.n_classes, self.k_sites = n_classes, k_sites
        self.eta, self.beta = eta, beta

    @torch.no_grad()
    def update(self, z_bar: torch.Tensor, y: torch.Tensor, site: torch.Tensor) -> None:
        """EMA update of Eq. (5). Call once per training step with detached
        z_bar (prototypes must not receive gradient from the encoder)."""
        new = self.mu.clone()
        for c in range(self.n_classes):
            per_site_means = []
            for k in range(self.k_sites):
                mask = (y == c) & (site == k)
                if mask.any():
                    per_site_means.append(z_bar[mask].mean(0))
            if per_site_means:
                new[c] = torch.stack(per_site_means).mean(0)  # site-balanced class mean
        if self.initialized.item() == 0:
            self.mu.copy_(new)
            self.initialized.fill_(1.)
        else:
            self.mu.mul_(self.eta).add_(new, alpha=1 - self.eta)

    def loss(self, z_bar: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """
        Eq. (4): pull toward own prototype, push from the nearest other one,
        as a bounded hinge/margin loss:

            max(0, margin + own_dist^2 - beta * other_dist^2)

        The unbounded quadratic form `own^2 - beta*other^2` (no hinge) has
        no lower bound: the optimizer can trivially drive the loss toward
        -infinity just by inflating the encoder's overall feature-space
        scale (which increases `other_dist` faster than `own_dist`)
        without learning anything about actual class separation -- this
        diverged to -446 within 8 real-data epochs before the margin/hinge
        was added. The hinge clips the loss at 0 once the margin is
        satisfied, so there is no incentive to keep inflating the scale
        past what's needed to separate the two classes.
        """
        d = torch.cdist(z_bar, self.mu)                       # (B, n_classes)
        own = d.gather(1, y.view(-1, 1)).squeeze(1)
        other = d.clone().scatter_(1, y.view(-1, 1), float("inf"))
        other_min = other.min(1).values
        return F.relu(margin + own.pow(2) - self.beta * other_min.pow(2)).mean()


class LossWeights(nn.Module):
    """
    Learnable softmax-normalised weights for the 3-component tokenizer loss.
    Guarantees w1 + w2 + w3 = 1 at all times.
    """
    def __init__(self, w_q=0.5, w_s=0.3, w_g=0.2):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor([
            math.log(w_q / (1 - w_q + 1e-8)),
            math.log(w_s / (1 - w_s + 1e-8)),
            math.log(w_g / (1 - w_g + 1e-8)),
        ]))
    @property
    def weights(self): return F.softmax(self.raw, dim=0)
    def forward(self, lq, ls, lg):
        w = self.weights; return w[0] * lq + w[1] * ls + w[2] * lg
    def __repr__(self):
        w = self.weights.detach()
        return f"LossWeights(q={w[0]:.3f} s={w[1]:.3f} g={w[2]:.3f} Σ={w.sum():.4f})"


class NormEMAVQ(nn.Module):
    """
    Normalised EMA VQ with dead-token reset (NeuroLM style), plus k-means-
    style codebook seeding to avoid the classic VQ "cold-start collapse":

    A randomly-initialised codebook of K=8192 entries is, at step 0, nearly
    orthogonal to the true encoder-output manifold. Every input then snaps
    to whichever random entry happens to be *least* orthogonal to it, and
    because the EMA dead-code reset only ever resamples from the *current
    batch* (never from a healthier reference distribution), the codebook
    cannot recover on its own once a single entry starts winning every
    assignment -- it just keeps re-seeding "dead" entries from the same
    narrow region the dominant entry already owns, reinforcing the
    collapse rather than fixing it. `seed_from_batch` initialises the
    codebook directly from real encoder activations (called once, before
    the first optimizer step), so entries start out spread across the
    actual data distribution instead of random noise.
    """
    def __init__(self, K=K_VQ, D=D_VQ, beta=1.0, decay=0.99):
        super().__init__()
        self.K, self.D, self.beta, self.decay = K, D, beta, decay
        self.embed = nn.Embedding(K, D)
        nn.init.uniform_(self.embed.weight, -1/K, 1/K)
        self.register_buffer("ema_cnt", torch.ones(K))
        self.register_buffer("ema_w",   self.embed.weight.data.clone())
        self.register_buffer("seeded",  torch.zeros(1))

    @torch.no_grad()
    def seed_from_batch(self, z: torch.Tensor) -> None:
        """Seed the codebook from real (normalised) encoder activations.
        Call once on a representative batch before Stage-1 training starts."""
        zf = F.normalize(z, dim=-1).reshape(-1, self.D)
        if zf.shape[0] >= self.K:
            pick = torch.randperm(zf.shape[0], device=zf.device)[:self.K]
            base = zf[pick]
        else:
            reps = (self.K + zf.shape[0] - 1) // zf.shape[0]
            base = zf.repeat(reps, 1)[:self.K]
        noise = torch.randn_like(base) * 0.01
        init  = F.normalize(base + noise, dim=-1)
        self.embed.weight.data.copy_(init)
        self.ema_w.copy_(init)
        self.ema_cnt.fill_(1.0)
        self.seeded.fill_(1.0)

    def forward(self, z):
        zn = F.normalize(z, dim=-1)
        en = F.normalize(self.embed.weight, dim=-1)
        zf = zn.reshape(-1, self.D)
        d  = zf.pow(2).sum(1, True) + en.pow(2).sum(1) - 2 * (zf @ en.T)
        idx = d.argmin(1)
        zq  = self.embed(idx).view(z.shape)
        if self.training:
            # EMA codebook bookkeeping must never touch the autograd graph:
            # `zf` carries a grad_fn back to the encoder, and without
            # `no_grad()` here, `self.ema_w` (a persistent buffer reassigned
            # every step) inherits that grad_fn. Because it is read again on
            # the *next* call, each step chains a fresh graph onto the
            # previous one instead of being freed, retaining every prior
            # step's saved activations indefinitely -- an unbounded memory
            # leak that eventually OOMs on any non-trivial Stage-1 run.
            with torch.no_grad():
                zf_ng = zf.detach()
                oh = F.one_hot(idx, self.K).float()
                self.ema_cnt = self.decay * self.ema_cnt + (1 - self.decay) * oh.sum(0)
                self.ema_w   = self.decay * self.ema_w   + (1 - self.decay) * (oh.T @ zf_ng)
                dead = self.ema_cnt < 0.5
                n_d  = int(dead.sum())
                if n_d:
                    n_p  = min(n_d, zf_ng.shape[0])
                    perm = torch.randperm(zf_ng.shape[0], device=zf_ng.device)[:n_p]
                    di   = torch.where(dead)[0][:n_p]
                    # Add noise so a reset entry doesn't immediately collide
                    # with the exact vector the dominant entry already owns
                    # (a bare copy just re-creates the collapse next step).
                    noise = torch.randn(n_p, self.D, device=zf_ng.device) * 0.05
                    self.ema_w.data[di]   = zf_ng[perm] + noise
                    self.ema_cnt.data[di] = 1.
                self.embed.weight.data = F.normalize(
                    self.ema_w / (self.ema_cnt.unsqueeze(1) + 1e-5), dim=-1)
        return z + (zq - z).detach(), self.beta * F.mse_loss(zq.detach(), z), idx

    @property
    def utilisation(self): return (self.ema_cnt > 0.5).float().mean().item()


class fMRIEncoder(nn.Module):
    """
    Encodes each subject from (a) raw patch-pooled BOLD dynamics and (b)
    static functional connectivity (FC).

    The FC branch matters: raw ROI-averaged timeseries alone is a very weak
    ASD signal (this is exactly why the literature -- BrainNetCNN, BrainGNN,
    BNT -- builds on the *connectivity* structure rather than raw signal).
    Without it, diagnosing empirically showed the encoder learns to output
    a near-constant representation regardless of subject (cosine similarity
    ~0.9998 between different subjects' pooled encodings): the per-timestep
    mean-pooling in the timeseries branch destroys almost all subject-
    specific structure before the transformer even sees it, so reconstruction
    + the other Stage-1 losses had nothing subject-specific left to preserve.
    Injecting the static FC matrix (which the dataset already computes but
    the model previously ignored entirely) gives the encoder a second,
    much richer per-subject signal to condition on.
    """
    def __init__(self, n_rois=200, T=160, d=D_ENC, nhead=8, nlayers=3, P=32):
        # n_rois=200: CC200 atlas is natively ~200 ROIs (Sec. 3.1); dataset.py
        # pads/truncates every subject to exactly N_ROIS=200, so this must match.
        super().__init__()
        self.P, self.n_rois = P, n_rois
        self.roi_proj = nn.Linear(n_rois, d)
        self.t_proj   = nn.Linear(P, d)
        n_pairs = n_rois * (n_rois - 1) // 2
        iu = torch.triu_indices(n_rois, n_rois, offset=1)
        self.register_buffer("_iu_row", iu[0], persistent=False)
        self.register_buffer("_iu_col", iu[1], persistent=False)
        self.fc_proj  = nn.Sequential(nn.Linear(n_pairs, d), nn.LayerNorm(d), nn.GELU())
        self.combine  = nn.Linear(d * 3, d)
        enc = nn.TransformerEncoderLayer(d, nhead, d*4, 0.1,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, nlayers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, fc=None):          # x: (B, T, N), fc: (B, N, N) or None
        B, T, N = x.shape
        T2  = T // self.P
        xp  = x[:, :T2*self.P, :].reshape(B, T2, self.P, N)
        if fc is None:
            fc = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        fc_vec = fc[:, self._iu_row, self._iu_col]              # (B, n_pairs)
        fc_emb = self.fc_proj(fc_vec).unsqueeze(1).expand(-1, T2, -1)  # (B, T2, d)
        z   = self.combine(torch.cat([self.roi_proj(xp.mean(2)),
                                      self.t_proj(xp.mean(-1)),
                                      fc_emb], -1))
        return self.norm(self.transformer(z))  # (B, T2, D_ENC)


class fMRIDecoder(nn.Module):
    def __init__(self, d_in=D_ENC, P=32, N=200):
        super().__init__()
        self.P, self.N = P, N
        dec = nn.TransformerEncoderLayer(d_in, 4, d_in*2, 0.1,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(dec, 2)
        self.out = nn.Linear(d_in, P * N)

    def forward(self, z):          # z: (B, T2, d_in)
        B, T2, _ = z.shape
        return self.out(self.transformer(z)).reshape(B, T2*self.P, self.N)


class fMRITokenizer(nn.Module):
    """
    Stage-1 model: Encoder -> Task Layer -> VQ Codebook -> Decoder, with the
    two DG modules attached to the pooled latent z_bar = mean_t(z):

      L_tok      = w1*L_quant + w2*L_CLIP + w3*L_site        (Eq. 1-3)
      L_Stage-1  = L_tok + gamma * L_proto                    (Sec. 3.5, gamma=0.10)

    L_site replaces fMRI-LM's binary fMRI/text domain loss with a genuine
    17-way site classifier behind a GRL (Sec. 3.4); L_proto pulls z_bar
    toward its class prototype and away from the other one (Sec. 3.5). The
    prototype EMA (Eq. 5) is updated once per step from the *detached*
    z_bar so no gradient from L_proto leaks into the encoder through mu.
    """
    def __init__(self, d_llm: int = 1024, n_rois: int = 200,
                 k_sites: int = K_SITES, gamma_proto: float = GAMMA_PROTO,
                 lambda_var: float = 1.0, lambda_cov: float = 1e-3):
        super().__init__()
        self.gamma_proto = gamma_proto
        self.lambda_var  = lambda_var   # VICReg variance weight (anti-collapse)
        self.lambda_cov  = lambda_cov   # VICReg covariance weight (decorrelation)
        self.encoder    = fMRIEncoder(n_rois=n_rois)
        self.task_layer = nn.Sequential(nn.Linear(D_ENC, D_ENC), nn.Tanh(),
                                        nn.Linear(D_ENC, D_VQ))
        self.quantizer  = NormEMAVQ()
        self.vq_to_enc  = nn.Linear(D_VQ, D_ENC)
        self.decoder    = fMRIDecoder(d_in=D_ENC, N=n_rois)
        self.proj_out   = nn.Sequential(nn.Linear(D_VQ, d_llm), nn.GELU())
        self.site_disc  = SiteDiscriminator(d_in=D_ENC, k_sites=k_sites)
        self.protos     = DynamicPrototypes(dim=D_ENC, k_sites=k_sites)
        self.fmri_head  = nn.Sequential(nn.Linear(D_ENC, 256), nn.LayerNorm(256))
        self.text_head  = nn.Sequential(nn.Linear(d_llm, 256),  nn.LayerNorm(256))
        self.log_temp   = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode(self, x, fc=None) -> torch.Tensor:
        z = self.encoder(x, fc=fc)
        zq, _, _ = self.quantizer(self.task_layer(z))
        return self.proj_out(zq)            # (B, T2, d_llm)

    @torch.no_grad()
    def seed_quantizer(self, x: torch.Tensor, fc=None) -> None:
        """Seed the VQ codebook from this batch's real encoder activations.
        Call once, before the first optimizer step of Stage-1 training, to
        avoid the codebook cold-start collapse (see NormEMAVQ docstring)."""
        z_vq = self.task_layer(self.encoder(x, fc=fc))
        self.quantizer.seed_from_batch(z_vq)

    def forward(self, x, fc=None, text_emb=None, label=None, site=None,
                lam: float = 0.5, loss_weights=None):
        """
        Args:
            x:        (B, T, N_ROIS) preprocessed fMRI time series.
            fc:       (B, N_ROIS, N_ROIS) static functional-connectivity
                      matrix (fc_corr) -- see fMRIEncoder docstring for why
                      this is needed alongside the raw timeseries.
            text_emb: (B, d_llm) paired descriptor embedding for L_CLIP (Sec. 3.2).
            label:    (B,) long, ASD/Control class in {0,1} -- required for L_proto.
            site:     (B,) long, site index in [0, K_SITES) -- required for
                      L_site and the prototype EMA (Eq. 5).
            lam:      current GRL strength (see `grl_schedule`).
            loss_weights: optional `LossWeights` module implementing the
                      learnable softmax combination of Eq. (1-3).
        """
        B = x.shape[0]
        z       = self.encoder(x, fc=fc)    # (B, T2, D_ENC)
        z_vq    = self.task_layer(z)
        zq, lc, _ = self.quantizer(z_vq)

        recon   = self.decoder(self.vq_to_enc(zq.detach()))
        l_quant = F.mse_loss(recon, x) + lc                  # Eq. (1)

        z_pool  = z.mean(1)                 # (B, D_ENC) -- this is z_bar in the paper

        # ── Eq. (3): 17-way site-adversarial GRL ────────────────────────────
        if site is not None:
            site_logits = self.site_disc(GradReversal.apply(z_pool, lam))
            l_site = F.cross_entropy(site_logits, site)
        else:
            l_site = torch.tensor(0., device=x.device)

        # ── Eq. (2): fMRI-text CLIP-style contrastive loss (Sec. 3.2) ───────
        l_contr = torch.tensor(0., device=x.device)
        if text_emb is not None:
            l_contr = siglip_loss(self.fmri_head(z_pool),
                                  self.text_head(text_emb), self.log_temp)

        # ── Eq. (4)-(5): dynamic site-balanced prototypes (Sec. 3.5) ────────
        l_proto = torch.tensor(0., device=x.device)
        if label is not None and site is not None:
            self.protos.update(z_pool.detach(), label, site)   # Eq. (5), EMA
            l_proto = self.protos.loss(z_pool, label)           # Eq. (4)

        # ── Anti-collapse regularization (VICReg; Bardes et al. 2022) ───────
        # Measured failure mode: three losses (site-adversarial GRL, the
        # prototype pull, and the VQ commitment) all *compress* z_pool, while
        # the only expansion pressure -- the CLIP loss to text -- has targets
        # that are ~0.96 cosine-similar to each other (weak). With nothing to
        # counter compression, the encoder collapses to a near-constant output
        # (pairwise cosine ~0.9999 between different subjects), and every
        # downstream stage becomes blind to the input, so the classifier can
        # only predict the majority class. The VICReg variance term forces
        # each latent dimension to keep batch std >= 1 (a hinge), which makes
        # a constant representation a high-loss solution; the covariance term
        # decorrelates dimensions so the variance isn't met by duplicating one
        # informative axis. This is the standard, literature-backed remedy for
        # representation collapse in non-contrastive SSL.
        # Both VICReg terms are batch statistics: at very small batch sizes
        # (e.g. B=2) they are too noisy to be meaningful -- the covariance is
        # near rank-deficient and its penalty explodes -- so gate them on a
        # minimum batch. Stage-1 training should therefore use a reasonably
        # large batch (the LLM is freed before training, so the small
        # tokenizer can afford it); see train_tokenizer.py.
        if B >= 8:
            l_var = self._variance_loss(z_pool)
            l_cov = self._covariance_loss(z_pool)
        else:
            l_var = torch.zeros((), device=x.device)
            l_cov = torch.zeros((), device=x.device)

        l_tok = (loss_weights(l_quant, l_contr, l_site)
                 if loss_weights is not None
                 else l_quant + l_contr + lam * l_site)
        total = (l_tok + self.gamma_proto * l_proto
                 + self.lambda_var * l_var + self.lambda_cov * l_cov)

        return {"total": total, "quant": l_quant, "contrast": l_contr,
                "site": l_site, "proto": l_proto,
                "var": l_var, "cov": l_cov, "z_pool": z_pool}

    @staticmethod
    def _variance_loss(z: torch.Tensor, target_std: float = 1.0,
                       eps: float = 1e-4) -> torch.Tensor:
        """VICReg variance term: hinge that keeps each dim's batch std >= 1."""
        std = torch.sqrt(z.var(dim=0) + eps)
        return F.relu(target_std - std).mean()

    @staticmethod
    def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
        """VICReg covariance term: push off-diagonal covariances toward 0."""
        B, D = z.shape
        if B < 2:
            return torch.zeros((), device=z.device)
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1)                # (D, D)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum() / D


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LLM Alignment  (F2F + F2T)
# ══════════════════════════════════════════════════════════════════════════════

ALPHA_F2F = 0.1

class AlignmentModel(nn.Module):
    """Frozen tokenizer + alignment heads (F2F autoregressive + F2T cosine)."""
    def __init__(self, tokenizer: fMRITokenizer, d_llm: int = 1024):
        super().__init__()
        self.tok = tokenizer
        for p in self.tok.parameters(): p.requires_grad = False
        self.projector  = nn.Sequential(nn.Linear(d_llm, d_llm), nn.GELU())
        self.f2f_head   = nn.Linear(d_llm, D_VQ)
        self.align_head = nn.Sequential(nn.Linear(d_llm, d_llm), nn.LayerNorm(d_llm))
        self.d_llm = d_llm

    def get_features(self, x, fc=None): return self.projector(self.tok.encode(x, fc=fc))

    def forward(self, x, fc=None, llm_emb=None):
        h = self.get_features(x, fc=fc)
        with torch.no_grad():
            z_vq = self.tok.task_layer(self.tok.encoder(x, fc=fc))
        l_f2f = F.mse_loss(self.f2f_head(h[:, :-1, :]), z_vq[:, 1:, :])
        l_f2t = torch.tensor(0., device=x.device)
        if llm_emb is not None:
            pool  = self.align_head(h.mean(1))
            l_f2t = 1.0 - F.cosine_similarity(
                F.normalize(pool, dim=-1), F.normalize(llm_emb, dim=-1)).mean()
        return {"total": l_f2t + ALPHA_F2F * l_f2f, "f2f": l_f2f, "f2t": l_f2t, "h": h}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ASD Classifier
# ══════════════════════════════════════════════════════════════════════════════

class ASDClassifier(nn.Module):
    """
    Multi-task ASD classifier on top of the Stage-2 alignment model, trained
    with episodic Group-DRO across the 17 ABIDE-I sites (Eq. 6, Sec. 3.7 --
    the training loop that drives the per-site reweighting lives in
    `GroupDRO` below and is called from train_instruction.py).

    Two complementary feature streams are fused (this is the "multimodal"
    in the paper's title):

      (1) LLM-token stream -- the language-aligned VQ tokens from Stage 2,
          attention-pooled. This carries the site-invariant, text-grounded
          representation.
      (2) Direct functional-connectivity stream -- the subject's FC matrix
          (upper triangle) passed through a heavily-regularized bottleneck.
          Empirically the FC connectivity carries most of the discriminative
          ASD signal (a plain FC->linear baseline already separates the
          classes), whereas the self-supervised + site-adversarial tokenizer
          can inadvertently discard ASD-relevant structure along with the
          site structure it is trained to remove. Fusing the raw FC back in
          gives the classifier direct access to that signal.

    The FC stream is the main overfitting risk (N_pairs ~ 19,900 features
    vs. ~549 training subjects), so it is regularized aggressively: an input
    dropout, a small linear bottleneck (no giant hidden layer), heavy
    dropout, and weight decay from the optimizer. This is the intended
    counter-overfitting design -- widen it and it will memorize the train
    split and collapse cross-site generalization.

    Primary task: ASD (1=ASD, 0=Control).
    Auxiliary task: sex prediction (weight=0.3) -- a light regularizer on the
    shared trunk.
    """
    def __init__(self, align_model: AlignmentModel, d_llm: int = 1024,
                 hidden: int = 512, n_rois: int = 200, fc_dim: int = 128,
                 finetune_encoder: bool = False):
        super().__init__()
        self.align = align_model
        for p in self.align.parameters(): p.requires_grad = False
        for p in self.align.projector.parameters():  p.requires_grad = True
        for p in self.align.align_head.parameters(): p.requires_grad = True
        # Optional end-to-end fine-tuning: let the frozen tokenizer encoder
        # adapt its representation to the actual ASD task (the self-supervised
        # Stage-1 features are not trained for discrimination, so a frozen
        # head on top of them can be starved of signal).
        if finetune_encoder:
            for p in self.align.tok.encoder.parameters():    p.requires_grad = True
            for p in self.align.tok.task_layer.parameters(): p.requires_grad = True

        # ── LLM-token stream ────────────────────────────────────────────────
        self.attn_pool = nn.Sequential(nn.Linear(d_llm, 1), nn.Softmax(dim=1))
        self.tok_feat = nn.Sequential(
            nn.Linear(d_llm, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.35))

        # ── Direct FC-connectivity stream (regularized bottleneck) ──────────
        n_pairs = n_rois * (n_rois - 1) // 2
        iu = torch.triu_indices(n_rois, n_rois, offset=1)
        self.register_buffer("_iu_row", iu[0], persistent=False)
        self.register_buffer("_iu_col", iu[1], persistent=False)
        self.fc_in_drop = nn.Dropout(0.5)                     # connection dropout
        self.fc_feat = nn.Sequential(
            nn.Linear(n_pairs, fc_dim), nn.LayerNorm(fc_dim), nn.GELU(), nn.Dropout(0.5))

        # ── Fusion trunk ────────────────────────────────────────────────────
        fused_dim = hidden + fc_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.GELU(), nn.Dropout(0.4))
        self.asd_head = nn.Linear(hidden // 2, 2)
        self.sex_head = nn.Linear(hidden // 2, 2)

    def forward(self, x, fc=None):
        h   = self.align.get_features(x, fc=fc)    # (B, T2, d_llm)
        atw = self.attn_pool(h)                    # (B, T2, 1)
        hp  = (h * atw).sum(1)                      # (B, d_llm)
        tok_f = self.tok_feat(hp)                   # (B, hidden)

        fc_vec = fc[:, self._iu_row, self._iu_col]  # (B, n_pairs)
        fc_f   = self.fc_feat(self.fc_in_drop(fc_vec))

        fused = self.trunk(torch.cat([tok_f, fc_f], dim=-1))
        return self.asd_head(fused), self.sex_head(fused)

    def loss(self, asd_lg, sex_lg, lbl_asd, lbl_sex):
        """Scalar (batch-mean) multi-task loss, kept for plain-ERM training/eval."""
        return (F.cross_entropy(asd_lg, lbl_asd, label_smoothing=0.08)
                + 0.3 * F.cross_entropy(sex_lg, lbl_sex, label_smoothing=0.05))

    def per_sample_loss(self, asd_lg, sex_lg, lbl_asd, lbl_sex):
        """
        Unreduced (B,) multi-task loss. Group-DRO (Eq. 6) needs the per-site
        *mean* loss L_k(theta) before the site-adversarial reweighting can be
        applied, so training must start from a per-sample loss rather than
        the already-averaged scalar returned by `loss`.
        """
        return (F.cross_entropy(asd_lg, lbl_asd, label_smoothing=0.08, reduction="none")
                + 0.3 * F.cross_entropy(sex_lg, lbl_sex, label_smoothing=0.05, reduction="none"))


class GroupDRO:
    """
    Episodic Group-DRO (Sagawa et al., 2020) with sites as groups (Eq. 6,
    Sec. 3.7): min_theta max_{q in Delta^{K-1}} sum_k q_k * L_k(theta).

    Each training step is one "episode": the K per-site mean losses are
    computed on the current minibatch, the adversarial site distribution q
    is updated multiplicatively toward whichever sites are currently
    hardest (exponentiated-gradient ascent, `q_k <- q_k * exp(nu * L_k)`,
    renormalized), and the model is updated on the q-weighted loss. Sites
    absent from a given minibatch keep their previous q_k (no update),
    which lets K_SITES be checked against a running best worst-site metric
    on a held-out validation split rather than the training batch alone.
    """
    def __init__(self, k_sites: int = K_SITES, step_size: float = 0.01):
        self.q = torch.ones(k_sites) / k_sites
        self.step_size = step_size
        self.k_sites = k_sites

    def step(self, per_sample_loss: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_sample_loss: (B,) unreduced loss per subject in the batch.
            site:            (B,) long site index in [0, K_SITES).
        Returns:
            scalar loss, differentiable w.r.t. theta, equal to the
            q-weighted sum of per-site mean losses *renormalized over the
            sites present in this batch* (so the loss scale stays ~O(1)
            regardless of how many of the K_SITES a small minibatch
            happens to contain). The persistent q over all K_SITES is
            updated multiplicatively (exponentiated-gradient ascent) and
            kept for worst-site bookkeeping across batches/epochs.
        """
        device = per_sample_loss.device
        present = torch.unique(site).tolist()
        per_site_loss = torch.zeros(self.k_sites, device=device)
        for k in present:
            per_site_loss[k] = per_sample_loss[site == k].mean()

        with torch.no_grad():
            q = self.q.to(device).clone()
            for k in present:
                q[k] = q[k] * math.exp(self.step_size * per_site_loss[k].item())
            q = q / q.sum()
            self.q = q.cpu()

        q_batch = torch.stack([q[k] for k in present])
        q_batch = (q_batch / q_batch.sum()).detach()
        l_batch = torch.stack([per_site_loss[k] for k in present])
        return (q_batch * l_batch).sum()

    @property
    def worst_site(self) -> int:
        return int(self.q.argmax().item())

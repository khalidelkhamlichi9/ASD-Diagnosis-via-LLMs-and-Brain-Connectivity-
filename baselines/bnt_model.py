"""
baselines/bnt_model.py
───────────────────────
Implémentation fidèle du Brain Network Transformer (BNT).

Référence : Kan et al., "Brain Network Transformer", NeurIPS 2022.
             https://arxiv.org/abs/2210.06681

Architecture :
  1. Orthonormal Clustering (OC) : partitionne les ROIs en K clusters via
     une projection linéaire orthogonale apprise. Remplace le positional
     encoding par une structure de communauté cérébrale.
  2. BNT block × n_layers : transformer multi-têtes sur les représentations
     de clusters, avec skip-connection.
  3. Readout global : pooling → MLP → logits (ASD / Control)

Entrée : matrice FC (B, N, N), N=200 ROIs (CC200 atlas)
Sortie : (B, n_classes) logits

Différence principale avec un ViT standard :
- L'OC produit des «cluster tokens» dépendant des données (non positionnels)
- Les têtes d'attention opèrent sur les K clusters (pas les N ROIs)
- La matrice FC complète (N×N) est l'entrée — pas de patches temporels
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Orthonormal Clustering (OC) — Sec. 3.1 de Kan et al.
# ─────────────────────────────────────────────────────────────────────────────

class OrthonormalClustering(nn.Module):
    """
    Projection ROI → Cluster avec contrainte orthonormale sur W.

    Kan et al. Eq. (2) : H = softmax(FC @ W^T) @ W
      - W ∈ R^{K×N} est la matrice de clustering (K clusters, N ROIs)
      - softmax(FC @ W^T) : assignation douce des ROIs aux clusters (B, N, K)
      - @ W : propage les features dans l'espace des clusters (B, N, K) @ (K, N)

    La contrainte orthonormale (W W^T ≈ I_K) est imposée via une
    régularisation Stiefel (terme de loss L_OC) : elle force chaque cluster
    à capturer une dimension indépendante de la connectivité cérébrale,
    évitant l'effondrement vers des clusters redondants.

    Note : en pratique on apprend W sans contrainte dure et on régularise
    par L_OC = ||W W^T - I_K||_F² pondéré par lambda_oc.
    """

    def __init__(self, n_roi: int = 200, n_clusters: int = 8, d_model: int = 64):
        super().__init__()
        self.n_roi      = n_roi
        self.n_clusters = n_clusters
        self.d_model    = d_model

        # Matrice de clustering W ∈ R^{K×N}
        self.W = nn.Parameter(torch.empty(n_clusters, n_roi))
        nn.init.orthogonal_(self.W)

        # Projection des cluster tokens → espace latent d_model
        self.cluster_proj = nn.Linear(n_roi, d_model)

    def orthonormal_loss(self) -> torch.Tensor:
        """
        Régularisation Stiefel : L_OC = ||W W^T - I_K||_F²
        Minimisée conjointement avec la loss principale, poids λ_oc=0.01.
        """
        K = self.W.shape[0]
        WWT = self.W @ self.W.t()                    # (K, K)
        return (WWT - torch.eye(K, device=self.W.device)).pow(2).mean()

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fc : (B, N, N) matrice FC symétrique.

        Returns:
            H  : (B, K, d_model) — représentations des K clusters.
        """
        # Score d'assignation ROI→cluster : (B, N, K)
        assign = torch.softmax(fc @ self.W.t(), dim=-1)  # (B, N, K)

        # Agrégation dans l'espace des clusters : (B, K, N)
        H_raw = assign.transpose(1, 2) @ fc              # (B, K, N)

        # Projection → d_model
        H = self.cluster_proj(H_raw)                     # (B, K, d_model)
        return H


# ─────────────────────────────────────────────────────────────────────────────
# BNT Block — Transformer multi-têtes sur les cluster tokens
# ─────────────────────────────────────────────────────────────────────────────

class BNTBlock(nn.Module):
    """
    Un bloc Transformer standard appliqué aux K cluster tokens.

    Architecture : Pre-norm (stabilité) + skip-connections.
      x → LayerNorm → MultiheadAttention → + x
        → LayerNorm → FFN(d_model → 4*d_model → d_model) → + x
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, K, d_model)"""
        # Self-attention avec skip
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # FFN avec skip
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# BrainNetworkTransformer — modèle complet
# ─────────────────────────────────────────────────────────────────────────────

class BrainNetworkTransformer(nn.Module):
    """
    Brain Network Transformer (Kan et al., NeurIPS 2022).

    Pipeline complet :
      FC (B, N, N)
      → OrthonormalClustering → H (B, K, d_model)
      → N×BNTBlock (transformer sur les K cluster tokens)
      → LayerNorm finale
      → Mean pooling → (B, d_model)
      → MLP readout → (B, n_classes)

    Paramètres :
      n_roi       : nombre de ROIs (200 pour CC200)
      n_clusters  : K (nombre de clusters cérébraux, 8 par défaut comme dans le papier)
      d_model     : dimension cachée (64 par défaut)
      n_heads     : têtes d'attention (4 par défaut)
      n_layers    : nombre de BNT blocks (2 par défaut)
      dropout     : dropout intermédiaire
      n_classes   : 2 (ASD / Control)
      lambda_oc   : poids de la régularisation Stiefel
    """

    def __init__(
        self,
        n_roi:      int   = 200,
        n_clusters: int   = 8,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        dropout:    float = 0.3,
        n_classes:  int   = 2,
        lambda_oc:  float = 0.01,
    ):
        super().__init__()
        self.lambda_oc = lambda_oc

        # ── 1. Orthonormal Clustering ────────────────────────────────────
        self.oc = OrthonormalClustering(n_roi, n_clusters, d_model)

        # ── 2. Pile de BNT blocks ────────────────────────────────────────
        self.blocks = nn.ModuleList([
            BNTBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # ── 3. Readout MLP ───────────────────────────────────────────────
        # Mean-pooling sur les K tokens → MLP → logits
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_classes),
        )

    def forward(self, fc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fc : (B, N, N) matrices FC symétriques (Fisher-Z correlation).

        Returns:
            logits   : (B, n_classes)
            l_oc     : scalaire — perte de régularisation orthonormale.
        """
        # ── OC → cluster tokens ──────────────────────────────────────────
        H = self.oc(fc)                                  # (B, K, d_model)

        # ── Transformer blocks ───────────────────────────────────────────
        for block in self.blocks:
            H = block(H)
        H = self.norm_out(H)                             # (B, K, d_model)

        # ── Readout : mean-pool sur K clusters ───────────────────────────
        pooled = H.mean(dim=1)                           # (B, d_model)
        logits = self.readout(pooled)                    # (B, n_classes)

        # ── Régularisation Stiefel ────────────────────────────────────────
        l_oc = self.oc.orthonormal_loss()

        return logits, l_oc

    def get_cluster_assignments(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Retourne les assignations soft ROI→cluster (utile pour visualisation).

        Returns:
            assign : (B, N, K) assignations de softmax
        """
        with torch.no_grad():
            assign = torch.softmax(fc @ self.oc.W.t(), dim=-1)
        return assign

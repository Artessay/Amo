# Copyright 2025 Rihong Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-objective *advantage* baselines for Amo.

These estimators consume the per-objective token-level score tensors
(``token_level_scores_<objective>``) produced by the Amo multi-objective reward
managers, rather than a single pre-scalarized reward. They are wired into
``verl.trainer.ppo.ray_trainer.compute_advantage`` alongside GDPO, because -- like
GDPO -- they need the *dict* of per-objective tensors, not just
``token_level_rewards``.

Implemented estimators (all keep the same rollout/KL/token budget; only the
multi-objective advantage construction differs):

* ``gdpo_weighted`` -- GDPO with per-objective weights. Each objective's group
  z-scored advantage is scaled by ``w_j`` before summation, then batch-whitened.
  ``w = uniform`` recovers plain GDPO.

* ``rvpo`` -- Reward-Variance Policy Optimization. Aggregates the per-objective
  group z-scores with a soft-min instead of the mean::

      A = -(1/k) log( (1/m) sum_j exp(-k Z_j) ).

  ``k -> 0`` recovers the GDPO mean; ``k -> inf`` approaches the worst objective.
  Followed by batch whitening, matching the GDPO pipeline.

* ``mgda`` / ``gapo`` -- gradient-conflict aggregation, applied in
  *advantage space* as an efficient, clearly-labelled adaptation of the
  last-layer MGDA / GAPO trick (the paper versions solve the min-norm problem on
  per-objective policy gradients). We treat each objective's per-sample
  group-normalized advantage vector ``a_j in R^N`` as a stand-in for its gradient
  direction, solve for the min-norm convex combination ``alpha`` on the simplex,
  and return ``sum_j alpha_j a_j``. GAPO first rescales each ``a_j`` by
  ``1/||a_j||^p`` before solving. This isolates the *aggregation rule*; it is NOT
  a faithful full-parameter-gradient MGDA and must be reported as an adaptation.

Every function returns ``(advantages, returns)`` broadcast over the response
tokens, matching the signature expected by the trainer.
"""

from __future__ import annotations

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _per_objective_group_advantages(
    token_level_scores_list: list[torch.Tensor],
    response_mask: torch.Tensor,
    index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
) -> list[torch.Tensor]:
    """GRPO group-normalize each objective's token-level scores independently.

    Returns a list of ``(bs, response_length)`` tensors, one per objective. This
    is the shared front-end for every GDPO-family estimator.
    """
    out = []
    for token_level_scores in token_level_scores_list:
        adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_scores,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        out.append(adv)
    return out


def _normalize_weights(weights, num_objectives: int, device, dtype) -> torch.Tensor:
    if weights is None:
        return torch.full((num_objectives,), 1.0 / num_objectives, device=device, dtype=dtype)
    w = torch.tensor([float(x) for x in weights], device=device, dtype=dtype)
    if w.numel() != num_objectives:
        raise ValueError(f"[Amo][adv] weights length {w.numel()} != num_objectives {num_objectives}")
    if torch.any(w < 0):
        raise ValueError(f"[Amo][adv] weights must be non-negative, got {weights}")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"[Amo][adv] weights must sum to positive, got {weights}")
    return w / total


def compute_gdpo_weighted_advantage(
    token_level_scores_list: list[torch.Tensor],
    response_mask: torch.Tensor,
    index: np.ndarray,
    weights=None,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted GDPO: weighted sum of per-objective group z-scores, then whiten."""
    per_obj = _per_objective_group_advantages(
        token_level_scores_list, response_mask, index, norm_adv_by_std_in_grpo
    )
    m = len(per_obj)
    w = _normalize_weights(weights, m, per_obj[0].device, per_obj[0].dtype)
    stacked = torch.stack(per_obj, dim=0)  # (m, bs, T)
    combined = (stacked * w.view(m, 1, 1)).sum(dim=0)  # (bs, T)
    advantages = verl_F.masked_whiten(combined, response_mask) * response_mask
    return advantages, advantages


def compute_rvpo_advantage(
    token_level_scores_list: list[torch.Tensor],
    response_mask: torch.Tensor,
    index: np.ndarray,
    k: float = 1.0,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RVPO: soft-min over per-objective group z-scores, then whiten.

    We reduce each objective's token-level advantage to a per-sample scalar (the
    last-token value carries the outcome credit; summing over the response tokens
    recovers it), soft-min-combine across objectives, then broadcast back over
    the tokens and batch-whiten -- matching the GDPO pipeline.
    """
    per_obj = _per_objective_group_advantages(
        token_level_scores_list, response_mask, index, norm_adv_by_std_in_grpo
    )
    # Per-sample scalar Z_j for each objective.
    z = torch.stack([adv.sum(dim=-1) for adv in per_obj], dim=0)  # (m, bs)
    m = z.shape[0]
    k = float(k)
    if k <= 0:
        agg = z.mean(dim=0)  # GDPO mean limit
    else:
        # -1/k log( mean_j exp(-k Z_j) ), computed stably.
        neg = -k * z  # (m, bs)
        lse = torch.logsumexp(neg, dim=0) - np.log(m)  # (bs,)
        agg = -lse / k
    combined = agg.unsqueeze(-1) * response_mask  # (bs, T)
    advantages = verl_F.masked_whiten(combined, response_mask) * response_mask
    return advantages, advantages


def _min_norm_two_vectors(u: torch.Tensor, v: torch.Tensor) -> float:
    """Closed-form min ||gamma u + (1-gamma) v||^2 over gamma in [0,1]."""
    uu = torch.dot(u, u)
    vv = torch.dot(v, v)
    uv = torch.dot(u, v)
    denom = uu - 2 * uv + vv
    if denom <= 1e-12:
        return 0.5
    gamma = ((vv - uv) / denom).item()
    return float(min(1.0, max(0.0, gamma)))


def _frank_wolfe_min_norm(vectors: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """Min-norm point in the convex hull of rows of ``vectors`` (MGDA).

    Solves ``min_{alpha in simplex} || sum_j alpha_j g_j ||^2`` via the
    Frank-Wolfe / Wolfe algorithm used in Sener & Koltun (2018). Returns the
    ``alpha`` weights (shape ``(m,)``). Robust for the small ``m`` here.
    """
    m = vectors.shape[0]
    if m == 1:
        return vectors.new_ones(1)
    if m == 2:
        gamma = _min_norm_two_vectors(vectors[0], vectors[1])
        return vectors.new_tensor([gamma, 1.0 - gamma])
    alpha = vectors.new_full((m,), 1.0 / m)
    gram = vectors @ vectors.t()  # (m, m)
    for _ in range(iters):
        # gradient of 1/2 alpha^T G alpha is G alpha; pick min-index vertex.
        grad = gram @ alpha
        t = int(torch.argmin(grad).item())
        # line search between alpha and vertex e_t.
        e_t = vectors.new_zeros(m)
        e_t[t] = 1.0
        d = e_t - alpha
        # minimize 1/2 (alpha+step d)^T G (alpha+step d) over step in [0,1].
        dGd = float((d @ (gram @ d)).item())
        gGd = float((alpha @ (gram @ d)).item())
        if dGd <= 1e-12:
            step = 0.0
        else:
            step = float(min(1.0, max(0.0, -gGd / dGd)))
        alpha = alpha + step * d
        if step < 1e-9:
            break
    alpha = torch.clamp(alpha, min=0.0)
    s = alpha.sum()
    if s <= 0:
        return vectors.new_full((m,), 1.0 / m)
    return alpha / s


def compute_mgda_advantage(
    token_level_scores_list: list[torch.Tensor],
    response_mask: torch.Tensor,
    index: np.ndarray,
    weights=None,
    gapo_p: float | None = None,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MGDA / GAPO advantage-space aggregation (see module docstring).

    If ``gapo_p`` is None -> MGDA (min-norm over raw per-objective advantage
    vectors). If ``gapo_p`` is a float -> GAPO (rescale each objective vector by
    ``1/||.||^p`` before the min-norm solve). ``weights`` optionally pre-scales
    each objective (P-GAPO preference weighting).
    """
    per_obj = _per_objective_group_advantages(
        token_level_scores_list, response_mask, index, norm_adv_by_std_in_grpo
    )
    m = len(per_obj)
    device, dtype = per_obj[0].device, per_obj[0].dtype
    # Per-sample scalar vector for each objective (stand-in for its gradient).
    a = torch.stack([adv.sum(dim=-1) for adv in per_obj], dim=0)  # (m, N)

    vecs = a.clone()
    if gapo_p is not None:
        norms = vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        vecs = vecs / norms.pow(float(gapo_p))
    if weights is not None:
        w = _normalize_weights(weights, m, device, dtype)
        vecs = vecs * w.view(m, 1)

    alpha = _frank_wolfe_min_norm(vecs)  # (m,)
    combined_scalar = (a * alpha.view(m, 1)).sum(dim=0)  # (N,)
    combined = combined_scalar.unsqueeze(-1) * response_mask
    advantages = verl_F.masked_whiten(combined, response_mask) * response_mask
    return advantages, advantages

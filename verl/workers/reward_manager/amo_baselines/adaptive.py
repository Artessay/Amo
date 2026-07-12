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
"""Adaptive-weight / constrained multi-objective baselines.

These baselines all keep the GRPO-form advantage but, unlike the fixed
scalarizers, hold *state across training steps* and adapt how objectives are
combined. They isolate the question: is HVPO's benefit coming from adaptive
balancing of objectives, or from the Pareto/HV geometry itself?

Selected via ``amo_strategy.adaptive_config.method``:

* ``lagrangian`` -- Safe-RLHF-style constrained optimization. Maximize the
  primary objective subject to each remaining objective being kept above a
  registered budget: ``r = r_primary - sum_c lambda_c (d_c - r_c)`` with dual
  ascent on ``lambda_c >= 0``. Reproduces "maximize helpfulness s.t. harmlessness
  >= threshold" rather than a symmetric sum.

* ``fair_stable`` -- Fair-and-Stable reward composition (EMNLP 2024, "Fast RL").
  Total reward is a dynamic weighted sum; weights follow a gradient-free mirror
  descent on the per-objective mean progress so a high-scale / easy objective
  cannot dominate training long-term.

* ``ctwa`` -- Covariance-Targeted Weight Adaptation (arXiv 2026). Estimate the
  on-policy covariance between each objective and the current scalar advantage
  weight; if an objective's EMA covariance drops below a registered target,
  raise its (log-space) scalarization weight. Guarantees every objective keeps a
  positive training signal.

* ``dynamic_hv`` -- Hypervolume-guided dynamic reward weighting (TACL 2026).
  Maintain a Pareto buffer of *group-mean* objective vectors and scale the
  scalarized training reward by ``0.5 + 1.5 tanh(dHV)`` where ``dHV`` is the
  group's marginal hypervolume improvement over the buffer. This is HV used at
  the *group/meta* level (contrast with HVPO's per-response exclusive-HV credit).

All state lives on the manager instance, which persists for the whole run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_baselines.common import (
    AmoBaselineRewardManager,
    group_indices,
    normalize_weights,
)
from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator
from verl.workers.reward_manager.amo_utils.pareto_cache import ParetoCache


@register("amo_adaptive")
class AmoAdaptiveRewardManager(AmoBaselineRewardManager):
    """Adaptive-weight / constrained multi-objective reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: dict,
        reward_fn_key: str = "data_source",
        adaptive_config: dict | None = None,
        **_: Any,
    ) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        cfg = dict(adaptive_config) if adaptive_config else {}
        self.adaptive_config = cfg
        self.num_obj = len(self.compute_score)

        self.method: str = cfg.get("method", "lagrangian")
        assert self.method in ("lagrangian", "fair_stable", "ctwa", "dynamic_hv"), (
            f"bad adaptive method {self.method}"
        )

        # ---- Lagrangian (Safe-RLHF style) ----
        # primary objective index; the rest are constraints r_c >= budget_c.
        self.primary_index: int = int(cfg.get("primary_index", 0))
        # per-objective cost budgets d_c (only constraint entries are used).
        budgets = cfg.get("budgets")
        if budgets is None:
            self.budgets = torch.zeros(self.num_obj, dtype=torch.float32)
        else:
            self.budgets = torch.tensor([float(x) for x in budgets], dtype=torch.float32)
            assert self.budgets.numel() == self.num_obj
        self.lambda_lr: float = float(cfg.get("lambda_lr", 0.05))
        self.lambda_init: float = float(cfg.get("lambda_init", 0.0))
        self.lambda_max: float = float(cfg.get("lambda_max", 10.0))
        self._lambdas = torch.full((self.num_obj,), self.lambda_init, dtype=torch.float32)

        # ---- Fair-and-Stable / CTWA / dynamic_hv share a weight state ----
        self._weights = normalize_weights(cfg.get("weights"), self.num_obj).clone()
        self.weight_lr: float = float(cfg.get("weight_lr", 0.1))
        self.weight_floor: float = float(cfg.get("weight_floor", 1e-3))
        self.update_window: int = int(cfg.get("update_window", 1))
        self._step: int = 0

        # Fair-and-Stable: running per-objective progress baseline.
        self._ema_score = None
        self.ema_decay: float = float(cfg.get("ema_decay", 0.9))

        # CTWA: covariance EMA + targets.
        self._ema_cov = torch.zeros(self.num_obj, dtype=torch.float32)
        self.cov_tau: float = float(cfg.get("cov_tau", 0.1))
        targets = cfg.get("cov_targets")
        if targets is None:
            self.cov_targets = torch.zeros(self.num_obj, dtype=torch.float32)
        else:
            self.cov_targets = torch.tensor([float(x) for x in targets], dtype=torch.float32)
            assert self.cov_targets.numel() == self.num_obj
        self._log_weights = torch.log(self._weights.clamp_min(self.weight_floor))

        # dynamic_hv: group-level Pareto buffer + scaling.
        self._hv_cache = ParetoCache(
            max_size=int(cfg.get("hv_buffer_size", 256)),
            eps=float(cfg.get("hv_eps", 1e-9)),
            strategy="fifo",
        )
        self.hv_ref = cfg.get("hv_reference_point")  # list or None -> zeros

        print(f"[Amo][adaptive] method={self.method} config={cfg}")

    # ------------------------------------------------------------------
    def _compute_scalar_rewards(
        self,
        score_tensor: torch.Tensor,
        uids: np.ndarray,
        is_train: bool,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        if self.method == "lagrangian":
            return self._lagrangian(score_tensor, is_train, extra)
        if self.method == "fair_stable":
            return self._fair_stable(score_tensor, is_train, extra)
        if self.method == "ctwa":
            return self._ctwa(score_tensor, uids, is_train, extra)
        return self._dynamic_hv(score_tensor, uids, is_train, extra)

    # ------------------------------------------------------------------
    # Lagrangian (Safe-RLHF)
    # ------------------------------------------------------------------
    def _lagrangian(self, scores: torch.Tensor, is_train: bool, extra: dict) -> torch.Tensor:
        m = self.num_obj
        p = self.primary_index
        lam = self._lambdas.to(scores.dtype)
        # reward = r_primary - sum_{c != p} lambda_c * (d_c - r_c)
        # (d_c - r_c) > 0 means the constraint is violated -> penalize.
        reward = scores[:, p].clone()
        for c in range(m):
            if c == p:
                continue
            reward = reward - lam[c] * (self.budgets[c] - scores[:, c])

        if is_train:
            # Dual ascent: increase lambda_c when the constraint is violated
            # (batch-mean r_c below budget d_c).
            with torch.no_grad():
                mean_c = scores.mean(dim=0)
                violation = (self.budgets - mean_c)  # >0 -> violated
                new_lam = self._lambdas.clone()
                for c in range(m):
                    if c == p:
                        continue
                    new_lam[c] = float(
                        np.clip(
                            self._lambdas[c].item() + self.lambda_lr * violation[c].item(),
                            0.0,
                            self.lambda_max,
                        )
                    )
                self._lambdas = new_lam
                for c in range(m):
                    extra["reward_extra_info"].setdefault(f"lambda_{c}", [])
        return reward

    # ------------------------------------------------------------------
    # Fair-and-Stable dynamic composition (mirror descent on weights)
    # ------------------------------------------------------------------
    def _fair_stable(self, scores: torch.Tensor, is_train: bool, extra: dict) -> torch.Tensor:
        w = self._weights.to(scores.dtype)
        reward = scores @ w

        if is_train:
            with torch.no_grad():
                mean_c = scores.mean(dim=0)
                if self._ema_score is None:
                    self._ema_score = mean_c.clone()
                # Progress since last window = how much each objective improved.
                progress = mean_c - self._ema_score
                self._ema_score = self.ema_decay * self._ema_score + (1 - self.ema_decay) * mean_c
                self._step += 1
                if self._step % self.update_window == 0:
                    # Mirror descent: up-weight lagging objectives (low progress).
                    # Use negative progress as the "loss" gradient surrogate.
                    grad = -progress
                    log_w = torch.log(self._weights.clamp_min(self.weight_floor))
                    log_w = log_w - self.weight_lr * grad
                    new_w = torch.softmax(log_w, dim=0)
                    new_w = new_w.clamp_min(self.weight_floor)
                    self._weights = new_w / new_w.sum()
                for c in range(self.num_obj):
                    extra["reward_extra_info"].setdefault(f"weight_{c}", [])
        return reward

    # ------------------------------------------------------------------
    # CTWA (covariance-targeted weight adaptation)
    # ------------------------------------------------------------------
    def _ctwa(self, scores: torch.Tensor, uids: np.ndarray, is_train: bool, extra: dict) -> torch.Tensor:
        w = self._weights.to(scores.dtype)
        reward = scores @ w

        if is_train:
            with torch.no_grad():
                # Scalar advantage weight per response: group-centered scalar reward.
                scalar = reward
                id2rows = group_indices(uids)
                adv = torch.zeros_like(scalar)
                for _uid, rows in id2rows.items():
                    idx = torch.tensor(rows, dtype=torch.long)
                    block = scalar.index_select(0, idx)
                    adv.index_copy_(0, idx, block - block.mean())
                # Per-objective covariance with the advantage weight.
                cov = torch.zeros(self.num_obj, dtype=torch.float32)
                for c in range(self.num_obj):
                    rc = scores[:, c]
                    cov[c] = ((rc - rc.mean()) * adv).mean()
                self._ema_cov = (1 - self.cov_tau) * self._ema_cov + self.cov_tau * cov
                # Raise log-weight for objectives whose covariance is below target.
                delta = torch.clamp(self.cov_targets - self._ema_cov, min=0.0)
                self._log_weights = self._log_weights + self.weight_lr * delta
                new_w = torch.softmax(self._log_weights, dim=0).clamp_min(self.weight_floor)
                self._weights = new_w / new_w.sum()
                for c in range(self.num_obj):
                    extra["reward_extra_info"].setdefault(f"ctwa_cov_{c}", [])
                    extra["reward_extra_info"].setdefault(f"ctwa_weight_{c}", [])
        return reward

    # ------------------------------------------------------------------
    # Hypervolume-guided dynamic reward weighting (group/meta level)
    # ------------------------------------------------------------------
    def _dynamic_hv(self, scores: torch.Tensor, uids: np.ndarray, is_train: bool, extra: dict) -> torch.Tensor:
        w = self._weights.to(scores.dtype)
        base = scores @ w  # scalarized training reward
        if not is_train:
            return base

        ref = (
            torch.tensor(self.hv_ref, dtype=torch.float32)
            if self.hv_ref is not None
            else torch.zeros(self.num_obj, dtype=torch.float32)
        )
        with torch.no_grad():
            snapshot = self._hv_cache.get_snapshot()
            if snapshot:
                buf = torch.tensor(snapshot, dtype=torch.float32)
                hv_before = HypervolumeCalculator.calculate_hypervolume(buf, ref)
            else:
                buf = scores.new_zeros((0, self.num_obj))
                hv_before = torch.tensor(0.0)

            id2rows = group_indices(uids)
            reward = base.clone()
            group_means = []
            for _uid, rows in id2rows.items():
                idx = torch.tensor(rows, dtype=torch.long)
                gmean = scores.index_select(0, idx).mean(dim=0, keepdim=True)  # (1, m)
                group_means.append(gmean)
                union = torch.cat([buf, gmean], dim=0)
                hv_after = HypervolumeCalculator.calculate_hypervolume(union, ref)
                dhv = (hv_after - hv_before)
                scale = 0.5 + 1.5 * torch.tanh(dhv)
                reward.index_copy_(0, idx, base.index_select(0, idx) * scale)
            if group_means:
                self._hv_cache.update(torch.cat(group_means, dim=0).tolist())
            extra["reward_extra_info"].setdefault("dynamic_hv_buffer_size", [])
        return reward

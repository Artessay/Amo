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
"""Fixed-scalarization multi-objective baselines (LS sweep + Tchebycheff).

Two of the cheapest, most important controlled baselines for HVPO. Both keep the
GRPO-form advantage (``algorithm.adv_estimator=grpo``) and change *only* how the
per-objective score vector becomes a scalar reward:

* ``linear`` -- linear scalarization ``sum_j w_j r_j`` (MORLHF). Sweeping the
  weight simplex traces the linear-scalarization Pareto front. This is the
  minimum bar every complex method must beat and the reference for deciding
  whether a non-convex front actually needs HV/Tchebycheff.

* ``tchebycheff`` -- (augmented) Tchebycheff scalarization against a fixed ideal
  point. Reaches concave front regions that linear weights miss.

Config lives under ``amo_strategy.scalarize_config``::

    amo_strategy:
      scalarize_config:
        method: linear            # linear | tchebycheff
        weights: [0.5, 0.5]       # null -> uniform; length must == #objectives
        normalize: none           # none | affine (per-objective calibration)
        calib_lower: [0,0]        # affine lower bounds (only if normalize=affine)
        calib_upper: [1,1]        # affine upper bounds
        ideal: [1,1]              # tchebycheff ideal point (defaults to 1s)
        rho: 0.05                 # augmented-Tchebycheff coefficient

Weights / ideal / calibration bounds must be chosen on a frozen calibration
split, never from the test statistics, and reused verbatim across every method.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_baselines.common import (
    AmoBaselineRewardManager,
    linear_scalarize,
    normalize_weights,
    tchebycheff_scalarize,
)


@register("amo_scalarize")
class AmoScalarizeRewardManager(AmoBaselineRewardManager):
    """Linear / Tchebycheff scalarization reward manager (advantage stays GRPO)."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: dict,
        reward_fn_key: str = "data_source",
        scalarize_config: dict | None = None,
        **_: Any,
    ) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        cfg = dict(scalarize_config) if scalarize_config else {}
        self.scalarize_config = cfg

        num_obj = len(self.compute_score)
        self.method: str = cfg.get("method", "linear")
        assert self.method in ("linear", "tchebycheff"), f"bad scalarize method {self.method}"

        self.weights = normalize_weights(cfg.get("weights"), num_obj)

        # Optional per-objective affine calibration r~ = (r - lo) / (hi - lo).
        self.normalize: str = cfg.get("normalize", "none")
        assert self.normalize in ("none", "affine")
        if self.normalize == "affine":
            lo = cfg.get("calib_lower")
            hi = cfg.get("calib_upper")
            assert lo is not None and hi is not None, "affine normalize needs calib_lower/calib_upper"
            self.calib_lower = torch.tensor([float(x) for x in lo], dtype=torch.float32)
            self.calib_upper = torch.tensor([float(x) for x in hi], dtype=torch.float32)
            assert self.calib_lower.numel() == num_obj and self.calib_upper.numel() == num_obj

        # Tchebycheff ideal point + augmentation.
        ideal = cfg.get("ideal")
        if ideal is None:
            # Objectives normalized to [0,1] -> utopia at all-ones.
            self.ideal = torch.ones(num_obj, dtype=torch.float32)
        else:
            self.ideal = torch.tensor([float(x) for x in ideal], dtype=torch.float32)
            assert self.ideal.numel() == num_obj
        self.rho: float = float(cfg.get("rho", 0.0))

        print(
            f"[Amo][scalarize] method={self.method} weights={self.weights.tolist()} "
            f"normalize={self.normalize} ideal={self.ideal.tolist()} rho={self.rho}"
        )

    def _calibrate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalize == "affine":
            span = (self.calib_upper - self.calib_lower).clamp_min(1e-8)
            return (scores - self.calib_lower) / span
        return scores

    def _compute_scalar_rewards(
        self,
        score_tensor: torch.Tensor,
        uids: np.ndarray,
        is_train: bool,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        scores = self._calibrate(score_tensor)
        if self.method == "linear":
            return linear_scalarize(scores, self.weights)
        return tchebycheff_scalarize(scores, self.weights, self.ideal, self.rho)

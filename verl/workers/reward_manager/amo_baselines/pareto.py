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
"""Pareto-selection *response credit* baselines (MOO mechanism ablation).

These are not off-the-shelf LLM-alignment algorithms; they adapt classic
multi-objective-EA survival/credit rules into a per-response scalar reward, on
the *same* rollout groups HVPO uses. They answer: does HVPO beat common Pareto
ranking / survival rules? In the paper these should be called
"NSGA-II-style / SMS-EMOA-style response credit", not "NSGA-II training an LLM".

Selected via ``amo_strategy.pareto_config.method``:

* ``nsga2`` -- within each ``uid`` group, fast-non-dominated-sort the response
  objective vectors, then credit each response by ``-(rank) + crowding_bonus``:
  earlier fronts get higher reward and, within a front, more isolated points
  (larger crowding distance) are preferred, encouraging front coverage.

* ``smsemoa`` -- credit each *non-dominated* response by its exclusive
  hypervolume contribution within the group (dominated responses get a small
  negative fallback). This is the closest MOO cousin of HVPO and is best read as
  a component ablation of it.

The GRPO-form advantage is unchanged; only the scalar credit differs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_baselines.common import (
    AmoBaselineRewardManager,
    group_indices,
)
from verl.workers.reward_manager.amo_utils.hybrid_reward import HybridRewardModel


def fast_non_dominated_sort(points: torch.Tensor) -> list[int]:
    """Return the non-domination rank (0 = best front) of each row.

    Maximization convention: ``a`` dominates ``b`` iff ``a >= b`` in every
    coordinate and ``a > b`` in at least one. Straightforward O(n^2 m) sort,
    which is fine for the small rollout groups (G ~ 4-16) used here.
    """
    n = points.shape[0]
    if n == 0:
        return []
    dominated_by = [0] * n  # how many points dominate i
    dominates: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ge = bool(torch.all(points[j] >= points[i]))
            gt = bool(torch.any(points[j] > points[i]))
            if ge and gt:  # j dominates i
                dominated_by[i] += 1
    # Assign fronts iteratively.
    rank = [-1] * n
    current = [i for i in range(n) if dominated_by[i] == 0]
    counts = dominated_by[:]
    # Precompute domination lists for peeling.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ge = bool(torch.all(points[i] >= points[j]))
            gt = bool(torch.any(points[i] > points[j]))
            if ge and gt:  # i dominates j
                dominates[i].append(j)
    front = 0
    assigned = 0
    while current:
        nxt: list[int] = []
        for i in current:
            rank[i] = front
            assigned += 1
        for i in current:
            for j in dominates[i]:
                counts[j] -= 1
                if counts[j] == 0 and rank[j] == -1:
                    nxt.append(j)
        current = nxt
        front += 1
    # Any leftover (should not happen) gets the last front.
    for i in range(n):
        if rank[i] == -1:
            rank[i] = front
    return rank


def crowding_distance(points: torch.Tensor, members: list[int]) -> dict[int, float]:
    """NSGA-II crowding distance for a set of rows forming one front.

    Boundary points get +inf (mapped to a large finite bonus by the caller).
    """
    out = {i: 0.0 for i in members}
    if len(members) <= 2:
        return {i: float("inf") for i in members}
    m = points.shape[1]
    for d in range(m):
        vals = [(points[i, d].item(), i) for i in members]
        vals.sort(key=lambda t: t[0])
        lo = vals[0][0]
        hi = vals[-1][0]
        span = hi - lo
        out[vals[0][1]] = float("inf")
        out[vals[-1][1]] = float("inf")
        if span <= 0:
            continue
        for k in range(1, len(vals) - 1):
            prev_v = vals[k - 1][0]
            next_v = vals[k + 1][0]
            out[vals[k][1]] += (next_v - prev_v) / span
    return out


@register("amo_pareto")
class AmoParetoRewardManager(AmoBaselineRewardManager):
    """NSGA-II-style / SMS-EMOA-style response-credit reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: dict,
        reward_fn_key: str = "data_source",
        pareto_config: dict | None = None,
        **_: Any,
    ) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        cfg = dict(pareto_config) if pareto_config else {}
        self.pareto_config = cfg
        self.num_obj = len(self.compute_score)

        self.method: str = cfg.get("method", "nsga2")
        assert self.method in ("nsga2", "smsemoa"), f"bad pareto method {self.method}"

        # NSGA-II crowding bonus scaling + finite value for boundary (inf) points.
        self.crowding_coef: float = float(cfg.get("crowding_coef", 1.0))
        self.crowding_inf: float = float(cfg.get("crowding_inf", 2.0))

        # SMS-EMOA HV reference point + distance fallback.
        self.hv_reference_point = cfg.get("hv_reference_point")  # list or None
        self.distance_metric: str = cfg.get("distance_metric", "chebyshev")
        assert self.distance_metric in ("chebyshev", "euclidean", "none")

        print(f"[Amo][pareto] method={self.method} config={cfg}")

    # ------------------------------------------------------------------
    def _compute_scalar_rewards(
        self,
        score_tensor: torch.Tensor,
        uids: np.ndarray,
        is_train: bool,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        rewards = torch.zeros(score_tensor.shape[0], dtype=torch.float32)
        id2rows = group_indices(uids)
        for _uid, rows in id2rows.items():
            block = score_tensor[rows]  # (g, m)
            if self.method == "nsga2":
                credit = self._nsga2_credit(block)
            else:
                credit = self._smsemoa_credit(block)
            for local, gidx in enumerate(rows):
                rewards[gidx] = credit[local]
        return rewards

    def _nsga2_credit(self, block: torch.Tensor) -> torch.Tensor:
        g = block.shape[0]
        ranks = fast_non_dominated_sort(block)
        # Group members by front for crowding.
        front2members: dict[int, list[int]] = {}
        for i, r in enumerate(ranks):
            front2members.setdefault(r, []).append(i)
        crowd = {}
        for _front, members in front2members.items():
            cd = crowding_distance(block, members)
            crowd.update(cd)
        credit = torch.zeros(g, dtype=torch.float32)
        for i in range(g):
            cd = crowd[i]
            cd_bonus = self.crowding_inf if cd == float("inf") else cd
            # earlier front (lower rank) -> higher credit; crowding as tie-break.
            credit[i] = -float(ranks[i]) + self.crowding_coef * cd_bonus
        return credit

    def _smsemoa_credit(self, block: torch.Tensor) -> torch.Tensor:
        ref = (
            torch.tensor(self.hv_reference_point, dtype=torch.float32)
            if self.hv_reference_point is not None
            else block.min(dim=0).values
        )
        # ensure ref dominated by all points in group
        ref = torch.minimum(ref, block.min(dim=0).values)
        empty = block.new_zeros((0, block.shape[1]))
        return HybridRewardModel.compute_group_hybrid_rewards(
            block, empty, ref, distance_metric=self.distance_metric
        )

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
"""Unit tests for the HVPO multi-objective reward machinery.

Run with:  pytest tests/amo/test_hvpo.py -v
These tests are CPU-only and require no model, GPU or external server.
"""

import torch

from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator
from verl.workers.reward_manager.amo_utils.hybrid_reward import HybridRewardModel
from verl.workers.reward_manager.amo_utils.pareto_cache import ParetoCache


# ----------------------------------------------------------------------
# Hypervolume correctness
# ----------------------------------------------------------------------
def test_hv_known_value_2d():
    pts = torch.tensor([[3.0, 4.0], [4.0, 3.0]])
    ref = torch.tensor([0.0, 0.0])
    hv = HypervolumeCalculator.calculate_hypervolume(pts, ref).item()
    # Union of [0,3]x[0,4] and [0,4]x[0,3] = 12 + 12 - 9 = 15
    assert abs(hv - 15.0) < 1e-6


def test_hv_ignores_dominated_points():
    pts = torch.tensor([[3.0, 4.0], [4.0, 3.0], [1.0, 1.0]])  # (1,1) is dominated
    ref = torch.tensor([0.0, 0.0])
    hv = HypervolumeCalculator.calculate_hypervolume(pts, ref).item()
    assert abs(hv - 15.0) < 1e-6


def test_hv_ignores_points_below_reference():
    pts = torch.tensor([[3.0, 4.0], [4.0, 3.0], [-5.0, 100.0]])
    ref = torch.tensor([0.0, 0.0])
    hv = HypervolumeCalculator.calculate_hypervolume(pts, ref).item()
    assert abs(hv - 15.0) < 1e-6


def test_hv_single_point_is_box_volume():
    pts = torch.tensor([[2.0, 3.0, 4.0]])
    ref = torch.tensor([0.0, 0.0, 0.0])
    hv = HypervolumeCalculator.calculate_hypervolume(pts, ref).item()
    assert abs(hv - 24.0) < 1e-6


def test_hv_monotonicity():
    """Adding a non-dominated point can only increase HV."""
    ref = torch.tensor([0.0, 0.0])
    base = torch.tensor([[3.0, 1.0]])
    more = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    hv_base = HypervolumeCalculator.calculate_hypervolume(base, ref).item()
    hv_more = HypervolumeCalculator.calculate_hypervolume(more, ref).item()
    assert hv_more > hv_base


# ----------------------------------------------------------------------
# The core fix: exclusive group contribution rewards diversity
# ----------------------------------------------------------------------
def test_exclusive_contribution_rewards_diversity():
    """A spread-out group should earn strictly more total reward than a group
    of duplicates covering the same region — this is the diversity pressure a
    plain weighted-sum (vanilla) reward cannot express."""
    ref = torch.tensor([0.0, 0.0])
    empty_front = torch.zeros((0, 2))

    diverse = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    duplicate = torch.tensor([[3.0, 1.0], [3.0, 1.0]])

    r_div = HybridRewardModel.compute_group_hybrid_rewards(diverse, empty_front, ref)
    r_dup = HybridRewardModel.compute_group_hybrid_rewards(duplicate, empty_front, ref)

    assert r_div.sum().item() > r_dup.sum().item()
    # Duplicates share the region: each exclusive contribution is ~0.
    assert abs(r_dup.sum().item()) < 1e-6


def test_exclusive_contribution_balanced_beats_extreme_equal_sum():
    """Two candidates with the SAME scalar sum but different balance: the
    balanced one adds more exclusive volume against an existing front."""
    ref = torch.tensor([0.0, 0.0])
    # Front already covers a corner near (2, 0.2).
    front = torch.tensor([[2.0, 0.2]])

    # group member A: balanced (1.0, 1.0)  sum = 2.0
    # group member B: extreme  (1.9, 0.1)  sum = 2.0 but close to front corner
    group = torch.tensor([[1.0, 1.0], [1.9, 0.1]])
    r = HybridRewardModel.compute_group_hybrid_rewards(group, front, ref)
    # The balanced member opens up genuinely new area; the extreme one overlaps
    # the region already dominated by the front point.
    assert r[0].item() > r[1].item()


def test_dominated_point_gets_negative_reward():
    ref = torch.tensor([0.0, 0.0])
    front = torch.tensor([[5.0, 5.0]])
    group = torch.tensor([[1.0, 1.0]])  # strictly dominated by the front
    r = HybridRewardModel.compute_group_hybrid_rewards(group, front, ref, distance_metric="chebyshev")
    assert r[0].item() < 0.0


def test_single_member_group_equals_plain_contribution():
    ref = torch.tensor([0.0, 0.0])
    front = torch.tensor([[2.0, 3.0], [5.0, 2.0]])
    point = torch.tensor([[4.0, 5.0]])
    r_group = HybridRewardModel.compute_group_hybrid_rewards(point, front, ref)
    r_single = HybridRewardModel.compute_hybrid_reward(point[0], front, ref)
    assert abs(r_group[0].item() - r_single.item()) < 1e-6


# ----------------------------------------------------------------------
# Pareto cache invariants
# ----------------------------------------------------------------------
def _fresh_cache(max_size=1024, eps=1e-9):
    # ParetoCache is a process singleton; reset it for a clean test.
    ParetoCache._instance = None
    cache = ParetoCache(max_size=max_size, eps=eps, strategy="fifo")
    cache.clear()
    return cache


def test_cache_keeps_only_nondominated():
    cache = _fresh_cache()
    cache.update([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    cache.update([[0.5, 0.5]])  # dominated, must not appear
    snap = cache.get_snapshot()
    assert [0.5, 0.5] not in snap
    # All retained points are mutually non-dominated.
    for a in snap:
        for b in snap:
            if a is b:
                continue
            ge = all(x >= y for x, y in zip(a, b))
            gt = any(x > y for x, y in zip(a, b))
            assert not (ge and gt), f"{a} dominates {b} but both are in cache"


def test_cache_dominating_point_evicts_others():
    cache = _fresh_cache()
    cache.update([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    cache.update([[3.0, 3.0]])  # dominates everything
    snap = cache.get_snapshot()
    assert snap == [[3.0, 3.0]]


def test_cache_respects_max_size():
    cache = _fresh_cache(max_size=5)
    cache.update([[float(i), float(10 - i)] for i in range(20)])
    assert cache.size() <= 5


# ----------------------------------------------------------------------
# End-to-end sanity: front update + exclusive reward stay consistent
# ----------------------------------------------------------------------
def test_front_update_then_contribution_is_zero_for_cached_point():
    """After a point is absorbed into the front, re-presenting it should add
    (approximately) no new volume."""
    cache = _fresh_cache()
    ref = torch.tensor([0.0, 0.0])
    cache.update([[3.0, 1.0], [1.0, 3.0]])
    front = torch.tensor(cache.get_snapshot(), dtype=torch.float32)

    repeat = torch.tensor([[3.0, 1.0]])
    r = HybridRewardModel.compute_group_hybrid_rewards(repeat, front, ref)
    assert r[0].item() <= 1e-6

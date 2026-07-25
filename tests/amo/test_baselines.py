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
"""CPU-only unit tests for the Amo multi-objective *baseline* machinery.

These cover the pure numeric cores of every baseline (scalarization, group
z-scoring, RVPO soft-min limits, MGDA/GAPO min-norm aggregation, NSGA-II sort +
crowding, SMS-EMOA credit, and the adaptive-weight dynamics). No model, GPU or
reward server is required.

Run with:  pytest tests/amo/test_baselines.py -v
"""

import numpy as np
import torch

from verl.workers.reward_manager.amo_baselines.common import (
    group_zscore,
    linear_scalarize,
    normalize_weights,
    tchebycheff_scalarize,
)
from verl.workers.reward_manager.amo_baselines.pareto import (
    crowding_distance,
    fast_non_dominated_sort,
)
from verl.trainer.ppo import amo_mo_advantages as amo_adv
from verl.trainer.ppo.ray_trainer import _ordered_amo_objective_score_keys


# ----------------------------------------------------------------------
# Scalarization helpers
# ----------------------------------------------------------------------
def test_normalize_weights_uniform_and_explicit():
    w = normalize_weights(None, 4)
    assert torch.allclose(w, torch.full((4,), 0.25))
    w2 = normalize_weights([1, 3], 2)
    assert torch.allclose(w2, torch.tensor([0.25, 0.75]))


def test_objective_weight_order_matches_reward_function_insertion_order():
    batch = {
        "token_level_scores_coherence": object(),
        "token_level_scores_fluency": object(),
        "token_level_scores_relevance": object(),
        "token_level_scores_consistency": object(),
    }

    assert _ordered_amo_objective_score_keys(batch) == list(batch)


def test_normalize_weights_rejects_bad():
    for bad in ([1, 2, 3], [-1, 2], [0, 0]):
        try:
            normalize_weights(bad, 2)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for weights={bad}")


def test_linear_scalarize_matches_manual():
    scores = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    w = normalize_weights([0.25, 0.75], 2)
    out = linear_scalarize(scores, w)
    assert torch.allclose(out, torch.tensor([0.25, 0.75, 0.5]))


def test_tchebycheff_prefers_balanced_point():
    # ideal = (1,1); equal weights. Balanced (0.6,0.6) should beat lopsided
    # (0.9,0.1) under Tchebycheff even though linear sum is equal (both 1.0... not here)
    ideal = torch.ones(2)
    w = torch.tensor([0.5, 0.5])
    balanced = torch.tensor([[0.6, 0.6]])
    lopsided = torch.tensor([[0.9, 0.3]])
    tb = tchebycheff_scalarize(balanced, w, ideal, rho=0.0)
    tl = tchebycheff_scalarize(lopsided, w, ideal, rho=0.0)
    # balanced gap = 0.5*0.4=0.2 -> -0.2 ; lopsided gap = 0.5*0.7=0.35 -> -0.35
    assert tb.item() > tl.item()
    assert abs(tb.item() - (-0.2)) < 1e-6
    assert abs(tl.item() - (-0.35)) < 1e-6


def test_augmented_tchebycheff_adds_linear_term():
    ideal = torch.ones(2)
    w = torch.tensor([0.5, 0.5])
    pt = torch.tensor([[0.6, 0.6]])
    plain = tchebycheff_scalarize(pt, w, ideal, rho=0.0).item()
    aug = tchebycheff_scalarize(pt, w, ideal, rho=0.1).item()
    # augmented adds rho * (w . r) = 0.1 * 0.6 = 0.06
    assert abs((aug - plain) - 0.06) < 1e-6


# ----------------------------------------------------------------------
# Group z-score (GDPO standardization)
# ----------------------------------------------------------------------
def test_group_zscore_centers_each_group_and_objective():
    scores = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
    index = np.array(["a", "a", "b", "b"], dtype=object)
    z = group_zscore(scores, index)
    # each group/objective column should be zero-mean.
    for uid, rows in {"a": [0, 1], "b": [2, 3]}.items():
        blk = z[rows]
        assert torch.allclose(blk.mean(dim=0), torch.zeros(2), atol=1e-5)


# ----------------------------------------------------------------------
# RVPO soft-min limits
# ----------------------------------------------------------------------
def _make_scores_batch(scalar_per_obj):
    """Build token_level_scores_list with the scalar placed at the last token."""
    n, m = scalar_per_obj.shape
    T = 3
    mask = torch.ones(n, T)
    tls = []
    for j in range(m):
        t = torch.zeros(n, T)
        t[:, -1] = scalar_per_obj[:, j]
        tls.append(t)
    return tls, mask


def test_rvpo_k_to_zero_recovers_gdpo_mean():
    torch.manual_seed(0)
    scalar = torch.randn(8, 3)
    tls, mask = _make_scores_batch(scalar)
    index = np.array(["g"] * 8, dtype=object)
    a_mean, _ = amo_adv.compute_rvpo_advantage(tls, mask, index, k=0.0)
    a_gdpo, _ = amo_adv.compute_gdpo_weighted_advantage(tls, mask, index, weights=None)
    # Both whiten the mean-aggregated per-objective z-scores -> identical.
    assert torch.allclose(a_mean, a_gdpo, atol=1e-5)


def test_rvpo_large_k_approaches_worst_objective():
    # One sample has a clearly worst objective; soft-min should weight it.
    scalar = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, -5.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    tls, mask = _make_scores_batch(scalar)
    index = np.array(["g"] * 4, dtype=object)
    # pre-whiten aggregated scalar: compare ranking of sample 1 (has a -5 obj).
    per_obj = amo_adv._per_objective_group_advantages(tls, mask, index)
    z = torch.stack([adv.sum(dim=-1) for adv in per_obj], dim=0)  # (m, n)
    m = z.shape[0]
    neg = -1e3 * z
    lse = torch.logsumexp(neg, dim=0) - np.log(m)
    softmin = -lse / 1e3
    hardmin = z.min(dim=0).values
    # soft-min converges to hard-min up to the exact log(m)/k correction term.
    assert torch.allclose(softmin, hardmin, atol=np.log(m) / 1e3 + 1e-4)


# ----------------------------------------------------------------------
# MGDA / GAPO min-norm aggregation
# ----------------------------------------------------------------------
def test_min_norm_two_opposed_vectors():
    # g0 = (1,0), g1 = (-1,0): min-norm convex combo is 0.5/0.5 -> zero vector.
    v = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    alpha = amo_adv._frank_wolfe_min_norm(v)
    assert torch.allclose(alpha, torch.tensor([0.5, 0.5]), atol=1e-4)


def test_min_norm_three_vectors_simplex():
    torch.manual_seed(1)
    v = torch.randn(3, 10)
    alpha = amo_adv._frank_wolfe_min_norm(v)
    assert alpha.numel() == 3
    assert abs(alpha.sum().item() - 1.0) < 1e-5
    assert torch.all(alpha >= -1e-6)
    # The resulting combined vector norm must be <= any single vector norm.
    combined = (v * alpha.view(3, 1)).sum(dim=0)
    single_norms = v.norm(dim=1)
    assert combined.norm().item() <= single_norms.min().item() + 1e-4


def test_mgda_and_gapo_shapes_and_finiteness():
    torch.manual_seed(2)
    scalar = torch.randn(12, 3)
    tls, mask = _make_scores_batch(scalar)
    index = np.array((["a"] * 6) + (["b"] * 6), dtype=object)
    a_mgda, _ = amo_adv.compute_mgda_advantage(tls, mask, index)
    a_gapo, _ = amo_adv.compute_mgda_advantage(tls, mask, index, gapo_p=1.0)
    assert a_mgda.shape == mask.shape
    assert a_gapo.shape == mask.shape
    assert torch.isfinite(a_mgda).all()
    assert torch.isfinite(a_gapo).all()


# ----------------------------------------------------------------------
# NSGA-II sort + crowding
# ----------------------------------------------------------------------
def test_fast_non_dominated_sort_two_fronts():
    # (2,2) and (3,1)... make a clear front-0 and a dominated front-1.
    pts = torch.tensor([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
    ranks = fast_non_dominated_sort(pts)
    # (3,3) dominates the others -> front 0; (2,2) dominates (1,1); (1,1) worst.
    assert ranks[0] == 0
    assert ranks[2] == 1
    assert ranks[1] == 2


def test_non_dominated_front_all_rank_zero():
    pts = torch.tensor([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
    ranks = fast_non_dominated_sort(pts)
    assert ranks == [0, 0, 0]


def test_crowding_distance_boundaries_infinite():
    pts = torch.tensor([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    cd = crowding_distance(pts, [0, 1, 2, 3])
    # extreme points along the sort are boundary -> inf.
    assert cd[0] == float("inf")
    assert cd[3] == float("inf")
    assert cd[1] != float("inf") and cd[2] != float("inf")


# ----------------------------------------------------------------------
# SMS-EMOA credit via the pareto manager helper
# ----------------------------------------------------------------------
def test_smsemoa_credit_rewards_nondominated():
    from verl.workers.reward_manager.amo_utils.hybrid_reward import HybridRewardModel

    block = torch.tensor([[3.0, 1.0], [1.0, 3.0], [0.5, 0.5]])  # third is dominated
    ref = block.min(dim=0).values
    empty = block.new_zeros((0, 2))
    credit = HybridRewardModel.compute_group_hybrid_rewards(block, empty, ref, "chebyshev")
    # dominated point gets the smallest credit.
    assert credit.argmin().item() == 2


# ----------------------------------------------------------------------
# Manager-level scalar-reward hook (constructed without a tokenizer)
# ----------------------------------------------------------------------
def _fake_score_fns(m):
    """A dict of m dummy reward fns; only len() and keys() are used here."""
    return {f"obj{j}": (lambda **kw: 0.0) for j in range(m)}


def _new_manager(cls, m, **cfg_kwargs):
    """Instantiate a baseline manager bypassing tokenizer/scoring setup."""
    mgr = cls.__new__(cls)
    # Minimal attributes used by _compute_scalar_rewards.
    mgr.tokenizer = None
    mgr.num_examine = 0
    mgr.reward_fn_key = "data_source"
    mgr.compute_score = _fake_score_fns(m)
    return mgr


def test_scalarize_manager_linear_and_tchebycheff():
    from verl.workers.reward_manager.amo_baselines.scalarize import AmoScalarizeRewardManager

    scores = torch.tensor([[0.9, 0.3], [0.6, 0.6]])
    uids = np.array(["g", "g"], dtype=object)

    mgr = _new_manager(AmoScalarizeRewardManager, 2)
    AmoScalarizeRewardManager.__init__(
        mgr, None, 0, _fake_score_fns(2),
        scalarize_config={"method": "linear", "weights": [0.5, 0.5]},
    )
    lin = mgr._compute_scalar_rewards(scores, uids, True, {"reward_extra_info": {}})
    assert torch.allclose(lin, torch.tensor([0.6, 0.6]))

    mgr2 = _new_manager(AmoScalarizeRewardManager, 2)
    AmoScalarizeRewardManager.__init__(
        mgr2, None, 0, _fake_score_fns(2),
        scalarize_config={"method": "tchebycheff", "weights": [0.5, 0.5]},
    )
    tch = mgr2._compute_scalar_rewards(scores, uids, True, {"reward_extra_info": {}})
    # balanced (0.6,0.6) beats lopsided (0.9,0.3) under Tchebycheff.
    assert tch[1].item() > tch[0].item()


def test_lagrangian_manager_updates_lambda_on_violation():
    from verl.workers.reward_manager.amo_baselines.adaptive import AmoAdaptiveRewardManager

    mgr = _new_manager(AmoAdaptiveRewardManager, 2)
    AmoAdaptiveRewardManager.__init__(
        mgr, None, 0, _fake_score_fns(2),
        adaptive_config={
            "method": "lagrangian", "primary_index": 0,
            "budgets": [0.0, 0.8], "lambda_lr": 0.5, "lambda_init": 0.0,
        },
    )
    # objective 1 mean = 0.2 < budget 0.8 -> constraint violated -> lambda_1 up.
    scores = torch.tensor([[0.5, 0.2], [0.5, 0.2]])
    uids = np.array(["g", "g"], dtype=object)
    lam_before = mgr._lambdas[1].item()
    mgr._compute_scalar_rewards(scores, uids, True, {"reward_extra_info": {}})
    assert mgr._lambdas[1].item() > lam_before


def test_ctwa_manager_raises_lagging_objective_weight():
    from verl.workers.reward_manager.amo_baselines.adaptive import AmoAdaptiveRewardManager

    mgr = _new_manager(AmoAdaptiveRewardManager, 2)
    AmoAdaptiveRewardManager.__init__(
        mgr, None, 0, _fake_score_fns(2),
        adaptive_config={
            "method": "ctwa", "cov_targets": [0.0, 5.0], "cov_tau": 1.0, "weight_lr": 1.0,
        },
    )
    w_before = mgr._weights.clone()
    torch.manual_seed(0)
    scores = torch.rand(8, 2)
    uids = np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    mgr._compute_scalar_rewards(scores, uids, True, {"reward_extra_info": {}})
    # objective 1 has an unreachable covariance target -> its weight should grow.
    assert mgr._weights[1].item() > w_before[1].item()
    assert abs(mgr._weights.sum().item() - 1.0) < 1e-5


def test_pareto_manager_scalar_shapes():
    from verl.workers.reward_manager.amo_baselines.pareto import AmoParetoRewardManager

    torch.manual_seed(0)
    scores = torch.rand(8, 3)
    uids = np.array((["a"] * 4) + (["b"] * 4), dtype=object)
    for method in ("nsga2", "smsemoa"):
        mgr = _new_manager(AmoParetoRewardManager, 3)
        AmoParetoRewardManager.__init__(
            mgr, None, 0, _fake_score_fns(3), pareto_config={"method": method},
        )
        out = mgr._compute_scalar_rewards(scores, uids, True, {"reward_extra_info": {}})
        assert out.shape == (8,)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

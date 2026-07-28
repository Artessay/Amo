"""CPU-only regression tests for paper-faithful response-wise HVPO."""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_hvpo_outcome_advantage
from verl.workers.reward_manager.amo_hvpo import AmoHvpoRewardManager, compute_hvpo_credit


def _credit(points, lower=None, upper=None, reference=None):
    points = torch.tensor(points, dtype=torch.float32)
    dim = points.shape[1]
    lower = torch.tensor(lower or [0.0] * dim)
    upper = torch.tensor(upper or [1.0] * dim)
    reference = torch.tensor(reference or [0.0] * dim)
    return compute_hvpo_credit(points, lower, upper, reference)


def test_rooted_singleton_hypervolume():
    q, z, d = _credit([[0.25, 1.0], [0.5, 0.5]])
    assert torch.allclose(q, torch.tensor([0.5, 0.5]))
    assert torch.equal(z, torch.tensor([[0.25, 1.0], [0.5, 0.5]]))
    assert torch.equal(d, torch.zeros(2))


def test_reference_boundary_is_exactly_zero():
    q, _, _ = _credit([[0.0, 0.8]])
    assert q.item() == 0.0


def test_directional_shortfall_uses_worst_fixed_reference_deficit():
    q, z, d = _credit([[-0.2, 0.7], [-0.1, -0.4]])
    assert torch.allclose(q, torch.tensor([-0.2, -0.4]))
    assert torch.allclose(q, -d)
    assert torch.equal(z, torch.tensor([[-0.2, 0.7], [-0.1, -0.4]]))


def test_fixed_affine_calibration_is_not_clipped():
    q, z, _ = _credit([[-2.0, 14.0]], lower=[0.0, 10.0], upper=[2.0, 12.0])
    assert torch.equal(z, torch.tensor([[-1.0, 2.0]]))
    assert q.item() == -1.0


def test_credit_is_response_wise_and_pareto_monotone():
    first, _, _ = _credit([[0.4, 0.6]])
    together, _, _ = _credit([[0.4, 0.6], [0.9, 0.1], [0.5, 0.7]])
    assert first.item() == together[0].item()
    assert together[2] > together[0]


def test_objective_permutation_symmetry():
    q1, _, _ = _credit([[0.2, 0.8, 0.5]])
    q2, _, _ = _credit([[0.5, 0.2, 0.8]])
    assert torch.allclose(q1, q2)


def test_leave_one_out_advantage_and_past_scale():
    rewards = torch.tensor([[1.0], [2.0], [4.0], [10.0], [13.0]])
    mask = torch.ones_like(rewards)
    uid = np.array(["a", "a", "a", "b", "b"], dtype=object)
    adv, returns, batch_std = compute_hvpo_outcome_advantage(rewards, mask, uid, scale=2.0)
    expected_diff = torch.tensor([-2.0, -0.5, 2.5, -3.0, 3.0])
    assert torch.allclose(adv[:, 0], expected_diff / (2.0 + 1e-6), atol=1e-6)
    assert torch.equal(adv, returns)
    assert batch_std == pytest.approx(float(expected_diff.std(unbiased=False)))


def test_leave_one_out_rejects_singleton_group():
    with pytest.raises(ValueError, match="at least two"):
        compute_hvpo_outcome_advantage(
            torch.tensor([[1.0]]), torch.tensor([[1.0]]), np.array(["only"], dtype=object), scale=1.0
        )


def test_manager_requires_frozen_anchors():
    scores = {"a": lambda **_: 0.0, "b": lambda **_: 0.0}
    with pytest.raises(ValueError, match="frozen"):
        AmoHvpoRewardManager(None, 0, scores, hv_config={})
    manager = AmoHvpoRewardManager(
        None, 0, scores,
        hv_config={"calib_lower": [0, 0], "calib_upper": [1, 1], "reference_point": [0, 0]},
    )
    assert torch.equal(manager.calib_lower, torch.zeros(2))

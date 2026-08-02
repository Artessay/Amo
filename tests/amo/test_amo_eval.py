"""Regression tests for offline multi-objective evaluation metrics."""

import math

import pytest

from verl.trainer.amo_eval import compute_root_hypervolume


def test_root_hypervolume_uses_objective_dimension():
    assert compute_root_hypervolume(12.0, 2) == pytest.approx(math.sqrt(12.0))
    assert compute_root_hypervolume(8.0, 3) == pytest.approx(2.0)
    assert compute_root_hypervolume(0.0, 4) == 0.0


def test_root_hypervolume_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least 1"):
        compute_root_hypervolume(1.0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_root_hypervolume(-1.0, 2)
"""Regression tests for offline multi-objective evaluation metrics."""

import math

import pytest

from verl.trainer.amo_eval import (
    calibrate_objective_vectors,
    compute_mean_rooted_singleton_hypervolume,
    compute_root_hypervolume,
    load_metric_calibration,
)


def test_root_hypervolume_uses_objective_dimension():
    assert compute_root_hypervolume(12.0, 2) == pytest.approx(math.sqrt(12.0))
    assert compute_root_hypervolume(8.0, 3) == pytest.approx(2.0)
    assert compute_root_hypervolume(0.0, 4) == 0.0


def test_root_hypervolume_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least 1"):
        compute_root_hypervolume(1.0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_root_hypervolume(-1.0, 2)


def test_mean_rooted_singleton_hypervolume_roots_before_averaging():
    vectors = [
        [1.0, 1.0],
        [0.25, 0.25],
    ]

    # Per-response rooted HV values are 1.0 and 0.25, respectively.
    assert compute_mean_rooted_singleton_hypervolume(vectors) == pytest.approx(0.625)


def test_mean_rooted_singleton_hypervolume_respects_reference_boundary():
    vectors = [
        [2.0, 3.0],
        [0.5, 100.0],
        [1.0, 5.0],
    ]

    # Only the first response is strictly above rho=(1, 1); its H is sqrt(2).
    result = compute_mean_rooted_singleton_hypervolume(vectors, ref_point=[1.0, 1.0])
    assert result == pytest.approx(math.sqrt(2.0) / 3.0)


def test_frozen_affine_calibration_precedes_rooted_hv():
    raw_vectors = [
        [0.0, 10.0],
        [5.0, 15.0],
    ]
    calibrated = calibrate_objective_vectors(
        raw_vectors,
        calib_lower=[0.0, 10.0],
        calib_upper=[10.0, 20.0],
    )

    assert calibrated.tolist() == [[0.0, 0.0], [0.5, 0.5]]
    assert compute_mean_rooted_singleton_hypervolume(calibrated) == pytest.approx(0.25)


def test_rooted_hv_rejects_invalid_vector_shapes():
    with pytest.raises(ValueError, match="non-empty 2D"):
        compute_mean_rooted_singleton_hypervolume([])
    with pytest.raises(ValueError, match="dimension"):
        compute_mean_rooted_singleton_hypervolume([[1.0, 1.0]], ref_point=[0.0])


def test_metric_calibration_file_converts_raw_reference(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        """{
            "calib_lower": [-4.0, -2.0],
            "calib_upper": [4.0, 6.0],
            "hv_reference": [-4.0, -2.0]
        }"""
    )

    lower, upper, reference = load_metric_calibration(path)
    assert lower == [-4.0, -2.0]
    assert upper == [4.0, 6.0]
    assert reference == [0.0, 0.0]

#!/usr/bin/env python3
"""Diagnose HVPO credit assignment from saved ParaDetox training rollouts.

The diagnostic is deliberately offline: it only reads numeric-step JSONL files
written by ``trainer.rollout_data_dir``.  It reconstructs the pilot's running
min/max objective geometry, uses the repository hypervolume implementation, and
compares the resulting geometric signal with the rewards that were actually
logged and standardized by GRPO/HVPO.

Example::

    python scripts/paradetox/diagnose_hvpo_rollouts.py \
      --rollout-dir results/ParaDetox/run/qwen_hvpo/rollouts \
      --max-step 200 --rollouts-per-prompt 8 \
      --output results/ParaDetox/run/hvpo_mechanism.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import torch

from verl.workers.reward_manager.amo_utils.hybrid_reward import HybridRewardModel
from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator


OBJECTIVES = ("detox_sta", "detox_sim", "detox_fluency")
LOGGED_FIELDS = ("hybrid_rewards", "hv_contribution", "distance_penalty")
GRPO_EPSILON = 1e-6
ZERO_TOLERANCE = 1e-12


def _step_files(directory: Path, max_step: int) -> dict[int, Path]:
    if max_step < 1:
        raise ValueError("max_step must be positive")
    if not directory.is_dir():
        raise FileNotFoundError(f"Rollout directory does not exist: {directory}")
    files = {
        int(path.stem): path
        for path in directory.glob("*.jsonl")
        if path.stem.isdigit() and 0 < int(path.stem) <= max_step
    }
    if not files:
        raise FileNotFoundError(f"No numeric rollout JSONL files through step {max_step} in {directory}")
    return dict(sorted(files.items()))


def _prompt_key(row: dict) -> tuple[str, str]:
    return str(row.get("input", "")), json.dumps(row.get("gts"), sort_keys=True, ensure_ascii=False)


def _distribution(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    if not np.all(np.isfinite(array)):
        raise ValueError("Diagnostic encountered a non-finite numeric value")
    quantiles = np.quantile(array, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(array.max()),
    }


def _front_mask(points: np.ndarray) -> np.ndarray:
    """Return the maximization Pareto-front mask (duplicates all survive)."""
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        for j, other in enumerate(points):
            if i != j and np.all(other >= point) and np.any(other > point):
                keep[i] = False
                break
    return keep


def _group_standardize(values: np.ndarray) -> np.ndarray:
    """Mirror ``compute_grpo_outcome_advantage`` for one rollout group."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return values.copy()  # GRPO's singleton fallback is mean=0, std=1.
    std = values.std(ddof=1)
    return (values - values.mean()) / (std + GRPO_EPSILON)


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or first.std() <= ZERO_TOLERANCE or second.std() <= ZERO_TOLERANCE:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= ZERO_TOLERANCE:
        return None
    return float(np.dot(first, second) / denominator)


def _sign(values: np.ndarray, tolerance: float = ZERO_TOLERANCE) -> np.ndarray:
    return np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))


def _alignment(first: np.ndarray, second: np.ndarray) -> dict:
    """Alignment of two already group-standardized reward/advantage vectors."""
    sign_first = _sign(first)
    sign_second = _sign(second)
    both_active = (sign_first != 0) & (sign_second != 0)
    opposition = sign_first * sign_second == -1
    return {
        "num_rollouts": int(len(first)),
        "pearson": _pearson(first, second),
        "cosine": _cosine(first, second),
        "sign_class_disagreement_fraction": float(np.mean(sign_first != sign_second)),
        "opposite_sign_fraction": float(np.mean(opposition)),
        "opposite_sign_fraction_among_both_nonzero": (
            float(np.mean(opposition[both_active])) if np.any(both_active) else None
        ),
        "one_zero_other_nonzero_fraction": float(np.mean((sign_first == 0) != (sign_second == 0))),
    }


def _repo_hv(points: np.ndarray, reference: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    point_tensor = torch.as_tensor(points, dtype=dtype)
    reference_tensor = torch.as_tensor(reference, dtype=dtype)
    return HypervolumeCalculator.calculate_hypervolume(point_tensor, reference_tensor)


def _repo_exclusive_hv(points: np.ndarray, reference: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    """Full-set exclusive contributions using repository HV arithmetic."""
    full = _repo_hv(points, reference, dtype)
    contributions = []
    for index in range(len(points)):
        without = np.delete(points, index, axis=0)
        reduced = _repo_hv(without, reference, dtype)
        # Subtract tensors, rather than their Python values, to retain float32
        # cancellation/rounding exactly as the training reward implementation.
        contributions.append(float((full - reduced).item()))
    return np.asarray(contributions, dtype=np.float64)


def _load_groups(
    directory: Path,
    max_step: int,
    rollouts_per_prompt: int,
    objectives: tuple[str, ...],
    normalization: str,
    reference_point: np.ndarray,
    distance_metric: str,
) -> tuple[list[dict], dict[int, list[dict]], list[int], list[int]]:
    if rollouts_per_prompt < 2:
        raise ValueError("rollouts_per_prompt must be at least 2 for group-relative diagnostics")
    files = _step_files(directory, max_step)
    steps = list(files)
    missing_steps = sorted(set(range(1, max(steps) + 1)) - set(steps))
    if normalization == "cumulative-minmax" and missing_steps:
        preview = ", ".join(str(step) for step in missing_steps[:10])
        suffix = " ..." if len(missing_steps) > 10 else ""
        raise ValueError(
            "cumulative-minmax reconstruction requires contiguous rollout files "
            f"from step 1 through {max(steps)}; missing steps: {preview}{suffix}"
        )

    running_min: torch.Tensor | None = None
    running_max: torch.Tensor | None = None
    records: list[dict] = []
    by_step: dict[int, list[dict]] = {}

    for step, path in files.items():
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"Empty rollout file: {path}")
        required = (*objectives, *LOGGED_FIELDS)
        missing = sorted({key for key in required if any(key not in row for row in rows)})
        if missing:
            raise KeyError(f"Missing fields in {path}: {missing}")
        if any(int(row.get("step", step)) != step for row in rows):
            raise ValueError(f"Rows in {path} contain a different step")

        raw = torch.tensor(
            [[float(row[name]) for name in objectives] for row in rows], dtype=torch.float32
        )
        if not bool(torch.isfinite(raw).all()):
            raise ValueError(f"Non-finite objective value in {path}")
        if normalization == "cumulative-minmax":
            batch_min, batch_max = raw.min(dim=0).values, raw.max(dim=0).values
            running_min = batch_min if running_min is None else torch.minimum(running_min, batch_min)
            running_max = batch_max if running_max is None else torch.maximum(running_max, batch_max)
            geometry = ((raw - running_min) / (running_max - running_min).clamp_min(1e-8)).clamp(0.0, 1.0)
        elif normalization == "per-step-minmax":
            batch_min, batch_max = raw.min(dim=0).values, raw.max(dim=0).values
            geometry = ((raw - batch_min) / (batch_max - batch_min).clamp_min(1e-8)).clamp(0.0, 1.0)
        elif normalization == "none":
            geometry = raw
        else:  # protected by argparse, retained for library callers
            raise ValueError(f"Unknown normalization: {normalization}")

        # The current static-reference manager computes one clamped reference
        # from the whole reward batch, then reuses it for every uid group.
        batch_reference = torch.minimum(
            torch.as_tensor(reference_point, dtype=torch.float32), geometry.min(dim=0).values
        ).numpy()

        grouped: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(_prompt_key(row), []).append(index)
        wrong_sizes = [len(indices) for indices in grouped.values() if len(indices) != rollouts_per_prompt]
        if wrong_sizes:
            raise ValueError(
                f"Expected {rollouts_per_prompt} rollouts/prompt in {path}; found {wrong_sizes[0]}. "
                "The dump omits uid, so duplicate input/reference rows cannot be disambiguated."
            )

        step_records = []
        for group_index, (key, indices) in enumerate(grouped.items()):
            raw_group = raw[indices].numpy().astype(np.float64)
            geometry_group = geometry[indices].numpy().astype(np.float64)
            current = np.asarray([float(rows[i]["hybrid_rewards"]) for i in indices])
            logged_hv = np.asarray([float(rows[i]["hv_contribution"]) for i in indices])
            logged_distance = np.asarray([float(rows[i]["distance_penalty"]) for i in indices])

            exclusive32 = _repo_exclusive_hv(geometry_group, batch_reference, torch.float32)
            exclusive64 = _repo_exclusive_hv(geometry_group, batch_reference, torch.float64)
            recomputed = HybridRewardModel.compute_group_hybrid_rewards(
                torch.as_tensor(geometry_group, dtype=torch.float32),
                torch.zeros((0, len(objectives)), dtype=torch.float32),
                torch.as_tensor(batch_reference, dtype=torch.float32),
                distance_metric=distance_metric,
            ).numpy().astype(np.float64)

            record = {
                "step": step,
                "group_index": group_index,
                "prompt_sha1": hashlib.sha1((key[0] + "\0" + key[1]).encode()).hexdigest()[:12],
                "raw": raw_group,
                "geometry": geometry_group,
                "reference": batch_reference.astype(np.float64),
                "equal_linear": raw_group.mean(axis=1),
                "current": current,
                "logged_hv": logged_hv,
                "logged_distance": logged_distance,
                "exclusive32": exclusive32,
                "exclusive64": exclusive64,
                "recomputed_current": recomputed,
            }
            records.append(record)
            step_records.append(record)
        by_step[step] = step_records

    return records, by_step, steps, missing_steps


def _objective_saturation(records: list[dict], objectives: tuple[str, ...]) -> dict:
    points = np.concatenate([record["raw"] for record in records], axis=0)
    result = {}
    for column, name in enumerate(objectives):
        values = points[:, column]
        group_ranges = np.asarray([np.ptp(record["raw"][:, column]) for record in records])
        group_stds = np.asarray([record["raw"][:, column].std() for record in records])
        result[name] = {
            "distribution": _distribution(values),
            "fraction_near_lower_0p01": float(np.mean(values <= 0.01)),
            "fraction_near_upper_0p99": float(np.mean(values >= 0.99)),
            "fraction_at_lower_1e_6": float(np.mean(values <= 1e-6)),
            "fraction_at_upper_1_minus_1e_6": float(np.mean(values >= 1.0 - 1e-6)),
            "within_group_range": _distribution(group_ranges),
            "within_group_std": _distribution(group_stds),
            "fraction_groups_range_le_1e_4": float(np.mean(group_ranges <= 1e-4)),
        }
    return {
        "nominal_objective_bounds": [0.0, 1.0],
        "near_boundary_margin": 0.01,
        "objectives": result,
    }


def _front_metrics(records: list[dict]) -> dict:
    unique_counts, rollout_counts, duplicate_rates, positive_credit_counts = [], [], [], []
    for record in records:
        points = record["raw"]
        unique = np.unique(points, axis=0)
        unique_counts.append(int(_front_mask(unique).sum()))
        rollout_counts.append(int(_front_mask(points).sum()))
        duplicate_rates.append(1.0 - len(unique) / len(points))
        positive_credit_counts.append(int(np.count_nonzero(record["logged_hv"] > ZERO_TOLERANCE)))
    size = len(records[0]["raw"])
    return {
        "definition": "Maximization Pareto front; unique-front size removes exact objective duplicates.",
        "unique_nondominated_count": _distribution(unique_counts),
        "nondominated_rollout_count_including_duplicates": _distribution(rollout_counts),
        "logged_positive_exclusive_credit_count": _distribution(positive_credit_counts),
        "mean_duplicate_objective_point_fraction": float(np.mean(duplicate_rates)),
        "single_unique_front_point_fraction": float(np.mean(np.asarray(unique_counts) == 1)),
        "all_unique_points_nondominated_fraction": float(
            np.mean([count == len(np.unique(r["raw"], axis=0)) for count, r in zip(unique_counts, records)])
        ),
        "mean_unique_front_fraction_of_rollout_group": float(np.mean(unique_counts) / size),
    }


def _logged_signal(records: list[dict]) -> dict:
    current = np.concatenate([record["current"] for record in records])
    hv = np.concatenate([record["logged_hv"] for record in records])
    distance = np.concatenate([record["logged_distance"] for record in records])
    linear = np.concatenate([record["equal_linear"] for record in records])
    current_group_std = np.asarray([record["current"].std(ddof=1) for record in records])
    linear_group_std = np.asarray([record["equal_linear"].std(ddof=1) for record in records])
    positive_mean = float(hv[hv > ZERO_TOLERANCE].mean()) if np.any(hv > ZERO_TOLERANCE) else 0.0
    fallback_abs_mean = (
        float(np.abs(distance[distance < -ZERO_TOLERANCE]).mean())
        if np.any(distance < -ZERO_TOLERANCE)
        else 0.0
    )
    return {
        "current_hybrid_reward": _distribution(current),
        "positive_hv_contribution_field": _distribution(hv),
        "negative_distance_fallback_field": _distribution(distance),
        "equal_linear_reward": _distribution(linear),
        "positive_hv_rollout_fraction": float(np.mean(hv > ZERO_TOLERANCE)),
        "nonzero_distance_fallback_fraction": float(np.mean(distance < -ZERO_TOLERANCE)),
        "zero_hybrid_reward_fraction": float(np.mean(np.abs(current) <= ZERO_TOLERANCE)),
        "fallback_abs_mean_over_positive_hv_mean": (
            fallback_abs_mean / positive_mean if positive_mean > 0 else None
        ),
        "hybrid_std_over_equal_linear_std": (
            float(current.std() / linear.std()) if linear.std() > 0 else None
        ),
        "within_group_current_sample_std": _distribution(current_group_std),
        "within_group_equal_linear_sample_std": _distribution(linear_group_std),
        "fraction_groups_current_std_le_grpo_epsilon": float(
            np.mean(current_group_std <= GRPO_EPSILON)
        ),
        "max_abs_logged_decomposition_error": float(np.max(np.abs(current - hv - distance))),
    }


def _alignment_metrics(records: list[dict], objectives: tuple[str, ...]) -> dict:
    current_chunks: list[np.ndarray] = []
    channels: dict[str, list[np.ndarray]] = {
        "equal_linear": [],
        "logged_hv_only": [],
        "recomputed_exclusive_hv_float32": [],
        "recomputed_exclusive_hv_float64": [],
        "recomputed_current_hybrid_float32": [],
        **{name: [] for name in objectives},
    }
    top_agreements = Counter()
    for record in records:
        current = _group_standardize(record["current"])
        current_chunks.append(current)
        raw_channels = {
            "equal_linear": record["equal_linear"],
            "logged_hv_only": record["logged_hv"],
            "recomputed_exclusive_hv_float32": record["exclusive32"],
            "recomputed_exclusive_hv_float64": record["exclusive64"],
            "recomputed_current_hybrid_float32": record["recomputed_current"],
            **{name: record["raw"][:, i] for i, name in enumerate(objectives)},
        }
        current_top = int(np.argmax(record["current"]))
        for name, values in raw_channels.items():
            channels[name].append(_group_standardize(values))
            top_agreements[name] += int(current_top == int(np.argmax(values)))

    current_all = np.concatenate(current_chunks)
    comparisons = {}
    for name, chunks in channels.items():
        metrics = _alignment(current_all, np.concatenate(chunks))
        metrics["top_rollout_argmax_agreement_fraction"] = top_agreements[name] / len(records)
        comparisons[name] = metrics
    return {
        "definition": (
            "Each scalar channel is centered and divided by its prompt-group sample std + 1e-6, "
            "matching GRPO/HVPO advantage normalization; current_hybrid_reward is the reference channel."
        ),
        "num_prompt_groups": len(records),
        "comparisons_to_current_hybrid_reward": comparisons,
    }


def _float_precision_metrics(records: list[dict]) -> dict:
    float32 = np.concatenate([record["exclusive32"] for record in records])
    float64 = np.concatenate([record["exclusive64"] for record in records])
    absolute = np.abs(float32 - float64)
    relative = absolute / np.maximum(np.abs(float64), 1e-15)
    strict32, strict64 = float32 > 0.0, float64 > 0.0
    material32, material64 = _sign(float32), _sign(float64)
    max_flat = int(np.argmax(absolute))
    group_size = len(records[0]["exclusive32"])
    record = records[max_flat // group_size]
    member = max_flat % group_size
    return {
        "definition": (
            "Same reconstructed normalized coordinates and reference point are cast to float32/float64; "
            "exclusive HV is computed with the repository HypervolumeCalculator in both dtypes."
        ),
        "absolute_difference": _distribution(absolute),
        "relative_difference_with_1e_15_floor": _distribution(relative),
        "strict_positive_classification_disagreement_fraction": float(np.mean(strict32 != strict64)),
        "material_sign_class_disagreement_fraction_at_1e_12": float(
            np.mean(material32 != material64)
        ),
        "float32_positive_float64_nonpositive_count": int(np.count_nonzero(strict32 & ~strict64)),
        "float64_positive_float32_nonpositive_count": int(np.count_nonzero(strict64 & ~strict32)),
        "float32_negative_count": int(np.count_nonzero(float32 < 0.0)),
        "float64_negative_count": int(np.count_nonzero(float64 < 0.0)),
        "max_abs_difference_example": {
            "step": int(record["step"]),
            "group_index": int(record["group_index"]),
            "prompt_sha1": record["prompt_sha1"],
            "member_index": member,
            "float32": float(float32[max_flat]),
            "float64": float(float64[max_flat]),
            "absolute_difference": float(absolute[max_flat]),
        },
    }


def _logged_reproduction(records: list[dict]) -> dict:
    logged = np.concatenate([record["current"] for record in records])
    recomputed = np.concatenate([record["recomputed_current"] for record in records])
    error = np.abs(logged - recomputed)
    return {
        "assumptions": (
            "intra-group empty Pareto front, static reference, no contribution scaling, and the selected "
            "normalization/distance metric"
        ),
        "absolute_error": _distribution(error),
        "fraction_with_abs_error_le_1e_6": float(np.mean(error <= 1e-6)),
        "max_abs_logged_reward": float(np.max(np.abs(logged))),
    }


def _expected_marginal_credit(
    records: list[dict], max_exact_subsets: int, monte_carlo_subsets: int, seed: int
) -> dict:
    """Estimate E[HV(S)-HV(S\\{i}) | |S|=k, i in S] for every k."""
    if max_exact_subsets < 1 or monte_carlo_subsets < 1:
        raise ValueError("max_exact_subsets and monte_carlo_subsets must be positive")
    rng = np.random.default_rng(seed)
    group_size = len(records[0]["geometry"])
    accumulators = {
        k: {
            "marginal_sum": 0.0,
            "marginal_sum_sq": 0.0,
            "marginal_count": 0,
            "positive_count": 0,
            "total_credit_sum": 0.0,
            "set_hv_sum": 0.0,
            "set_count": 0,
            "unique_set_count": 0,
            "possible_set_count": 0,
            "modes": Counter(),
            # Retained only in memory. JSON receives pooled summaries below,
            # never per-rollout values.
            "expected_member_chunks": [],
            "standardized_marginal_chunks": [],
            "standardized_current_chunks": [],
            "standardized_linear_chunks": [],
            "current_top_agreement_count": 0,
            "linear_top_agreement_count": 0,
        }
        for k in range(1, group_size + 1)
    }

    for record in records:
        points, reference = record["geometry"], record["reference"]
        hv_cache: dict[int, float] = {0: 0.0}

        def hv(mask: int) -> float:
            if mask not in hv_cache:
                indices = [i for i in range(group_size) if mask & (1 << i)]
                hv_cache[mask] = float(_repo_hv(points[indices], reference, torch.float64).item())
            return hv_cache[mask]

        for k in range(1, group_size + 1):
            possible = math.comb(group_size, k)
            if possible <= max_exact_subsets:
                subsets = list(itertools.combinations(range(group_size), k))
                mode = "exact"
            else:
                subsets = [
                    tuple(sorted(rng.choice(group_size, size=k, replace=False).tolist()))
                    for _ in range(monte_carlo_subsets)
                ]
                mode = "seeded_monte_carlo"
            accumulator = accumulators[k]
            masks = []
            member_sum = np.zeros(group_size, dtype=np.float64)
            member_count = np.zeros(group_size, dtype=np.int64)
            for subset in subsets:
                mask = sum(1 << index for index in subset)
                masks.append(mask)
                set_hv = hv(mask)
                marginals = np.asarray([set_hv - hv(mask ^ (1 << index)) for index in subset])
                for local_index, member_index in enumerate(subset):
                    member_sum[member_index] += marginals[local_index]
                    member_count[member_index] += 1
                accumulator["marginal_sum"] += float(marginals.sum())
                accumulator["marginal_sum_sq"] += float(np.square(marginals).sum())
                accumulator["marginal_count"] += len(marginals)
                accumulator["positive_count"] += int(np.count_nonzero(marginals > ZERO_TOLERANCE))
                accumulator["total_credit_sum"] += float(marginals.sum())
                accumulator["set_hv_sum"] += set_hv
                accumulator["set_count"] += 1
            accumulator["unique_set_count"] += len(set(masks))
            accumulator["possible_set_count"] += possible
            accumulator["modes"][mode] += 1

            # A tiny Monte Carlo budget can omit a member. Add one conditional
            # draw for each missing rollout; it does not enter set-level estimates.
            for member_index in np.flatnonzero(member_count == 0):
                other_indices = [i for i in range(group_size) if i != member_index]
                sampled_others = (
                    rng.choice(other_indices, size=k - 1, replace=False).tolist()
                    if k > 1
                    else []
                )
                conditional_subset = [int(member_index), *sampled_others]
                conditional_mask = sum(1 << index for index in conditional_subset)
                marginal = (
                    hv(conditional_mask) - hv(conditional_mask ^ (1 << int(member_index)))
                )
                member_sum[member_index] += marginal
                member_count[member_index] += 1

            expected_member = member_sum / member_count
            standardized_marginal = _group_standardize(expected_member)
            standardized_current = _group_standardize(record["current"])
            standardized_linear = _group_standardize(record["equal_linear"])
            accumulator["expected_member_chunks"].append(expected_member)
            accumulator["standardized_marginal_chunks"].append(standardized_marginal)
            accumulator["standardized_current_chunks"].append(standardized_current)
            accumulator["standardized_linear_chunks"].append(standardized_linear)
            marginal_top = int(np.argmax(expected_member))
            accumulator["current_top_agreement_count"] += int(
                marginal_top == int(np.argmax(record["current"]))
            )
            accumulator["linear_top_agreement_count"] += int(
                marginal_top == int(np.argmax(record["equal_linear"]))
            )

    by_k = {}
    for k, accumulator in accumulators.items():
        count = accumulator["marginal_count"]
        mean = accumulator["marginal_sum"] / count
        variance = max(accumulator["marginal_sum_sq"] / count - mean * mean, 0.0)
        expected_set_hv = accumulator["set_hv_sum"] / accumulator["set_count"]
        expected_total = accumulator["total_credit_sum"] / accumulator["set_count"]
        expected_members = np.concatenate(accumulator["expected_member_chunks"])
        standardized_marginal = np.concatenate(accumulator["standardized_marginal_chunks"])
        standardized_current = np.concatenate(accumulator["standardized_current_chunks"])
        standardized_linear = np.concatenate(accumulator["standardized_linear_chunks"])
        current_alignment = _alignment(standardized_marginal, standardized_current)
        linear_alignment = _alignment(standardized_marginal, standardized_linear)
        current_alignment["top_rollout_argmax_agreement_fraction"] = (
            accumulator["current_top_agreement_count"] / len(records)
        )
        linear_alignment["top_rollout_argmax_agreement_fraction"] = (
            accumulator["linear_top_agreement_count"] / len(records)
        )
        by_k[str(k)] = {
            "expected_per_member_exclusive_hv": mean,
            "std_per_member_exclusive_hv": math.sqrt(variance),
            "positive_marginal_fraction": accumulator["positive_count"] / count,
            "expected_total_exclusive_credit_per_set": expected_total,
            "expected_set_hypervolume": expected_set_hv,
            "exclusive_credit_sum_over_set_hv": (
                expected_total / expected_set_hv if expected_set_hv > 0 else None
            ),
            "sets_evaluated": accumulator["set_count"],
            "unique_sets_evaluated": accumulator["unique_set_count"],
            "possible_sets_across_groups": accumulator["possible_set_count"],
            "estimation_mode_group_counts": dict(accumulator["modes"]),
            "per_rollout_conditional_expected_credit": _distribution(expected_members),
            "group_standardized_per_rollout_alignment": {
                "to_current_hybrid_reward": current_alignment,
                "to_equal_linear": linear_alignment,
            },
        }
    k1 = by_k["1"]["expected_per_member_exclusive_hv"]
    for metrics in by_k.values():
        metrics["mean_credit_relative_to_k1"] = (
            metrics["expected_per_member_exclusive_hv"] / k1 if abs(k1) > 1e-15 else None
        )
    return {
        "definition": (
            "E[HV(S)-HV(S without i) | |S|=k, i uniformly incident to S]. Each rollout's "
            "conditional expectation is standardized within its prompt group exactly like GRPO, then "
            "pooled for alignment; only summary metrics are serialized."
        ),
        "dtype": "float64 via repository HypervolumeCalculator",
        "seed": seed,
        "max_exact_subsets_per_group_and_k": max_exact_subsets,
        "monte_carlo_subsets_per_group_and_k": monte_carlo_subsets,
        "by_set_size_k": by_k,
    }


def _compact_step_metrics(records: list[dict], objectives: tuple[str, ...]) -> dict:
    saturation = _objective_saturation(records, objectives)["objectives"]
    front = _front_metrics(records)
    signal = _logged_signal(records)
    alignment = _alignment_metrics(records, objectives)["comparisons_to_current_hybrid_reward"][
        "equal_linear"
    ]
    return {
        "num_prompt_groups": len(records),
        "num_rollouts": sum(len(record["raw"]) for record in records),
        "objective_mean": {name: saturation[name]["distribution"]["mean"] for name in objectives},
        "objective_near_upper_fraction": {
            name: saturation[name]["fraction_near_upper_0p99"] for name in objectives
        },
        "mean_unique_nondominated_count": front["unique_nondominated_count"]["mean"],
        "mean_logged_positive_credit_count": front["logged_positive_exclusive_credit_count"]["mean"],
        "positive_hv_rollout_fraction": signal["positive_hv_rollout_fraction"],
        "nonzero_distance_fallback_fraction": signal["nonzero_distance_fallback_fraction"],
        "current_reward_std": signal["current_hybrid_reward"]["std"],
        "current_vs_equal_linear_after_group_standardization": alignment,
    }


def analyze_rollouts(
    rollout_dir: Path,
    max_step: int,
    rollouts_per_prompt: int,
    *,
    objectives: tuple[str, ...] = OBJECTIVES,
    normalization: str = "cumulative-minmax",
    reference_point: Iterable[float] | None = None,
    distance_metric: str = "chebyshev",
    max_exact_subsets: int = 4096,
    monte_carlo_subsets: int = 2048,
    seed: int = 42,
) -> dict:
    if not objectives:
        raise ValueError("At least one objective is required")
    reference = np.asarray(
        list(reference_point) if reference_point is not None else [0.0] * len(objectives),
        dtype=np.float64,
    )
    if reference.shape != (len(objectives),):
        raise ValueError(f"reference_point must contain {len(objectives)} values")
    if distance_metric not in {"chebyshev", "euclidean", "none"}:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    records, by_step, steps, missing_steps = _load_groups(
        rollout_dir,
        max_step,
        rollouts_per_prompt,
        objectives,
        normalization,
        reference,
        distance_metric,
    )
    return {
        "schema_version": 1,
        "input": {
            "rollout_dir": str(rollout_dir.resolve()),
            "requested_max_step": max_step,
            "last_analyzed_step": max(steps),
            "steps": steps,
            "missing_steps_before_last": missing_steps,
            "rollouts_per_prompt": rollouts_per_prompt,
            "num_prompt_groups": len(records),
            "num_rollouts": sum(len(record["raw"]) for record in records),
            "objectives": list(objectives),
            "geometry_normalization": normalization,
            "static_reference_point": reference.tolist(),
            "distance_metric": distance_metric,
        },
        "objective_saturation": _objective_saturation(records, objectives),
        "pareto_front_size": _front_metrics(records),
        "logged_hv_and_fallback_scales": _logged_signal(records),
        "expected_marginal_credit_by_set_size": _expected_marginal_credit(
            records, max_exact_subsets, monte_carlo_subsets, seed
        ),
        "group_standardized_reward_alignment": _alignment_metrics(records, objectives),
        "float32_vs_float64_exclusive_hv": _float_precision_metrics(records),
        "logged_reward_reproduction": _logged_reproduction(records),
        "per_step": {
            str(step): _compact_step_metrics(step_records, objectives)
            for step, step_records in by_step.items()
        },
    }


def _print_summary(result: dict, output: Path) -> None:
    info = result["input"]
    saturation = result["objective_saturation"]["objectives"]
    front = result["pareto_front_size"]
    signal = result["logged_hv_and_fallback_scales"]
    alignment = result["group_standardized_reward_alignment"][
        "comparisons_to_current_hybrid_reward"
    ]["equal_linear"]
    precision = result["float32_vs_float64_exclusive_hv"]
    marginal = result["expected_marginal_credit_by_set_size"]["by_set_size_k"]
    last_k = str(info["rollouts_per_prompt"])

    upper = ", ".join(
        f"{name}={metrics['fraction_near_upper_0p99']:.1%}" for name, metrics in saturation.items()
    )
    print(
        f"HVPO diagnostic: steps {info['steps'][0]}..{info['last_analyzed_step']}, "
        f"{info['num_prompt_groups']} groups / {info['num_rollouts']} rollouts"
    )
    print(f"  objective >=0.99: {upper}")
    print(
        f"  unique Pareto front/group: {front['unique_nondominated_count']['mean']:.2f}; "
        f"positive HV={signal['positive_hv_rollout_fraction']:.1%}, "
        f"fallback={signal['nonzero_distance_fallback_fraction']:.1%}"
    )
    print(
        f"  standardized current vs equal-linear: r={alignment['pearson']}, "
        f"cos={alignment['cosine']}, opposite-sign={alignment['opposite_sign_fraction']:.1%}"
    )
    print(
        f"  expected exclusive credit k=1 -> k={last_k}: "
        f"{marginal['1']['expected_per_member_exclusive_hv']:.6g} -> "
        f"{marginal[last_k]['expected_per_member_exclusive_hv']:.6g}"
    )
    print(
        f"  float32/64 strict positive disagreement: "
        f"{precision['strict_positive_classification_disagreement_fraction']:.2%}; JSON: {output}"
    )


def _write_synthetic_step(path: Path, step: int, groups: list[np.ndarray]) -> None:
    prepared = []
    for group_index, points in enumerate(groups):
        reward = HybridRewardModel.compute_group_hybrid_rewards(
            torch.tensor(points, dtype=torch.float32),
            torch.zeros((0, points.shape[1]), dtype=torch.float32),
            torch.zeros(points.shape[1], dtype=torch.float32),
        ).numpy()
        rows = []
        for member, (point, value) in enumerate(zip(points, reward)):
            rows.append(
                {
                    "input": f"prompt {group_index}",
                    "gts": f"reference {group_index}",
                    "output": f"answer {member}",
                    "step": step,
                    **dict(zip(OBJECTIVES, point.tolist())),
                    "hybrid_rewards": float(value),
                    "hv_contribution": float(max(value, 0.0)),
                    "distance_penalty": float(min(value, 0.0)),
                }
            )
        prepared.append(rows)
    # Deliberately interleave groups: real rollout dumps are not contiguous by uid.
    ordered = [prepared[group][member] for member in range(len(prepared[0])) for group in range(len(prepared))]
    path.write_text("".join(json.dumps(row) + "\n" for row in ordered), encoding="utf-8")


def _self_test() -> None:
    groups = [
        np.asarray([[1.0, 0.20, 0.80], [0.20, 0.95, 0.80], [0.45, 0.45, 0.80]]),
        np.asarray([[0.80, 0.80, 0.80], [0.80, 0.80, 0.80], [0.10, 0.10, 0.10]]),
    ]
    with TemporaryDirectory() as temporary:
        rollout_dir = Path(temporary) / "rollouts"
        rollout_dir.mkdir()
        _write_synthetic_step(rollout_dir / "1.jsonl", 1, groups)
        _write_synthetic_step(rollout_dir / "2.jsonl", 2, groups)
        result = analyze_rollouts(
            rollout_dir,
            max_step=2,
            rollouts_per_prompt=3,
            normalization="none",
            max_exact_subsets=8,
            monte_carlo_subsets=7,
            seed=9,
        )
        assert result["input"]["num_prompt_groups"] == 4
        assert result["objective_saturation"]["objectives"]["detox_sta"][
            "fraction_near_upper_0p99"
        ] > 0
        assert set(result["expected_marginal_credit_by_set_size"]["by_set_size_k"]) == {
            "1",
            "2",
            "3",
        }
        for by_k in result["expected_marginal_credit_by_set_size"]["by_set_size_k"].values():
            alignments = by_k["group_standardized_per_rollout_alignment"]
            assert set(alignments) == {"to_current_hybrid_reward", "to_equal_linear"}
            assert alignments["to_current_hybrid_reward"]["num_rollouts"] == 12
            assert 0.0 <= alignments["to_equal_linear"][
                "top_rollout_argmax_agreement_fraction"
            ] <= 1.0
        assert result["pareto_front_size"]["mean_duplicate_objective_point_fraction"] > 0
        assert result["logged_reward_reproduction"]["fraction_with_abs_error_le_1e_6"] == 1.0
        assert result["float32_vs_float64_exclusive_hv"]["absolute_difference"]["count"] == 12
        json.dumps(result, allow_nan=False)

        # Force MC for k=1,2 and verify the seed makes the estimate reproducible.
        first = analyze_rollouts(
            rollout_dir,
            2,
            3,
            normalization="none",
            max_exact_subsets=1,
            monte_carlo_subsets=5,
            seed=123,
        )["expected_marginal_credit_by_set_size"]
        second = analyze_rollouts(
            rollout_dir,
            2,
            3,
            normalization="none",
            max_exact_subsets=1,
            monte_carlo_subsets=5,
            seed=123,
        )["expected_marginal_credit_by_set_size"]
        assert first == second

        try:
            analyze_rollouts(
                rollout_dir,
                2,
                1,
                normalization="none",
            )
        except ValueError as error:
            assert "at least 2" in str(error)
        else:
            raise AssertionError("singleton rollout groups must be rejected")

        gap_dir = Path(temporary) / "rollouts_with_gap"
        gap_dir.mkdir()
        _write_synthetic_step(gap_dir / "1.jsonl", 1, groups)
        _write_synthetic_step(gap_dir / "3.jsonl", 3, groups)
        try:
            analyze_rollouts(
                gap_dir,
                3,
                3,
                normalization="cumulative-minmax",
            )
        except ValueError as error:
            assert "contiguous rollout files" in str(error)
            assert "missing steps: 2" in str(error)
        else:
            raise AssertionError("cumulative min/max reconstruction must reject missing steps")
    print("diagnose_hvpo_rollouts self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--rollouts-per-prompt", type=int)
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    parser.add_argument(
        "--normalization",
        choices=("cumulative-minmax", "per-step-minmax", "none"),
        default="cumulative-minmax",
        help="Geometry used for recomputation; pilot/current default is cumulative-minmax.",
    )
    parser.add_argument("--reference-point", nargs="+", type=float, default=None)
    parser.add_argument("--distance-metric", choices=("chebyshev", "euclidean", "none"), default="chebyshev")
    parser.add_argument("--max-exact-subsets", type=int, default=4096)
    parser.add_argument("--monte-carlo-subsets", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    missing = [
        flag
        for flag, value in (
            ("--rollout-dir", args.rollout_dir),
            ("--max-step", args.max_step),
            ("--rollouts-per-prompt", args.rollouts_per_prompt),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required unless --self-test: {', '.join(missing)}")

    result = analyze_rollouts(
        args.rollout_dir,
        args.max_step,
        args.rollouts_per_prompt,
        objectives=tuple(args.objectives),
        normalization=args.normalization,
        reference_point=args.reference_point,
        distance_metric=args.distance_metric,
        max_exact_subsets=args.max_exact_subsets,
        monte_carlo_subsets=args.monte_carlo_subsets,
        seed=args.seed,
    )
    output = args.output or (
        args.rollout_dir / f"hvpo_mechanism_diagnostic_through_step_{result['input']['last_analyzed_step']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _print_summary(result, output)


if __name__ == "__main__":
    main()

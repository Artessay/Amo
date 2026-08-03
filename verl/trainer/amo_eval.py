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
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""

from collections import defaultdict

import json
import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


def compute_hypervolume(vectors, ref_point=None):
    """Compute the dominated hypervolume of a set of objective vectors.

    This is the aggregate multi-objective quality indicator used to compare
    methods (e.g. HVPO vs. the vanilla weighted-sum baseline). A larger HV means
    the model's objective vectors jointly dominate a larger region, i.e. a better
    coverage of the objective space / Pareto front. Unlike per-objective means,
    HV rewards a *balanced* trade-off across objectives rather than sacrificing
    one objective for another.

    Args:
        vectors: list of objective vectors (each a list/tuple of floats), all of
            the same dimension. Objectives are treated as *maximization*.
        ref_point: reference point (list of floats). Defaults to all zeros, which
            is appropriate for objectives normalized to [0, 1].

    Returns:
        float hypervolume value (0.0 if input is empty).
    """
    import torch

    from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator

    if len(vectors) == 0:
        return 0.0
    dim = len(vectors[0])
    if ref_point is None:
        ref_point = [0.0] * dim
    pts = torch.tensor(vectors, dtype=torch.float64)
    ref = torch.tensor(ref_point, dtype=torch.float64)
    return float(HypervolumeCalculator.calculate_hypervolume(pts, ref).item())


def compute_root_hypervolume(hypervolume, num_objectives):
    """Return the m-th root of an already aggregated set hypervolume.

    This is a monotone rescaling of a set-level HV scalar. It is intentionally
    kept separate from :func:`compute_mean_rooted_singleton_hypervolume`, which
    is the response-level metric used by the paper.
    """
    if num_objectives < 1:
        raise ValueError("num_objectives must be at least 1")
    if hypervolume < 0:
        raise ValueError("hypervolume must be non-negative")
    return hypervolume ** (1.0 / num_objectives)


def calibrate_objective_vectors(vectors, calib_lower=None, calib_upper=None):
    """Apply an optional frozen affine calibration to objective vectors."""
    points = np.asarray(vectors, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if not np.all(np.isfinite(points)):
        raise ValueError("vectors must contain only finite values")

    if calib_lower is None and calib_upper is None:
        return points
    if calib_lower is None or calib_upper is None:
        raise ValueError("calib_lower and calib_upper must be provided together")

    lower = np.asarray(calib_lower, dtype=np.float64)
    upper = np.asarray(calib_upper, dtype=np.float64)
    expected_shape = (points.shape[1],)
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError("calibration dimension must match vectors")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("calibration bounds must contain only finite values")
    if np.any(upper <= lower):
        raise ValueError("each calib_upper value must exceed calib_lower")

    return (points - lower) / (upper - lower)


def load_metric_calibration(path):
    """Load frozen affine bounds and convert an optional raw reference point."""
    with open(path, encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)

    try:
        lower = calibration["calib_lower"]
        upper = calibration["calib_upper"]
    except KeyError as exc:
        raise ValueError("calibration file must contain calib_lower and calib_upper") from exc

    raw_reference = calibration.get("hv_reference")
    calibrated_reference = None
    if raw_reference is not None:
        calibrated_reference = calibrate_objective_vectors(
            [raw_reference],
            calib_lower=lower,
            calib_upper=upper,
        )[0].tolist()
    return lower, upper, calibrated_reference


def compute_mean_rooted_singleton_hypervolume(vectors, ref_point=None):
    """Compute mean response-level rooted singleton hypervolume.

    For every response vector ``z_i``, this computes

        H_i = prod_k(max(z_i,k - rho_k, 0)) ** (1 / m)

    and returns the arithmetic mean of ``H_i``. The root is applied before
    averaging, so this is not the root of a pooled-prompt set hypervolume.
    Input vectors must already be in the frozen calibrated coordinate system.
    """
    points = np.asarray(vectors, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    if not np.all(np.isfinite(points)):
        raise ValueError("vectors must contain only finite values")

    num_objectives = points.shape[1]
    if ref_point is None:
        reference = np.zeros(num_objectives, dtype=np.float64)
    else:
        reference = np.asarray(ref_point, dtype=np.float64)
    if reference.shape != (num_objectives,):
        raise ValueError("ref_point dimension must match vectors")
    if not np.all(np.isfinite(reference)):
        raise ValueError("ref_point must contain only finite values")

    gains = points - reference
    rooted_hv = np.zeros(points.shape[0], dtype=np.float64)
    strictly_positive = np.all(gains > 0.0, axis=1)

    # Evaluate the geometric mean in log space. A response on or below the
    # reference boundary in any coordinate has singleton HV exactly zero.
    rooted_hv[strictly_positive] = np.exp(
        np.mean(np.log(gains[strictly_positive]), axis=1)
    )
    return float(np.mean(rooted_hv))


def avg_scores_by_fn(score_lst):
    """
    Compute average scores for each reward function across all responses.

    Args:
        score_lst (list of dict): List of score dicts, one for each response.

    Returns:
        dict: Average scores for each reward function.
    """
    # Initialize result dict to store lists of scores for each reward function
    scores_by_fn = {fn_name: [] for fn_name in score_lst[0].keys()}

    # Collect all scores for each reward function
    for score_dict in score_lst:
        for fn_name, score in score_dict.items():
            scores_by_fn[fn_name].append(score)
    
    # Compute average for each reward function
    avg_scores = {fn_name: np.mean(scores) for fn_name, scores in scores_by_fn.items()}
    
    return avg_scores


@ray.remote
def process_item(config, data_source, response_lst, reward_data):
    reward_fn_dict: dict = get_custom_reward_fn(config)
    ground_truth = reward_data.get("ground_truth", "")
    extra_info = reward_data # reward data it self may contain extra info needed by reward functions
    
    assert isinstance(reward_fn_dict, dict), "reward_fn_dict must be a dict"

    def reward_item(data_source, r, ground_truth, extra_info):
        score_dict = {}
        for reward_fn_name, reward_fn in reward_fn_dict.items():
            score_dict[reward_fn_name] = reward_fn(data_source, r, ground_truth, extra_info)
        return score_dict
    
    # List of score dicts, one for each response
    score_lst = [reward_item(data_source, r, ground_truth, extra_info) for r in response_lst]

    # Preserve response-level scores for rooted singleton-HV evaluation. The
    # prompt mean is retained so existing marginal metrics and set-HV output
    # keep their original prompt-weighted semantics.
    return data_source, avg_scores_by_fn(score_lst), score_lst


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    print(local_path)
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_response_reward = defaultdict(list)
    # Create remote tasks
    remote_tasks = [
        process_item.remote(config, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score, response_scores = ray.get(result_id)
                data_source_reward[data_source].append(score)
                data_source_response_reward[data_source].append(response_scores)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        avg = avg_scores_by_fn(rewards)

        # Build per-prompt mean vectors in a stable objective order for the
        # legacy set-HV diagnostic. The formal rooted metric below uses the
        # preserved response-level vectors instead.
        fn_names = sorted(rewards[0].keys())
        vectors = [[float(r[name]) for name in fn_names] for r in rewards]

        metrics_config = config.get("metrics", {})
        calibration_path = metrics_config.get("calibration_path")
        calib_lower = metrics_config.get("calib_lower")
        calib_upper = metrics_config.get("calib_upper")
        ref_point = metrics_config.get("reference_point")

        if calibration_path is not None:
            if calib_lower is not None or calib_upper is not None:
                raise ValueError(
                    "set either metrics.calibration_path or inline calibration bounds, not both"
                )
            calib_lower, calib_upper, file_reference = load_metric_calibration(
                calibration_path
            )
            if ref_point is None:
                ref_point = file_reference
        if ref_point is None:
            ref_point = [0.0] * len(fn_names)

        calibrated_vectors = calibrate_objective_vectors(
            vectors,
            calib_lower=calib_lower,
            calib_upper=calib_upper,
        )

        # Legacy pooled-prompt set HV is retained as a diagnostic only. Taking
        # its m-th root does not turn it into a response-level metric.
        hv = compute_hypervolume(calibrated_vectors, ref_point=ref_point)
        root_set_hv = compute_root_hypervolume(hv, len(fn_names))

        # Compute H for every response before any averaging. We first average
        # within each prompt and then across prompts, matching E_x E_{y|x}[H].
        prompt_root_hvs = []
        num_responses = 0
        for response_scores in data_source_response_reward[data_source]:
            response_vectors = [
                [float(response_score[name]) for name in fn_names]
                for response_score in response_scores
            ]
            calibrated_response_vectors = calibrate_objective_vectors(
                response_vectors,
                calib_lower=calib_lower,
                calib_upper=calib_upper,
            )
            prompt_root_hvs.append(
                compute_mean_rooted_singleton_hypervolume(
                    calibrated_response_vectors,
                    ref_point=ref_point,
                )
            )
            num_responses += len(response_scores)
        mean_root_hv = float(np.mean(prompt_root_hvs))

        metric_dict[data_source] = {
            **avg,
            "mean_vector": {name: float(avg[name]) for name in fn_names},
            "hypervolume": hv,
            "root_set_hypervolume": root_set_hv,
            "root_hypervolume": mean_root_hv,
            "num_prompts": len(rewards),
            "num_responses": num_responses,
        }

    print(json.dumps(metric_dict, indent=4))

    save_path = local_path.replace(".parquet", ".json")
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4)


if __name__ == "__main__":
    main()

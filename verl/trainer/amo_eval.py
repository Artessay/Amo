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
    
    return data_source, avg_scores_by_fn(score_lst)


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
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        avg = avg_scores_by_fn(rewards)

        # Build per-prompt objective vectors (in a stable objective order) and
        # compute the dominated hypervolume as the aggregate multi-objective
        # quality indicator. Objectives are assumed to be normalized to [0, 1]
        # (as for the UniEval news metrics), so the reference point is the origin.
        fn_names = sorted(rewards[0].keys())
        vectors = [[float(r[name]) for name in fn_names] for r in rewards]
        hv = compute_hypervolume(vectors, ref_point=[0.0] * len(fn_names))

        metric_dict[data_source] = {
            **avg,
            "mean_vector": {name: float(avg[name]) for name in fn_names},
            "hypervolume": hv,
            "num_prompts": len(rewards),
        }

    print(json.dumps(metric_dict, indent=4))

    save_path = local_path.replace(".parquet", ".json")
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4)


if __name__ == "__main__":
    main()

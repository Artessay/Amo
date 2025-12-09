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
    ground_truth = reward_data["ground_truth"]
    
    assert isinstance(reward_fn_dict, dict), "reward_fn_dict must be a dict"

    def reward_item(data_source, r, ground_truth):
        score_dict = {}
        for reward_fn_name, reward_fn in reward_fn_dict.items():
            score_dict[reward_fn_name] = reward_fn(data_source, r, ground_truth)
        return score_dict
    
    # List of score dicts, one for each response
    score_lst = [reward_item(data_source, r, ground_truth) for r in response_lst]
    
    return data_source, avg_scores_by_fn(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
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
        metric_dict[data_source] = avg_scores_by_fn(rewards)

    print(metric_dict)

    save_path = local_path.replace(".parquet", ".json")
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4)


if __name__ == "__main__":
    main()

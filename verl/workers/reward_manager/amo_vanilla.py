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

from collections import defaultdict
from typing import Any

import torch
import asyncio
import inspect
from functools import partial

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("amo_vanilla")
class AmoVanillaRewardManager(AbstractRewardManager):
    """The multi-object reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score: dict, reward_fn_key="data_source") -> None:
        """
        Initialize the AmoVanillaRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        self.compute_score = compute_score  # Store the compute_score dict of reward functions
        assert isinstance(self.compute_score, dict), "In AmoVanillaLoopManager, compute_score should be a dict of reward functions."
        print(f"[Amo] Using multi-objective reward loop manager with reward functions: {list(self.compute_score.keys())}")
        
        self._loop = None  # Will be set when first async method is called
        self.is_async_reward_score = {
            reward_fn_name: inspect.iscoroutinefunction(reward_fn) 
            for reward_fn_name, reward_fn in self.compute_score.items()
        }
        print(f"[Amo] Reward functions async status: {self.is_async_reward_score}")


    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # [Amo] vanilla solution: weighted sum of all reward functions
        weights: list = data.meta_info["amo_weights"]

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # [Amo] compute individual scores
            single_run_item = asyncio.run(
                self.run_single_async(
                    data_source=data_source,
                    response_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )

            # [Amo] store individual scores and reward extra info
            individual_scores = single_run_item["individual_scores"]
            reward_extra_info_item = single_run_item["reward_extra_info"]
            for key, value in reward_extra_info_item.items():
                reward_extra_info[key].append(value)

            # [Amo] compute weighted sum
            reward = sum(w * s for w, s in zip(weights, individual_scores))

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

                # [Amo] print individual scores
                for reward_fn_name, score in zip(self.compute_score.keys(), individual_scores):
                    print(f"[{reward_fn_name} score]", score)
                print("[reward]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def run_single(self, data_source: str, response_str: str, ground_truth: str, extra_info: dict) -> dict:
        """Run the reward model on a single sample.

        Args:
            data_source: The data source of the sample.
            response_str: The response string of the sample.
            ground_truth: The ground truth answer of the sample.
            extra_info: The extra information of the sample.

        Returns:
            A dictionary containing the reward tensor and the reward extra information.
        """
    
        # [Amo] compute individual scores
        individual_scores = []
        reward_extra_info = {}
        for reward_fn_name, reward_fn in self.compute_score.items():
            result = reward_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # [Amo] Store the information including original reward
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[f"{reward_fn_name}_{key}"] = value
            else:
                score = result
                reward_extra_info[reward_fn_name] = result
            individual_scores.append(score)

        return {
            "individual_scores": individual_scores,
            "reward_extra_info": reward_extra_info,
        }
    
    @property
    def loop(self):
        """Get the current event loop, lazily initializing if needed."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # If no event loop is running, get or create one
                self._loop = asyncio.get_event_loop()
        return self._loop

    async def run_single_async(self, data_source: str, response_str: str, ground_truth: str, extra_info: dict) -> dict:
        """Run the reward model on a single sample.

        Args:
            data_source: The data source of the sample.
            response_str: The response string of the sample.
            ground_truth: The ground truth answer of the sample.
            extra_info: The extra information of the sample.

        Returns:
            A dictionary containing the reward tensor and the reward extra information.
        """
    
        # Step 1: Build a list of concurrent tasks (coroutine/sync run_in_executor)
        tasks = []
        for reward_fn_name, reward_fn in self.compute_score.items():
            if self.is_async_reward_score[reward_fn_name]:
                coro = reward_fn(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            else:
                # Use partial to package parameters to avoid closure issues
                coro = self.loop.run_in_executor(
                    None,
                    partial(
                        reward_fn,
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                )
            tasks.append(coro)

        # Step 2: Await all task results concurrently
        results = await asyncio.gather(*tasks)

        # Step 3: Aggregate results
        individual_scores = []
        reward_extra_info = {}
        for reward_fn_name, result in zip(self.compute_score.keys(), results):
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[f"{reward_fn_name}_{key}"] = value
            else:
                score = result
                reward_extra_info[reward_fn_name] = score
            individual_scores.append(score)

        return {
            "individual_scores": individual_scores,
            "reward_extra_info": reward_extra_info,
        }
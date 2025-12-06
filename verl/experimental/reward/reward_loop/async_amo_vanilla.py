# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import inspect
from functools import partial

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


@register("amo_vanilla")
class AmoVanillaLoopManager(RewardLoopManagerBase):
    """The multi-object reward manager."""

    def __init__(self, config, tokenizer, compute_score: dict, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)

        self.compute_score = compute_score  # Store the compute_score dict of reward functions
        assert isinstance(self.compute_score, dict), "In AmoVanillaLoopManager, compute_score should be a dict of reward functions."
        print(f"[Amo] Using multi-objective reward loop manager with reward functions: {list(self.compute_score.keys())}")
        
        self.is_async_reward_score = {
            reward_fn_name: inspect.iscoroutinefunction(reward_fn) 
            for reward_fn_name, reward_fn in self.compute_score.items()
        }
        print(f"[Amo] Reward functions async status: {self.is_async_reward_score}")

        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        raise NotImplementedError("AmoVanillaLoopManager has not been fully implemented.")
    
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # [Amo] vanilla solution: weighted sum of all reward functions
        weights: list = data.meta_info["amo_weights"]
        assert len(weights) == len(self.compute_score), "The number of weights should be equal to the number of reward functions."

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        # Step 1: Build a list of concurrent tasks (coroutine/sync run_in_executor)
        tasks = []
        reward_fn_names = []
        for reward_fn_name, reward_fn in self.compute_score.items():
            reward_fn_names.append(reward_fn_name)
            if self.is_async_reward_score[reward_fn_name]:
                coro = reward_fn(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
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
                        **extra_reward_kwargs,
                    ),
                )
            tasks.append(coro)

        # Step 2: Await all task results concurrently
        results = await asyncio.gather(*tasks)

        # Step 3: Aggregate results
        individual_scores = []
        reward_extra_info = {}
        for reward_fn_name, result in zip(reward_fn_names, results):
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[f"{reward_fn_name}_{key}"] = value
            else:
                score = result
                reward_extra_info[reward_fn_name] = score
            individual_scores.append(score)

        # [Amo] Step 4: Compute weighted sum
        reward = sum(w * s for w, s in zip(weights, individual_scores))

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

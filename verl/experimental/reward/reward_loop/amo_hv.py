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

import asyncio
from functools import partial

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.amo_vanilla import AmoVanillaRewardLoopManager


@register("amo_hv")
class AmoHVRewardLoopManager(AmoVanillaRewardLoopManager):
    """The multi-object reward manager."""

    def __init__(self, config, tokenizer, compute_score: dict, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)

        self.hv_dict = {}

    async def run_single(self, data: DataProto) -> dict:
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

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # [Amo] vanilla solution: weighted sum of all reward functions
        amo_weights: list = data.meta_info.get("amo_weights", [1.0] * len(self.compute_score))
        assert len(amo_weights) == len(self.compute_score), "The number of weights should be equal to the number of reward functions."

        # [Amo] compute reward for single item
        reward_result = await self.compute_individual_reward(
            data_source=data_source,
            response_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        individual_scores = reward_result["individual_scores"]
        reward_extra_info = reward_result["reward_extra_info"]

        # [Amo] Step 4: Compute weighted sum
        # print(f"[Amo] amo weights: {amo_weights}, individual_scores: {individual_scores}")
        reward = sum(w * s for w, s in zip(amo_weights, individual_scores))

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

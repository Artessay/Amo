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

import asyncio
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_vanilla import AmoVanillaRewardManager

from verl.workers.reward_manager.amo_utils.pareto_cache import ParetoCache
from verl.workers.reward_manager.amo_utils.hybrid_reward import HybridRewardModel

@register("amo_hvpo")
class AmoHvpoRewardManager(AmoVanillaRewardManager):
    """Multi-objective reward manager based on hypervolume (HV) contribution.

    This manager computes per-sample rewards as the incremental contribution of
    each sample to the dominated hypervolume.

    By default, contributions are computed within each ``uid`` group only
    (intra-group HV). When ``hv_config['use_global_pareto_cache']`` is enabled,
    the manager additionally maintains a bounded-size global Pareto front of
    objective vectors and computes rewards as::

        ΔHV(i) = HV(P ∪ {v_i}, r) - HV(P, r),

    where ``P`` is the global Pareto cache and ``r`` is a reference point
    chosen according to ``hv_config['reference_point_strategy']``:

    * ``"dynamic_batch"``: ``r`` is built from the union of the global cache
      and the current group's vectors (min per-dimension minus an optional
      margin, then clamped so it is dominated by all group points).
    * ``"static"``: ``r`` is the user-provided ``hv_config['reference_point']``.

    The cache behaviour is controlled via:

    * ``use_global_pareto_cache`` (bool): enable/disable global mode.
    * ``pareto_cache_max_size`` (int): maximum number of stored Pareto points.
    * ``pareto_cache_eps`` (float): dominance tolerance.
    * ``pareto_cache_strategy`` (str): eviction strategy (currently only
      ``"fifo"``, keeping the most recent non-dominated points).
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: dict,
        reward_fn_key: str = "data_source",
        hv_config: dict | None = None,
        **_: Any,
    ) -> None:
        """Initialize the AmoHvRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of examples to print for debugging.
            compute_score: Dict of reward functions (multi-objective).
            reward_fn_key: Key used to access the data source in
                ``non_tensor_batch``. Defaults to "data_source".
            hv_config: Configuration dict for HV-based reward shaping.
        """
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)

        # HV configuration
        hv_config = dict(hv_config) if hv_config is not None else {}
        self.hv_config: dict[str, Any] = hv_config

        print(f"[Amo][HV] Using HV reward manager with hv_config: {self.hv_config}")

        # config
        self.reward_scaling_mode: str = hv_config.get("reward_scaling_mode")

        self.distance_metric: str = hv_config.get("distance_metric")
        assert self.distance_metric in ["chebyshev", "manhattan", "euclidean", "none"]

        self._configure_reference_point()
        self._configure_pareto_cache()
        

    # ------------------------------------------------------------------
    # Init methods
    # ------------------------------------------------------------------
    def _configure_reference_point(self) -> None:
        """Configure reference point based on the selected strategy.

        Args:
            hv_config: Configuration dict for HV-based reward shaping.
        """
        hv_config: dict = self.hv_config

        # Reference point configuration
        self.reference_point_strategy: str = hv_config.get(
            "reference_point_strategy", "static"    # "dynamic_batch" or "static"
        )

        if self.reference_point_strategy == "static":
            self.reference_point = hv_config.get("reference_point", None)
            if self.reference_point is None:
                self.reference_point = [0] * len(self.compute_score)
                print(f"[Amo][HV] reference_point is set to {self.reference_point}")
            
            # check reference_point length
            if len(self.reference_point) != len(self.compute_score):
                print(f"[Amo][HV] reference_point: {self.reference_point}")
                raise ValueError(
                    f"[Amo][HV] reference_point length {len(self.reference_point)} "
                    f"does not match compute_score dimension {len(self.compute_score)}."
                )
        elif self.reference_point_strategy == "dynamic_batch":
            # Margin to subtract from min objective values when using dynamic reference points
            self.reference_point_margin: float = float(hv_config.get("reference_point_margin", 0.0))
        else:
            raise ValueError(
                f"[Amo][HV] reference_point_strategy {self.reference_point_strategy} "
                "is not supported. Please choose 'static' or 'dynamic_batch'."
            )

    def _configure_pareto_cache(self) -> None:
        """Configure Pareto cache based on the selected strategy.

        Args:
            hv_config: Configuration dict for HV-based reward shaping.
        """
        hv_config: dict = self.hv_config

        # Pareto cache configuration
        self.pareto_cache_max_size: int = int(hv_config.get("pareto_cache_max_size", 1024))
        self.pareto_cache_eps: float = float(hv_config.get("pareto_cache_eps", 1e-9))
        self.pareto_cache_strategy: str = hv_config.get("pareto_cache_strategy", "fifo")
        
        # Create ParetoCache instance
        self.pareto_cache = ParetoCache(
            max_size=self.pareto_cache_max_size,
            eps=self.pareto_cache_eps,
            strategy=self.pareto_cache_strategy
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:

        # If there is rm score, we directly return rm score. Otherwise, we compute
        # rewards via the multi-objective HV contribution.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        batch_size = len(data)
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        # First pass: compute individual multi-objective scores per sample and
        # gather metadata needed for group-wise HV computation.
        individual_scores_list: list[list[float]] = []
        data_sources: list[str] = []
        uids: list[str] = []
        data_splits: list[str] = []
        valid_response_lengths: list[int] = []
        prompt_strs: list[str] = []
        response_strs: list[str] = []
        ground_truths: list[Any] = []

        already_print_data_sources: dict[str, int] = {}

        for i in range(batch_size):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
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

            # [Amo] compute individual scores (one vector per sample)
            single_run_item = asyncio.run(
                self.compute_individual_reward(
                    data_source=data_source,
                    response_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )

            individual_scores = single_run_item["individual_scores"]
            reward_extra_info_item = single_run_item["reward_extra_info"]
            for key, value in reward_extra_info_item.items():
                reward_extra_info[key].append(value)

            # Store for HV computation
            individual_scores_list.append([float(s) for s in individual_scores])
            data_sources.append(data_source)

            # Try to generate a stable uid from extra_info first
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            # Use split and index to generate stable uid for the same prompt
            split = extra_info.get("split")
            assert split in ["train", "val", "test"], f"split should be 'train', 'val', or 'test', but got {split}"

            # # Try to get index from extra_info, which should be the same for all responses from the same prompt
            # index = extra_info.get("index")
            # assert index is not None, "index should not be None"
            # # Generate a stable uid based on split and index
            # uid = f"{split}_{index}"
            
            # Use uid from non_tensor_batch 
            uid = data_item.non_tensor_batch.get("uid")
            assert uid is not None, "uid should not be None"

            uids.append(uid)
            data_splits.append(split)
            valid_response_lengths.append(valid_response_length)
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(ground_truth)

        if not individual_scores_list:
            # Empty batch or no objectives; return zero reward tensor.
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor

        # ------------------------------------------------------------------
        # HV computation
        # ------------------------------------------------------------------
        score_tensor = torch.tensor(individual_scores_list, dtype=torch.float32)    # (batch_size, num_objectives)

        # Get Pareto cache snapshot for this batch
        pareto_tensor = self._get_pareto_cache_snapshot(score_tensor)

        # Determine reference point for this group
        ref_point = self._compute_reference_point(score_tensor, pareto_tensor)

        hybrid_rewards = torch.zeros(len(uids), dtype=torch.float32)
        hv_contributions = torch.zeros(len(uids), dtype=torch.float32)
        distance_penalty = torch.zeros(len(uids), dtype=torch.float32)

        assert all(split == data_splits[0] for split in data_splits), "All elements in data_splits should be the same"
        need_estimate_pareto_front: bool = data_splits[0] != "train"

        if need_estimate_pareto_front:
            # update pareto cache
            pareto_cache_point = score_tensor.mean(dim=0).tolist()
            self.pareto_cache.update(pareto_cache_point)
            print(f"[Amo][HV] Added point {pareto_cache_point} to Pareto cache, current size: {self.pareto_cache.size()}")

            # calculate reward through mean of individual scores
            hybrid_rewards = score_tensor.mean(dim=1)
            
        # Group indices by uid
        uid2indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid2indices[uid].append(idx)
        # print(f"[Amo][HV] uid2indices: {uid2indices}")

        for group_uid, indices in uid2indices.items():
            # group_size is equal to actor_rollout_ref.rollout.n for train and actor_rollout_ref.rollout.val_kwargs.n for val
            group_scores = score_tensor[indices]  # (group_size, dim)
            if group_scores.numel() == 0:
                continue

            # Compute HV contributions against the global Pareto cache.
            contributions = self._compute_hybrid_reward(
                group_scores, ref_point, pareto_tensor
            )
            assert contributions.shape == (len(indices),)

            contributions = self._scale_contributions(contributions, self.reward_scaling_mode)

            # Write rewards to the last token position and fill extra info
            for local_idx, global_idx in enumerate(indices):
                contribution = contributions[local_idx]
                
                if not need_estimate_pareto_front:
                    hybrid_rewards[global_idx] = contribution
                hv_contributions[global_idx] = contribution if contribution > 0 else 0.0
                distance_penalty[global_idx] = contribution if contribution < 0 else 0.0
                # reference_points_per_sample[global_idx] = ref_point.tolist()

        # ------------------------------------------------------------------
        # Debug printing (keep original behavior controlled by num_examine)
        # ------------------------------------------------------------------
        for i in range(batch_size):
            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])

                # Print individual scores for interpretability
                for reward_fn_name, score in zip(self.compute_score.keys(), individual_scores_list[i]):
                    print(f"[{reward_fn_name} score]", score)
                print("[hybrid_rewards]", hybrid_rewards[i].item())
                print("[hv_contribution]", hv_contributions[i].item())
                print("[distance_penalty]", distance_penalty[i].item())

            # Write hybrid rewards to the last token position
            valid_response_length = valid_response_lengths[i]
            assert valid_response_length > 0, f"valid_response_lengths[{i}] = {valid_response_length}"
            reward_tensor[i, valid_response_length - 1] = hybrid_rewards[i]

            # Attach HV-related extra information (aligned with sample order)
            reward_extra_info["hybrid_rewards"].append(hybrid_rewards[i].item())
            reward_extra_info["hv_contribution"].append(hv_contributions[i].item())
            reward_extra_info["distance_penalty"].append(distance_penalty[i].item())
            # reward_extra_info["reference_point"].append(reference_points_per_sample[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------    @staticmethod
    def _compute_hybrid_reward(
        self,
        group_vectors: torch.Tensor,
        ref_point: torch.Tensor,
        pareto_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid reward: ΔHV or distance penalty for a batch points.
        The calculation for a single point is given by HybridRewardModel
        
        Args:
            group_vectors: Objective vectors for the group.
            ref_point: Reference point for hypervolume calculation.
            pareto_vectors: Pareto front vectors.
        
        Returns:
            Hybrid reward tensor for each point.
        """
        group_size, dim = group_vectors.shape

        # Compute hybrid reward for each point in the group
        rewards = []
        for point in group_vectors:
            reward = HybridRewardModel.compute_hybrid_reward(
                point, pareto_vectors, ref_point,
                distance_metric=self.distance_metric,
            )
            rewards.append(reward)
        
        # Return tensor with rewards for each point
        trajectory_rewards = torch.stack(rewards)
        assert trajectory_rewards.shape == (group_size,), f"[Amo][HV] Hybrid reward shape mismatch: {trajectory_rewards.shape}"

        return trajectory_rewards
        

    def _compute_reference_point(self, group_scores: torch.Tensor, pareto_tensor: torch.Tensor) -> torch.Tensor:
        """Compute reference point for hypervolume calculation.

        Args:
            group_scores: Objective vectors for the group.
            pareto_tensor: Pareto front vectors.

        Returns:
            Reference point tensor.
        """
        group_min = group_scores.min(dim=0).values
        if self.reference_point_strategy == "dynamic_batch":
            if pareto_tensor.numel() > 0:
                # Use the union of the global Pareto frontier and the current group's
                # objective vectors to determine the reference point, then clamp so
                # that it is dominated by all group points.
                all_points = torch.cat([group_scores, pareto_tensor], dim=0)
                union_min = all_points.min(dim=0).values
                ref_point = union_min - self.reference_point_margin
            else:
                ref_point = group_min - self.reference_point_margin
        elif self.reference_point_strategy == "static":
            ref_point = torch.tensor(self.reference_point, dtype=group_scores.dtype, device=group_scores.device)
        else:
            raise ValueError(
                f"[Amo][HV] Unsupported reference_point_strategy: {self.reference_point_strategy}"
            )

        # Ensure reference point is dominated by all objective vectors in this group
        ref_point = torch.minimum(ref_point, group_min)
        return ref_point

    def _get_pareto_cache_snapshot(self, score_tensor: torch.Tensor) -> torch.Tensor:
        """Get a snapshot of the global Pareto cache and convert it to a tensor.

        The snapshot is read-only while computing rewards so that all samples in the batch
        see a consistent frontier.

        Args:
            score_tensor: Tensor of objective scores to match dtype and device.

        Returns:
            Tensor containing the Pareto cache points, or an empty tensor if the cache is empty.
        """
        # Take a snapshot of the global Pareto cache for this batch
        pareto_cache_snapshot: list[list[float]] = self.pareto_cache.get_snapshot()
        print(f"[Amo][HV] Pareto cache snapshot size: {len(pareto_cache_snapshot)}")
        print(f"[Amo][HV] Pareto cache snapshot: {pareto_cache_snapshot}")

        # Prepare Pareto cache tensor for this group (if enabled)
        if len(pareto_cache_snapshot) > 0:
            pareto_tensor = torch.tensor(
                pareto_cache_snapshot,
                dtype=score_tensor.dtype,
                device=score_tensor.device,
            )
            if pareto_tensor.shape[1] != score_tensor.shape[1]:
                raise ValueError(
                    f"[Amo][HV] Pareto cache dimension mismatch: got {pareto_tensor.shape[1]}, expected {score_tensor.shape[1]}."
                )
        else:
            # pareto_tensor is empty, create a zero tensor with the same dtype and device as score_tensor
            pareto_tensor = score_tensor.new_zeros((0, score_tensor.shape[1]))
            assert pareto_tensor.numel() == 0, "pareto_tensor should be empty"
        return pareto_tensor

    @staticmethod
    def _scale_contributions(contribs: torch.Tensor, mode: str) -> torch.Tensor:
        """Scale HV contributions within a group.

        Only the *contributions* are scaled; the objective vectors themselves
        are left untouched to avoid affecting HV geometry.
        """
        if contribs.numel() == 0:
            return contribs

        if mode == "none":
            return contribs
        if mode == "min-max":
            c_min = contribs.min()
            c_max = contribs.max()
            if (c_max - c_min) <= 0:
                return torch.zeros_like(contribs)
            return (contribs - c_min) / (c_max - c_min + 1e-8)
        if mode == "z-score":
            mean = contribs.mean()
            std = contribs.std(unbiased=False)
            if std <= 0:
                return contribs - mean
            return (contribs - mean) / (std + 1e-8)
        if mode == "tanh":
            return torch.tanh(contribs)

        raise ValueError(f"[Amo][HV] Unsupported reward_scaling_mode: {mode}")
